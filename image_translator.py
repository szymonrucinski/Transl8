import fitz
import os
import re
import google.generativeai as genai
import json
import tempfile
import pickle
from PIL import Image
import logging
from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class PageContent:
    """Represents the content of a single page"""
    page_number: int
    content: Dict
    image_path: str

class ImageBasedTranslator:
    def __init__(self, input_path: str, target_lang: str = None):
        self.doc = fitz.open(input_path)
        self.target_lang = target_lang
        self.temp_dir = tempfile.mkdtemp()
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 16384,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
        )
        self.chat = self.model.start_chat()
    
    def _save_page_as_image(self, page_num: int) -> str:
        """Save a PDF page as an image with size constraints"""
        page = self.doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        image_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
        pix.save(image_path)
        
        # Resize image to fit within 512x512 while maintaining aspect ratio
        with Image.open(image_path) as img:
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            scale = min(512 / width, 512 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(image_path)
        
        return image_path
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _extract_content_from_images(self, image_paths: List[str], page_nums: List[int]) -> List[Dict]:
        """Extract text content from multiple images using Gemini Vision with retry mechanism"""
        try:
            # Upload images to Gemini (max 10 images)
            images = [genai.upload_file(path, mime_type="image/png") for path in image_paths]
            
            # Create prompt for content extraction with anti-recitation guidance
            prompt = """Extract and structure the text content from these images, following these guidelines:

            1. For each image, identify and extract:
               - Main headings and subheadings
               - Body text
               - Any special formatting (lists, quotes, etc.)

            2. Format the content as JSON with this structure:
            {
                "chapters": [
                    {
                        "chapter_name": "Detected heading or 'Content' if no heading present",
                        "text": "Full text content with preserved formatting"
                    }
                ]
            }

            3. Important instructions:
               - Preserve the original text exactly as it appears
               - Maintain paragraph breaks and formatting
               - Don't summarize or rephrase the content
               - Don't add any interpretations or explanations
               - Handle each image independently
               - Focus on extracting text content only, avoid repeating or reciting patterns
               - If text appears unclear or repetitive, mark it as '[unclear]'

            Return an array of JSON objects, one for each image, maintaining the exact order of input images."""
            
            # Send request to Gemini with all images
            response = self.chat.send_message([prompt] + images)
            time.sleep(60)  # Increased rate limiting delay
            
            # Parse JSON response and clean up text
            contents = json.loads(response.text.strip().replace('```json\n', '').replace('\n```', ''))
            if not isinstance(contents, list):
                contents = [contents]  # Handle single response case
            
            # Clean up each chapter's text
            for content in contents:
                for chapter in content.get('chapters', []):
                    if chapter.get('text'):
                        chapter['text'] = re.sub(r'\nDigitized by Google', '', chapter['text'])
                        # Remove any repetitive patterns
                        chapter['text'] = re.sub(r'(\b\w+(?:\s+\w+){2,}?)\s*(?:\1\s*)+', r'\1', chapter['text'])
            
            # Print extracted content for each page
            for page_num, content in zip(page_nums, contents):
                print(f"\nExtracted content from page {page_num}:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
            
            return contents
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Resource has been exhausted" in error_str:
                logger.warning(f"Rate limit hit on pages {page_nums}, retrying with exponential backoff...")
                time.sleep(60)  # Additional cooldown before retry
                raise  # This will trigger the retry mechanism
            elif "RECITATION" in error_str:
                logger.warning(f"Recitation error detected on pages {page_nums}, retrying with modified prompt...")
                time.sleep(45)  # Cooldown before retry
                raise  # This will trigger the retry mechanism
            logger.error(f"Error extracting content from pages {page_nums}: {error_str}")
            return [{
                "chapters": [{"chapter_name": "Error", "text": ""}]
            } for _ in range(len(image_paths))]
            
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                logger.warning(f"Rate limit hit on page {page_num}, retrying with exponential backoff...")
                raise  # This will trigger the retry mechanism
            logger.error(f"Error extracting content from page {page_num}: {str(e)}")
            return {"chapters": [{"chapter_name": "Error", "text": ""}]}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _translate_text(self, text: str, page_num: int) -> str:
        """Translate text with retry mechanism"""
        try:
            prompt = f"""Translate the following text to {self.target_lang}:
            
            {text}
            
            Preserve any formatting or special characters."""
            
            response = self.chat.send_message(prompt)
            time.sleep(30)  # Rate limiting
            return response.text.strip()
            
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                logger.warning(f"Rate limit hit during translation on page {page_num}, retrying with exponential backoff...")
                raise  # This will trigger the retry mechanism
            logger.error(f"Translation error on page {page_num}: {str(e)}")
            return text  # Return original text on non-quota errors

    def process_document(self, max_workers: int = 10, test_mode: bool = False) -> List[PageContent]:
        """Process the entire document with parallel processing and integrated translation"""
        pages_content = []
        total_pages = min(10, len(self.doc)) if test_mode else len(self.doc)
        batch_size = 5  # Reduced batch size to minimize RECITATION errors
        error_occurred = False
        # progress_file = f'translation_progress_{int(time.time())}.pkl'
        
        # Load progress if exists
        # if os.path.exists(progress_file):
        #     try:
        #         with open(progress_file, 'rb') as f:
        #             pages_content = pickle.load(f)
        #         logger.info(f"Resuming from page {len(pages_content)}")
        #     except Exception as e:
        #         logger.error(f"Error loading progress: {str(e)}")
        
        def process_page_batch(page_nums: List[int]) -> List[PageContent]:
            try:
                # Skip already processed pages
                unprocessed_pages = []
                image_paths = []
                
                for page_num in page_nums:
                    if any(p.page_number == page_num for p in pages_content):
                        logger.info(f"Skipping already processed page {page_num}")
                        continue
                    
                    # Save page as image
                    image_path = self._save_page_as_image(page_num)
                    image_paths.append(image_path)
                    unprocessed_pages.append(page_num)
                
                if not unprocessed_pages:
                    return []
                
                # Extract content for all images in batch
                contents = self._extract_content_from_images(image_paths, unprocessed_pages)
                
                results = []
                for page_num, content, image_path in zip(unprocessed_pages, contents, image_paths):
                    result = PageContent(
                        page_number=page_num,
                        content=content,
                        image_path=image_path
                    )
                    results.append(result)
                
                # Save progress after batch
                # with open(progress_file, 'wb') as f:
                #     pickle.dump(pages_content + results, f)
                
                return results
                
            except Exception as e:
                nonlocal error_occurred
                error_occurred = True
                logger.error(f"Error processing pages {page_nums}: {str(e)}")
                return [PageContent(
                    page_number=page_num,
                    content={"chapters": [{"chapter_name": "Error", "text": ""}]},
                    image_path=""
                ) for page_num in page_nums]
        
        # Process remaining pages in smaller batches
        start_page = len(pages_content)
        for batch_start in range(start_page, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"\nProcessing batch {batch_start//batch_size + 1} (pages {batch_start+1}-{batch_end})")
            
            # Process current batch
            try:
                page_nums = list(range(batch_start, batch_end))
                results = process_page_batch(page_nums)
                pages_content.extend(results)
            except Exception as e:
                error_occurred = True
                logger.error(f"Error in batch processing: {str(e)}")
            
            # Take a longer break if an error occurred
            if error_occurred and batch_end < total_pages:
                logger.info("Error detected. Taking a 60-second break before continuing...")
                time.sleep(60)
                error_occurred = False
            else:
                # Normal delay between batches
                logger.info("Taking a 30-second break between batches...")
                time.sleep(30)
        
        # Sort results by page number
        pages_content.sort(key=lambda x: x.page_number)
        
        # # Clean up progress file
        # if os.path.exists(progress_file):
        #     os.remove(progress_file)
            
        return pages_content
    
    def translate_content(self, pages_content: List[PageContent]) -> List[PageContent]:
        """Translate extracted content in batches of 10"""
        if not self.target_lang:
            return pages_content

        batch_size = 10
        total_pages = len(pages_content)

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch = pages_content[batch_start:batch_end]

            logger.info(f"\nTranslating batch {batch_start//batch_size + 1} (pages {batch_start+1}-{batch_end})")

            for page in tqdm(batch, desc=f"Translating batch {batch_start//batch_size + 1}"):
                try:
                    chapters = page.content.get("chapters", [])
                    for chapter in chapters:
                        # Skip empty text content
                        if not chapter['text'] or chapter['text'].strip() == "":
                            logger.warning(f"Skipping empty text content in page {page.page_number}")
                            continue

                        prompt = f"""Translate the following text to {self.target_lang}, following these guidelines:

                        1. Maintain all formatting elements:
                           - Preserve paragraph breaks
                           - Keep bullet points and numbered lists
                           - Retain any special characters or symbols

                        2. Translation rules:
                           - Maintain the original tone and style
                           - Keep proper nouns unchanged
                           - Preserve any technical terms in their original form
                           - Ensure numbers and dates remain in the same format

                        Text to translate:
                        {chapter['text']}

                        Provide only the translated text without any explanations or comments."""

                        response = self.chat.send_message(prompt)
                        time.sleep(30)  # Rate limiting
                        chapter['text'] = response.text.strip()

                except Exception as e:
                    logger.error(f"Error translating page {page.page_number}: {str(e)}")

            # Take a break between batches
            if batch_end < total_pages:
                logger.info("Taking a 30-second break between batches...")
                time.sleep(30)
                error_occurred = False

        return pages_content
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

def save_as_pdf(pages_content: List[PageContent], output_path: str):
    """Save translated content as PDF with one page per translated content"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    import os
    
    # Create PDF document with Unicode font support
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Register DejaVu Sans font for Unicode support
    font_path = "/Users/szymon/Library/fonts/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/Arial Unicode.ttf"  # Fallback for macOS
    
    try:
        pdfmetrics.registerFont(TTFont('CustomFont', font_path))
        font_name = 'CustomFont'
    except Exception as e:
        logger.warning(f"Could not register custom font: {str(e)}. Falling back to built-in fonts.")
        font_name = 'Helvetica'
    
    # Configure styles with Unicode support
    styles.add(ParagraphStyle(
        name='ChapterTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        fontName=font_name,
        encoding='utf-8'
    ))
    
    # Configure normal text style with Unicode support
    styles['Normal'].fontSize = 12
    styles['Normal'].fontName = font_name
    styles['Normal'].encoding = 'utf-8'
    
    # Add content with page breaks
    for page in pages_content:
        for chapter in page.content.get('chapters', []):
            # Add chapter title
            story.append(Paragraph(chapter['chapter_name'], styles['ChapterTitle']))
            story.append(Spacer(1, 12))
            
            # Add chapter content
            story.append(Paragraph(chapter['text'], styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Add page break after each page's content
        story.append(PageBreak())
    
    # Build PDF
    doc.build(story)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF using image-based translation")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--target-lang", default="pl", help="Target language code (default: pl for Polish)")
    parser.add_argument("--test", action="store_true", help="Enable test mode (process only first 10 pages)")
    
    args = parser.parse_args()
    
    try:
        # Initialize translator
        translator = ImageBasedTranslator(args.input, args.target_lang)
        
        # Process document
        pages_content = translator.process_document(test_mode=args.test)
        
        # Translate content
        pages_content = translator.translate_content(pages_content)
        
        # Save as PDF
        save_as_pdf(pages_content, args.output)
        
        logger.info(f"Processing complete! Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        translator.cleanup()

if __name__ == "__main__":
    main()