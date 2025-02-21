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
    def __init__(self, input_path: str):
        self.doc = fitz.open(input_path)
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
            width, height = img.size
            scale = min(512 / width, 512 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(image_path)
        
        return image_path
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _extract_content_from_images(self, image_paths: List[str], page_nums: List[int]) -> List[Dict]:
        """Extract text content from multiple images using Gemini Vision with retry mechanism"""
        try:
            # Upload images to Gemini (max 10 images)
            images = [genai.upload_file(path, mime_type="image/png") for path in image_paths]
            
            # Create prompt for content extraction
            prompt = """Wyodrębnij i przetłumacz tekst z tych obrazów na język polski, formatując zawartość każdego obrazu jako Markdown. Dla każdego obrazu:

1. Dokładnie wyodrębnij cały widoczny tekst
2. Przetłumacz tekst na język polski
3. Sformatuj zawartość jako JSON z następującą strukturą:
{
    „chapters": [
        {
            „chapter_name": „Nazwa rozdziału (jeśli występuje, w przeciwnym razie ' ')”,
            „text”: „Przetłumaczony tekst w formacie Markdown”
        }
    ]
}

Ważne:
- Wyodrębnij i przetłumacz WSZYSTKIE tekst dokładnie na język polski na najwyższym poziomie
- Używaj poprawnego formatowania Markdown:
  * Użyj # dla głównych nagłówków
  * Użyj ## dla podtytułów
  * Używaj odpowiedniego formatowania list (-, *, liczby)
  * Zachowaj podkreślenia (*italic*, **bold**), jeśli są obecne.
  * Używaj > dla cytatów lub ważnego tekstu
  * Używaj odpowiednich odstępów między akapitami
- Nie podsumowuj ani nie interpretuj
- Przetwarzanie każdego obrazu osobno

Zwróć tablicę obiektów JSON, po jednym na obraz, w tej samej kolejności co obrazy wejściowe."""           
            # Send request to Gemini with all images
            response = self.chat.send_message([prompt] + images)
            time.sleep(45)  # Sleep to avoid rate limiting
            
            # Parse JSON response and clean up text
            contents = json.loads(response.text.strip().replace('```json\n', '').replace('\n```', ''))
            if not isinstance(contents, list):
                contents = [contents]  # Handle single response case
            
            # Clean up each chapter's text
            for content in contents:
                for chapter in content.get('chapters', []):
                    if chapter.get('text'):
                        chapter['text'] = re.sub(r'\nZdigitalizowane przez Google', '', chapter['text'])
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
                raise
            elif "RECITATION" in error_str:
                logger.warning(f"Recitation error detected on pages {page_nums}, retrying with modified prompt...")
                time.sleep(45)  # Cooldown before retry
                raise
            logger.error(f"Error extracting content from pages {page_nums}: {error_str}")
            return [{
                "chapters": [{"chapter_name": "Error", "text": ""}]
            } for _ in range(len(image_paths))]
    
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
                raise
            logger.error(f"Translation error on page {page_num}: {str(e)}")
            return text

    def process_document(self, max_workers: int = 10, test_mode: bool = False) -> List[PageContent]:
        """Process the entire document with parallel processing and integrated translation"""
        pages_content = []
        total_pages = min(10, len(self.doc)) if test_mode else len(self.doc)
        batch_size = 5  # Reduced batch size to minimize RECITATION errors
        error_occurred = False
        failed_pages = set()
        
        def process_page_batch(page_nums: List[int]) -> List[PageContent]:
            try:
                unprocessed_pages = []
                image_paths = []
                
                for page_num in page_nums:
                    if any(p.page_number == page_num for p in pages_content):
                        logger.info(f"Skipping already processed page {page_num}")
                        continue
                    
                    image_path = self._save_page_as_image(page_num)
                    image_paths.append(image_path)
                    unprocessed_pages.append(page_num)
                
                if not unprocessed_pages:
                    return []
                
                contents = self._extract_content_from_images(image_paths, unprocessed_pages)
                
                results = []
                for page_num, content, image_path in zip(unprocessed_pages, contents, image_paths):
                    if content.get('chapters', [{}])[0].get('chapter_name') == 'Error':
                        failed_pages.add(page_num)
                        logger.warning(f"Page {page_num} failed to process, will retry later")
                        continue
                    
                    result = PageContent(
                        page_number=page_num,
                        content=content,
                        image_path=image_path
                    )
                    results.append(result)
                
                return results
                
            except Exception as e:
                nonlocal error_occurred
                error_occurred = True
                logger.error(f"Error processing pages {page_nums}: {str(e)}")
                for page_num in page_nums:
                    failed_pages.add(page_num)
                return []
        
        # Process pages in smaller batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"\nProcessing batch {batch_start//batch_size + 1} (pages {batch_start+1}-{batch_end})")
            try:
                page_nums = list(range(batch_start, batch_end))
                results = process_page_batch(page_nums)
                pages_content.extend(results)
            except Exception as e:
                error_occurred = True
                logger.error(f"Error in batch processing: {str(e)}")
            
            if error_occurred and batch_end < total_pages:
                logger.info("Error detected. Taking a 60-second break before continuing...")
                time.sleep(60)
                error_occurred = False
            else:
                logger.info("Taking a 30-second break between batches...")
                time.sleep(30)
        
        # Retry failed pages
        while failed_pages:
            logger.info(f"Retrying {len(failed_pages)} failed pages after cooldown...")
            time.sleep(60)  # Wait before retrying
            
            retry_pages = list(failed_pages)
            failed_pages.clear()
            
            for i in range(0, len(retry_pages), batch_size):
                batch = retry_pages[i:i + batch_size]
                results = process_page_batch(batch)
                pages_content.extend(results)
                time.sleep(30)  # Cooldown between retry batches
        
        pages_content.sort(key=lambda x: x.page_number)
        return pages_content
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

def save_as_pdf(pages_content: List[PageContent], output_path: str):
    """Save translated content as PDF with proper Polish character support using markdown-pdf"""
    from markdown_pdf import MarkdownPdf, Section
    import os

    # Create PDF document with TOC support
    pdf = MarkdownPdf(toc_level=2)

    # Configure PDF metadata and font size
    pdf.meta.update({
        "producer": "Transl8",
        "creator": "Transl8 Image Translator",
        "title": "Translated Document",
    })
    pdf.font_size = 15  # Set base font size to 15 points

    # Process each page's content
    for page in pages_content:
        markdown_content = ""
        
        for chapter in page.content.get('chapters', []):
            chapter_name = chapter.get('chapter_name', '').strip()
            text = chapter.get('text', '').strip()
            
            # Add chapter title if available
            if chapter_name and chapter_name != ' ':
                markdown_content += f"# {chapter_name}\n\n"
            
            # Add chapter text
            if text:
                markdown_content += f"{text}\n\n"
        
        # Create a new section for each page
        if markdown_content:
            section = Section(
                markdown_content,
                paper_size="A4",  # Use A4 paper size
                borders=(36, 36, -36, -36)  # Set reasonable margins
            )
            pdf.add_section(section)
    
    # Save the PDF file
    pdf.save(output_path)
    logger.info(f"PDF generated successfully at {output_path} with Polish character support")
    
    # Verify the PDF contains text and not images
    try:
        output_doc = fitz.open(output_path)
        first_page_text = output_doc[0].get_text()
        if first_page_text:
            logger.info(f"PDF verification: Text found in output PDF")
            # Check for common Polish characters
            polish_chars = 'ąćęłńóśźż'
            missing_chars = []
            for char in polish_chars:
                if char not in first_page_text and char.upper() not in first_page_text:
                    missing_chars.append(char)
            
            if missing_chars:
                logger.warning(f"Polish characters possibly missing: {', '.join(missing_chars)}")
            else:
                logger.info("Polish character verification passed")
        else:
            logger.warning("PDF verification: No text found in output PDF")
        output_doc.close()
    except Exception as e:
        logger.warning(f"Could not verify PDF text content: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process PDF using image-based text extraction")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--test", action="store_true", help="Enable test mode (process only first 10 pages)")
    args = parser.parse_args()
    
    try:
        translator = ImageBasedTranslator(args.input)
        pages_content = translator.process_document(test_mode=args.test)
        save_as_pdf(pages_content, args.output)
        logger.info(f"Processing complete! Output saved to: {args.output}")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        translator.cleanup()

if __name__ == "__main__":
    main()
