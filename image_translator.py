import fitz
import os
import re
import google.generativeai as genai
import json
import tempfile
import logging
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

@dataclass
class PageContent:
    """Represents the content of a single page"""
    page_number: int
    content: Dict
    image_path: str

@dataclass
class TranslatorConfig:
    """Configuration for the translator"""
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 16384
    target_lang: str = "polski"  # Default target language
    batch_size: int = 5
    retry_attempts: int = 3
    dpi: int = 300
    max_image_size: Tuple[int, int] = (512, 512)
    cooldown_time: int = 45
    retry_cooldown: int = 60
    batch_cooldown: int = 30

class RateLimitException(Exception):
    """Exception raised for API rate limit errors"""
    pass

class RecitationException(Exception):
    """Exception raised for model recitation errors"""
    pass

class ImageBasedTranslator:
    def __init__(self, input_path: str, config: Optional[TranslatorConfig] = None):
        """
        Initialize the translator with a PDF document and configuration.
        
        Args:
            input_path: Path to the input PDF file
            config: Optional configuration, uses default if not provided
        """
        self.doc = fitz.open(input_path)
        self.temp_dir = tempfile.mkdtemp()
        self.config = config or TranslatorConfig()
        
        # Configure the Gemini model
        self.generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
        }
        
        # Initialize the model and chat
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=self.generation_config,
        )
        self.chat = self.model.start_chat()
    
    def _save_page_as_image(self, page_num: int) -> str:
        """
        Save a PDF page as an image with size constraints.
        
        Args:
            page_num: The page number to process
            
        Returns:
            Path to the saved image
        """
        page = self.doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.config.dpi/72, self.config.dpi/72))
        image_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
        pix.save(image_path)
        
        # Resize image to fit within max_image_size while maintaining aspect ratio
        max_width, max_height = self.config.max_image_size
        with Image.open(image_path) as img:
            width, height = img.size
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(image_path)
        
        return image_path
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitException, RecitationException))
    )
    def _extract_content_from_images(self, image_paths: List[str], page_nums: List[int]) -> List[Dict]:
        """
        Extract text content from multiple images using Gemini Vision with retry mechanism.
        
        Args:
            image_paths: List of paths to images
            page_nums: List of corresponding page numbers
            
        Returns:
            List of dictionaries containing extracted content
        """
        try:
            # Upload images to Gemini
            images = [genai.upload_file(path, mime_type="image/png") for path in image_paths]
            
            # Create prompt for content extraction
            prompt = self._create_extraction_prompt()
            
            # Send request to Gemini with all images
            response = self.chat.send_message([prompt] + images)
            time.sleep(self.config.cooldown_time)  # Sleep to avoid rate limiting
            
            # Parse JSON response and clean up text
            try:
                content_text = response.text.strip()
                # Extract JSON from code blocks if present
                if "```json" in content_text:
                    content_text = re.search(r'```json\n(.*?)\n```', content_text, re.DOTALL)
                    if content_text:
                        content_text = content_text.group(1)
                
                contents = json.loads(content_text)
                if not isinstance(contents, list):
                    contents = [contents]  # Handle single response case
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Raw response: {response.text}")
                # Attempt to recover by creating a simple structure
                contents = [{"chapters": [{"chapter_name": "Error", "text": response.text}]} for _ in range(len(image_paths))]
            
            # Clean up each chapter's text
            for content in contents:
                for chapter in content.get('chapters', []):
                    if chapter.get('text'):
                        chapter['text'] = self._clean_text(chapter['text'])
            
            # Print extracted content for each page (limited to first chapter for brevity)
            for page_num, content in zip(page_nums, contents):
                first_chapter = content.get('chapters', [{}])[0]
                logger.info(f"Extracted content from page {page_num}: {first_chapter.get('chapter_name', 'N/A')} (content length: {len(first_chapter.get('text', ''))} chars)")
            
            return contents
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Resource has been exhausted" in error_str:
                logger.warning(f"Rate limit hit on pages {page_nums}, retrying with exponential backoff...")
                time.sleep(self.config.retry_cooldown)  # Additional cooldown before retry
                raise RateLimitException(f"Rate limit hit: {error_str}")
            elif "RECITATION" in error_str:
                logger.warning(f"Recitation error detected on pages {page_nums}, retrying with modified prompt...")
                time.sleep(self.config.cooldown_time)  # Cooldown before retry
                raise RecitationException(f"Recitation error: {error_str}")
            
            logger.error(f"Error extracting content from pages {page_nums}: {error_str}")
            return [{
                "chapters": [{"chapter_name": "Error", "text": ""}]
            } for _ in range(len(image_paths))]
    
    def _create_extraction_prompt(self) -> str:
        """Create the prompt for content extraction"""
        return """Wyodrębnij i przetłumacz tekst z tych obrazów na język polski, formatując zawartość każdego obrazu jako Markdown. Dla każdego obrazu:

1. Dokładnie wyodrębnij cały widoczny tekst
2. Przetłumacz tekst na język polski
3. Sformatuj zawartość jako JSON z następującą strukturą:
{
    "chapters": [
        {
            "chapter_name": "Nazwa rozdziału (jeśli występuje, w przeciwnym razie ' ')",
            "text": "Przetłumaczony tekst w formacie Markdown"
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
    
    def _clean_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove digitized by Google footnote
        text = re.sub(r'\nZdigitalizowane przez Google', '', text)
        # Remove duplicated text
        text = re.sub(r'(\b\w+(?:\s+\w+){2,}?)\s*(?:\1\s*)+', r'\1', text)
        return text
    
    def process_document(self, max_workers: int = 1, test_mode: bool = False) -> List[PageContent]:
        """
        Process the entire document with parallel processing and integrated translation.
        
        Args:
            max_workers: Maximum number of worker threads (currently not used)
            test_mode: If True, process only the first 10 pages
            
        Returns:
            List of PageContent objects containing processed content
        """
        pages_content = []
        total_pages = min(10, len(self.doc)) if test_mode else len(self.doc)
        batch_size = self.config.batch_size
        error_occurred = False
        failed_pages = set()
        
        # Create progress bar
        with tqdm(total=total_pages, desc="Processing pages") as pbar:
            # Process pages in smaller batches
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"\nProcessing batch {batch_start//batch_size + 1} (pages {batch_start+1}-{batch_end})")
                
                try:
                    page_nums = list(range(batch_start, batch_end))
                    results = self._process_page_batch(page_nums, pages_content, failed_pages)
                    pages_content.extend(results)
                    pbar.update(len(results))  # Update progress bar
                except Exception as e:
                    error_occurred = True
                    logger.error(f"Error in batch processing: {str(e)}")
                
                if error_occurred and batch_end < total_pages:
                    logger.info(f"Error detected. Taking a {self.config.retry_cooldown}-second break before continuing...")
                    time.sleep(self.config.retry_cooldown)
                    error_occurred = False
                else:
                    logger.info(f"Taking a {self.config.batch_cooldown}-second break between batches...")
                    time.sleep(self.config.batch_cooldown)
            
            # Retry failed pages
            while failed_pages:
                logger.info(f"Retrying {len(failed_pages)} failed pages after cooldown...")
                time.sleep(self.config.retry_cooldown)
                
                retry_pages = list(failed_pages)
                failed_pages.clear()
                
                for i in range(0, len(retry_pages), batch_size):
                    batch = retry_pages[i:i + batch_size]
                    results = self._process_page_batch(batch, pages_content, failed_pages)
                    pages_content.extend(results)
                    pbar.update(len(results))  # Update progress bar
                    time.sleep(self.config.batch_cooldown)
        
        # Sort pages by page number
        pages_content.sort(key=lambda x: x.page_number)
        return pages_content
    
    def _process_page_batch(self, page_nums: List[int], existing_pages: List[PageContent], failed_pages: set) -> List[PageContent]:
        """
        Process a batch of pages.
        
        Args:
            page_nums: List of page numbers to process
            existing_pages: List of already processed pages
            failed_pages: Set to track failed pages
            
        Returns:
            List of new PageContent objects
        """
        try:
            unprocessed_pages = []
            image_paths = []
            
            for page_num in page_nums:
                if any(p.page_number == page_num for p in existing_pages):
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
                if not content.get('chapters', []) or content.get('chapters', [{}])[0].get('chapter_name') == 'Error':
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
            logger.error(f"Error processing pages {page_nums}: {str(e)}")
            for page_num in page_nums:
                failed_pages.add(page_num)
            return []
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {str(e)}")

def save_as_pdf(pages_content: List[PageContent], output_path: str):
    """
    Save translated content as PDF with proper Polish character support.
    
    Args:
        pages_content: List of PageContent objects
        output_path: Path to the output PDF file
    """
    from markdown_pdf import MarkdownPdf, Section
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
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
    parser = argparse.ArgumentParser(description="Process PDF using image-based text extraction and translation")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--test", action="store_true", help="Enable test mode (process only first 10 pages)")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model name to use")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for image extraction")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    args = parser.parse_args()
    
    try:
        # Create configuration from arguments
        config = TranslatorConfig(
            model_name=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size,
            dpi=args.dpi
        )
        
        logger.info(f"Starting PDF translation with {config.model_name} model")
        logger.info(f"Input: {args.input}, Output: {args.output}, Test mode: {args.test}")
        
        translator = ImageBasedTranslator(args.input, config)
        start_time = time.time()
        pages_content = translator.process_document(test_mode=args.test)
        
        if not pages_content:
            logger.error("No content was extracted. Check for errors.")
            return
            
        save_as_pdf(pages_content, args.output)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing complete! Output saved to: {args.output}")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        if 'translator' in locals():
            translator.cleanup()

if __name__ == "__main__":
    main()