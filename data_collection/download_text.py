import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from tqdm import tqdm
import argparse


# Popular Project Gutenberg books (public domain)
GUTENBERG_BOOKS = {
    "alice": {
        "id": "11",
        "title": "Alice in Wonderland",
        "url": "https://www.gutenberg.org/files/11/11-0.txt"
    },
    "sherlock": {
        "id": "1661",
        "title": "Sherlock Holmes",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt"
    },
    "pride": {
        "id": "1342",
        "title": "Pride and Prejudice",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt"
    },
    "frankenstein": {
        "id": "84",
        "title": "Frankenstein",
        "url": "https://www.gutenberg.org/files/84/84-0.txt"
    },
    "dracula": {
        "id": "345",
        "title": "Dracula",
        "url": "https://www.gutenberg.org/files/345/345-0.txt"
    },
    "huckfinn": {
        "id": "76",
        "title": "Huckleberry Finn",
        "url": "https://www.gutenberg.org/files/76/76-0.txt"
    },
    "gatsby": {
        "id": "64317",
        "title": "The Great Gatsby",
        "url": "https://www.gutenberg.org/files/64317/64317-0.txt"
    },
    "metamorphosis": {
        "id": "5200",
        "title": "Metamorphosis",
        "url": "https://www.gutenberg.org/files/5200/5200-0.txt"
    },
}


def download_book(url: str, output_path: str) -> bool:
    """
    Download a book from Project Gutenberg.
    
    Args:
        url: URL to the book text file
        output_path: Where to save the text
        
    Returns:
        Success boolean
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg header/footer boilerplate.
    
    Args:
        text: Raw book text
        
    Returns:
        Cleaned text
    """
    # Find start of actual content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip to next line after marker
            start_idx = text.find('\n', idx) + 1
            break
    
    # Find end of actual content
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
    ]
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    # Extract clean text
    clean_text = text[start_idx:end_idx].strip()
    
    return clean_text


def extract_sentences(text: str, min_length: int = 20, max_length: int = 200) -> list:
    """
    Extract valid sentences from text.
    
    Args:
        text: Book text
        min_length: Minimum sentence length
        max_length: Maximum sentence length
        
    Returns:
        List of sentences
    """
    # Split into sentences (simple version)
    import re
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    
    # Filter sentences
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        
        # Check length
        if min_length <= len(sent) <= max_length:
            # Check if mostly alphabetic
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in sent) / len(sent)
            if alpha_ratio > 0.8:
                valid_sentences.append(sent)
    
    return valid_sentences


def main():
    """Download and process books."""
    parser = argparse.ArgumentParser(description='Download training text from Project Gutenberg')
    parser.add_argument('--books', nargs='+', default=['alice', 'sherlock'],
                       choices=list(GUTENBERG_BOOKS.keys()),
                       help='Books to download')
    parser.add_argument('--sentences', type=int, default=20000,
                       help='Target number of sentences')
    parser.add_argument('--output-dir', default='data/text',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Free Training Text from Project Gutenberg")
    print("="*60)
    print(f"Books: {', '.join(args.books)}")
    print(f"Target sentences: {args.sentences}")
    print(f"Output directory: {output_dir}")
    print()
    
    all_sentences = []
    
    for book_key in args.books:
        book_info = GUTENBERG_BOOKS[book_key]
        print(f"\n📖 Downloading: {book_info['title']}")
        print(f"   URL: {book_info['url']}")
        
        # Download
        raw_path = output_dir / f"{book_key}_raw.txt"
        if raw_path.exists():
            print(f"   ✓ Already downloaded, loading from {raw_path}")
            with open(raw_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            success = download_book(book_info['url'], raw_path)
            if not success:
                print(f"   ✗ Failed to download")
                continue
            
            with open(raw_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"   ✓ Downloaded {len(text):,} characters")
        
        # Clean
        clean_text = clean_gutenberg_text(text)
        print(f"   ✓ Cleaned to {len(clean_text):,} characters")
        
        # Extract sentences
        sentences = extract_sentences(clean_text)
        all_sentences.extend(sentences)
        print(f"   ✓ Extracted {len(sentences):,} sentences")
        
        # Save cleaned text
        clean_path = output_dir / f"{book_key}_clean.txt"
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        # Be nice to Project Gutenberg servers
        time.sleep(1)
    
    # Save all sentences
    print(f"\n" + "="*60)
    print(f"Total sentences collected: {len(all_sentences):,}")
    
    # Limit to target number
    if len(all_sentences) > args.sentences:
        all_sentences = all_sentences[:args.sentences]
        print(f"Limited to: {len(all_sentences):,} sentences")
    
    # Save to file
    sentences_path = output_dir / 'sentences.txt'
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for sent in all_sentences:
            f.write(sent + '\n')
    
    print(f"\n✓ Saved to: {sentences_path}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total sentences: {len(all_sentences):,}")
    print(f"  Average length: {sum(len(s) for s in all_sentences) / len(all_sentences):.1f} chars")
    print(f"  Shortest: {min(len(s) for s in all_sentences)} chars")
    print(f"  Longest: {max(len(s) for s in all_sentences)} chars")
    
    print(f"\n✓ Download complete!")


if __name__ == "__main__":
    main()