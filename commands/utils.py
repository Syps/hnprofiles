import click
import requests
import json
import tiktoken
import re
import os
import hashlib
from pathlib import Path
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from langchain_ollama import OllamaLLM
import html2text

HN_API_URL = "https://hacker-news.firebaseio.com/v0"
SUBMIT_LIMIT = 300
CACHE_DIR = Path(".cache")

LOCAL_LLM = "gpt-oss:20b"
CLOUD_LLM = "gpt-4.1-mini"

model = ChatOpenAI(model=CLOUD_LLM)
enc = tiktoken.encoding_for_model("gpt-4o")
console = Console()

# Initialize Ollama for text extraction
ollama_model = OllamaLLM(
    model=LOCAL_LLM,
    base_url="http://127.0.0.1:11434"
)

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "hn_api").mkdir(exist_ok=True)
(CACHE_DIR / "openai").mkdir(exist_ok=True)
(CACHE_DIR / "web_content").mkdir(exist_ok=True)

TOKEN_CONFIRM_THRESHOLD = 50_000


def count_tokens_for_input(input_str: str) -> int:
    return len(enc.encode(input_str))


def get_cache_key(data: str) -> str:
    """Generate a cache key from data"""
    return hashlib.md5(data.encode()).hexdigest()


def load_from_cache(cache_file: Path) -> dict:
    """Load data from cache file if it exists"""
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_to_cache(cache_file: Path, data: dict) -> None:
    """Save data to cache file"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(data, f)


def cached_request(url: str, use_cache: bool = True) -> dict:
    """Make a cached HTTP request to HN API"""
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / "hn_api" / f"{cache_key}.json"
    
    # Try to load from cache first if use_cache is True
    if use_cache:
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            return cached_data
    
    # Make actual request and cache result
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })
    data = response.json()
    save_to_cache(cache_file, data)
    return data


def cached_web_request(url: str, use_cache: bool = True) -> str:
    """Make a cached HTTP request for web content"""
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / "web_content" / f"{cache_key}.json"
    
    if use_cache:
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            click.echo(f"Loading webpage content from cache...")
            return cached_data.get('content', '')
    
    click.echo(f"Fetching webpage: {url}")
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 (compatible; HNProfiles/1.0)'})
        response.raise_for_status()
        content = response.text
        
        # Cache the result
        save_to_cache(cache_file, {'content': content, 'url': url})
        return content
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch webpage: {e}")


def extract_story_id(url: str) -> int:
    """Extract story ID from HN URL"""
    match = re.search(r'id=([0-9]+)', url)
    if not match:
        raise ValueError(f"Could not extract story ID from URL: {url}")
    return int(match.group(1))


def parse_story_input(story_input: str) -> int:
    """Parse story input - can be URL or direct ID"""
    # If it's a URL, extract the ID
    if story_input.startswith('http'):
        return extract_story_id(story_input)
    # If it's just a number, use it directly
    try:
        return int(story_input)
    except ValueError:
        raise ValueError(f"Invalid story input: {story_input}. Must be HN URL or story ID.")


def get_story_item(story_id: int, use_cache: bool = True) -> dict:
    """Fetch story item from HN API with caching"""
    url = f"{HN_API_URL}/item/{story_id}.json"
    
    if use_cache:
        cache_key = get_cache_key(url)
        cache_file = CACHE_DIR / "hn_api" / f"{cache_key}.json"
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            click.echo(f"Loading story {story_id} from cache...")
            return cached_data
    
    click.echo(f"Fetching story {story_id}...")
    res = cached_request(url, use_cache)
    if not res:
        raise ValueError(f"Story {story_id} not found")
    return res


def fetch_comment_recursively(comment_id: int, comments_dict: dict, depth: int = 0, use_cache: bool = True) -> dict:
    """Recursively fetch a comment and all its children with caching"""
    res = cached_request(f"{HN_API_URL}/item/{comment_id}.json", use_cache)
    
    if not res or res.get('deleted') or res.get('dead'):
        return None
    
    comment = {
        'id': res['id'],
        'by': res.get('by', '[deleted]'),
        'text': res.get('text', ''),
        'time': res.get('time', 0),
        'parent': res.get('parent'),
        'depth': depth,
        'children': []
    }
    
    # Recursively fetch children
    if 'kids' in res:
        for kid_id in res['kids']:
            child = fetch_comment_recursively(kid_id, comments_dict, depth + 1, use_cache)
            if child:
                comment['children'].append(child)
    
    comments_dict[comment_id] = comment
    return comment


def fetch_all_story_comments(story: dict, use_cache: bool = True) -> dict:
    """Fetch all comments for a story recursively with caching"""
    comments_dict = {}
    root_comments = []
    
    if 'kids' not in story:
        click.echo("No comments found for this story")
        return {'comments': [], 'total_count': 0}
    
    click.echo(f"Fetching {len(story['kids'])} top-level comments recursively...")
    
    with tqdm(total=len(story['kids']), desc="Fetching comment trees", unit="trees") as pbar:
        for comment_id in story['kids']:
            comment = fetch_comment_recursively(comment_id, comments_dict, 0, use_cache)
            if comment:
                root_comments.append(comment)
            pbar.update(1)
            pbar.set_postfix({'total_comments': len(comments_dict)})
    
    click.echo(f"Fetched {len(comments_dict)} total comments")
    return {'comments': root_comments, 'total_count': len(comments_dict)}


def format_comments_for_llm(story: dict, comments_data: dict) -> str:
    """Format story and comments into LLM-readable text"""
    def format_comment(comment: dict, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        text = comment['text'].replace('<p>', '\n').replace('</p>', '')
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        formatted = f"{indent}[Comment by {comment['by']} (depth {comment['depth']})]\n"
        formatted += f"{indent}{text}\n\n"
        
        # Add children
        for child in comment['children']:
            formatted += format_comment(child, indent_level + 1)
        
        return formatted
    
    # Start with story info
    story_text = f"STORY: {story.get('title', 'No title')}\n"
    story_text += f"URL: {story.get('url', 'No URL')}\n"
    story_text += f"Score: {story.get('score', 0)} | Comments: {story.get('descendants', 0)}\n"
    if story.get('text'):
        clean_text = re.sub(r'<[^>]+>', '', story['text'])
        story_text += f"Text: {clean_text}\n"
    story_text += "\n" + "="*80 + "\n"
    story_text += "COMMENTS:\n\n"
    
    # Add all comments
    for comment in comments_data['comments']:
        story_text += format_comment(comment)
    
    return story_text


def get_all_user_comments(submits: list[int], use_cache: bool = True) -> list[dict]:
    comments = []
    total_comments = 0
    
    click.echo(f"Fetching up to {SUBMIT_LIMIT} comments from {len(submits)} submissions...")
    
    with tqdm(total=min(SUBMIT_LIMIT, len(submits)), desc="Fetching comments", unit="items") as pbar:
        for submitted in submits:
            if total_comments >= SUBMIT_LIMIT:
                break

            res = cached_request(f"{HN_API_URL}/item/{submitted}.json", use_cache)
            if res and res.get('type') == 'comment':
                comments.append(res)
                total_comments += 1
                pbar.set_postfix({'comments': total_comments})
            
            pbar.update(1)

    click.echo(f"Found {len(comments)} comments")
    return comments


def get_user(username: str, use_cache: bool = True) -> dict:
    user_api_url = f"{HN_API_URL}/user/{username}.json"
    
    if use_cache:
        cache_key = get_cache_key(user_api_url)
        cache_file = CACHE_DIR / "hn_api" / f"{cache_key}.json"
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            click.echo(f"Loading user profile for '{username}' from cache...")
            if cached_data:
                click.echo(f"Found user with {len(cached_data.get('submitted', []))} total submissions")
            return cached_data
    
    click.echo(f"Fetching user profile for '{username}'...")
    res = cached_request(user_api_url, use_cache)
    if res:
        click.echo(f"Found user with {len(res.get('submitted', []))} total submissions")
    return res


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML using html2text"""
    try:
        click.echo("Extracting text content using html2text...")
        
        # Configure html2text for clean output
        h = html2text.HTML2Text()
        h.ignore_links = False  # Keep links for context
        h.ignore_images = True  # Remove images
        h.ignore_emphasis = False  # Keep bold/italic formatting
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True  # Use unicode characters
        h.escape_snob = True  # Escape special characters properly
        
        # Convert HTML to clean text
        text_content = h.handle(html_content)
        
        # Clean up extra whitespace while preserving structure
        lines = text_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line for separation
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines).strip()
        
    except Exception as e:
        raise ValueError(f"Failed to extract text using html2text: {e}")


def scrape_hn_profile(username: str, use_cache: bool = True) -> dict:
    """Scrape HN user profile page and extract key information"""
    profile_url = f"https://news.ycombinator.com/user?id={username.strip()}"
    
    try:
        # Use existing web request caching
        html_content = cached_web_request(profile_url, use_cache)
        
        # Parse HTML to extract profile data
        profile_data = {
            'created': None,
            'karma': None,
            'about': None
        }
        
        # Extract created date - looks for the link with date
        created_match = re.search(r'<td valign="top">created:</td><td><a[^>]*>([^<]+)</a></td>', html_content)
        if created_match:
            profile_data['created'] = created_match.group(1).strip()
        
        # Extract karma - between karma td tags
        karma_match = re.search(r'<td valign="top">karma:</td><td>(\d+)</td>', html_content)
        if karma_match:
            profile_data['karma'] = int(karma_match.group(1))
        
        # Extract about section - more complex because it contains HTML
        about_match = re.search(r'<td valign="top">about:</td><td[^>]*>(.*?)</td>', html_content, re.DOTALL)
        if about_match:
            about_html = about_match.group(1)
            # Clean up the about HTML
            # First decode HTML entities
            about_text = about_html.replace('&#x2F;', '/').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            # Remove HTML tags but preserve line breaks
            about_text = re.sub(r'<p>', '\n', about_text)
            about_text = re.sub(r'<[^>]+>', '', about_text)
            # Clean up whitespace
            about_text = about_text.strip()
            profile_data['about'] = about_text if about_text else None
        
        return profile_data
        
    except Exception as e:
        click.echo(f"Warning: Failed to scrape HN profile: {e}")
        return {'created': None, 'karma': None, 'about': None}


def detect_personal_blog(about_text: str) -> str:
    """Use LOCAL_LLM to detect personal blog URL from HN about section"""
    if not about_text:
        return None
    
    # Pre-clean HTML content if present to make LOCAL_LLM more efficient
    clean_text = about_text
    if '<' in about_text and '>' in about_text:
        try:
            # Use html2text to clean any HTML in the about section
            h = html2text.HTML2Text()
            h.ignore_links = False  # Keep links - they're important for blog detection
            h.ignore_images = True
            h.ignore_emphasis = True
            h.body_width = 0
            h.unicode_snob = True
            clean_text = h.handle(about_text).strip()
        except Exception:
            # If html2text fails, use the original text
            clean_text = about_text
    
    prompt = f"""Analyze the following HN user's "about" section and identify if there's a personal blog URL.

Rules:
- Look for URLs that appear to be personal blogs, websites, or portfolios
- EXCLUDE: GitHub, Twitter, LinkedIn, Facebook, Instagram, YouTube, Reddit, and other social media
- EXCLUDE: Company websites, documentation sites, Wikipedia links
- INCLUDE: Personal domains and personal GitHub Pages sites
- Return ONLY the URL if found, or "NONE" if no personal blog is detected
- Do not include any explanations or additional text

About section:
{clean_text}"""

    try:
        click.echo("Analyzing profile for personal blog...")
        result = ollama_model.invoke(prompt)
        result = result.strip()
        
        # Check if result looks like a URL and isn't "NONE"
        if result and result != "NONE" and ("http" in result or "www." in result):
            # Clean up the result - extract just the URL
            url_match = re.search(r'https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+', result)
            if url_match:
                url = url_match.group(0)
                # Add https:// if missing
                if not url.startswith('http'):
                    url = 'https://' + url
                return url
        
        return None
        
    except Exception as e:
        click.echo(f"Warning: Failed to detect blog URL: {e}")
        return None


def scrape_blog_content(blog_url: str, use_cache: bool = True) -> str:
    """Scrape blog content using existing web request and text extraction"""
    try:
        # Fetch blog HTML using existing caching
        html_content = cached_web_request(blog_url, use_cache)
        
        # Extract text using existing function
        text_content = extract_text_from_html(html_content)
        
        return text_content
        
    except Exception as e:
        click.echo(f"Warning: Failed to scrape blog content: {e}")
        return None


def render_analysis_output(content: str):
    """Render AI analysis output with rich formatting"""
    console.print()
    console.print(Markdown(content))


def cached_openai_request(messages: list, cache_key: str, use_cache: bool = True) -> tuple[dict, bool]:
    """Make a cached OpenAI request"""
    cache_file = CACHE_DIR / "openai" / f"{cache_key}.json"
    
    if use_cache:
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            click.echo("Loading AI analysis from cache...")
            return cached_data, True
    
    # Make actual OpenAI request
    aggregate = None
    content_parts = []
    
    # Stream output and collect content
    console.print()
    for chunk in model.stream(messages, stream_usage=True):
        content_parts.append(chunk.content)
        aggregate = chunk if aggregate is None else aggregate + chunk
    
    # Render the complete content with markdown formatting
    full_content = ''.join(content_parts)
    console.print(Markdown(full_content))
    
    # Cache the result
    result = {
        'content': full_content,
        'usage_metadata': aggregate.usage_metadata
    }
    save_to_cache(cache_file, result)
    return result, False


def infer_profile_from_comments(comments: list[dict], use_cache: bool = True):
    prompt = """
    Based on the Hacker News comments, make an educated guess of the user's profile. Your output should be in the following format:
    Profile: {username}
    -------------------
    Nationality: {}
    Age: {}
    Occupation: {}
    Political Leaning: {}
    Interests: {}
    """
    total_input = prompt + json.dumps(comments)
    token_estimate = count_tokens_for_input(total_input)
    
    # Create cache key from input content
    cache_key = get_cache_key(total_input)

    click.echo(f"Analyzing {len(comments)} comments with {CLOUD_LLM} (estimated {token_estimate:,} tokens)...")
    
    messages = [
        SystemMessage(prompt),
        HumanMessage(json.dumps(comments)),
    ]

    result, cache_hit = cached_openai_request(messages, cache_key, use_cache)

    if cache_hit:
        print()
        print(result["content"])
    if not result['content'].endswith('\n'):
        print()
    print("\n-------------------")
    print("Usage:")
    print(json.dumps(result['usage_metadata'], indent=2))
    print("\n--------------------")
    print(f"Pre-send token count: {token_estimate}")


def infer_profile_from_structured_data(structured_data: str, username: str, use_cache: bool = True, debug: bool = False):
    """Enhanced profile inference using comments, HN profile, and blog data"""
    prompt = f"""
    Based on the comprehensive data about this Hacker News user, make an educated guess of their profile. 
    Analyze ALL provided data sources to create the most accurate assessment possible. 
    Make sure to utilize any Blog Content if available. Look there for more info like name, employer, etc.
    
    Your output should be in the following format:
    Profile: {username}
    -------------------
    Name: {{based on blog content or 'unknown'}}
    Employer: {{based on all available content or 'unknown'}}'
    Title: {{based on all available content or 'unknown'}}
    Nationality: {{based on all available evidence}}
    Age: {{estimated age range or specific age if clear}}
    Occupation: {{current or likely profession}}
    Political Leaning: {{if discernible from content}}
    Interests: {{key interests and hobbies}}
    
    Additional Insights: {{any other notable characteristics, writing style, expertise areas, etc.}}
    """
    
    total_input = prompt + structured_data
    token_count = count_tokens_for_input(total_input)

    # Debug mode: show prompt and ask for confirmation
    if debug:
        print("\n" + "="*80)
        print("DEBUG: PROMPT TO BE SENT TO CLOUD_LLM")
        print("="*80)
        print("SYSTEM MESSAGE:")
        print(prompt)
        print("\n" + "-"*80)
        print("USER MESSAGE (STRUCTURED DATA):")
        print(structured_data[:2000] + ("..." if len(structured_data) > 2000 else ""))
        print("\n" + "-"*80)
        print(f"Total input length: {len(total_input):,} characters")
        print(f"Estimated tokens: {token_count:,}")
        print("="*80)

    if token_count > TOKEN_CONFIRM_THRESHOLD:
        estimated_cost = (token_count / 1_000_000) * 0.40
        click.echo(f"Token count: {token_count:,}")
        click.echo(f"Estimated cost: ${estimated_cost:.4f}")
        if not click.confirm("Continue with the comprehensive analysis?"):
            click.echo("Analysis cancelled.")
            return
    
    # Create cache key from input content
    cache_key = get_cache_key(total_input)

    click.echo(f"Analyzing comprehensive user data with {CLOUD_LLM} (estimated {token_count:,} tokens)...")
    
    messages = [
        SystemMessage(prompt),
        HumanMessage(structured_data),
    ]

    result, cache_hit = cached_openai_request(messages, cache_key, use_cache)

    if cache_hit:
        print()
        print(result["content"])
    if not result['content'].endswith('\n'):
        print()
    print("\n-------------------")
    print("Usage:")
    print(json.dumps(result['usage_metadata'], indent=2))
    print("\n--------------------")
    print(f"Pre-send token count: {token_count}")

