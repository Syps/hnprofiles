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

HN_API_URL = "https://hacker-news.firebaseio.com/v0"
SUBMIT_LIMIT = 300
CACHE_DIR = Path(".cache")

model = ChatOpenAI(model="gpt-4.1-mini")
enc = tiktoken.encoding_for_model("gpt-4o")
console = Console()

# Initialize Ollama for text extraction
ollama_model = OllamaLLM(
    model="llama3.1:8b",
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
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0 (compatible; HNProfiles/1.0)'})
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
    """Extract clean text from HTML using Ollama llama3.1:8b"""
    prompt = """Extract the main article text content from this HTML webpage. 
    
Rules:
- Return ONLY the main article/story text content
- Remove all HTML tags, CSS, JavaScript, navigation menus, ads, sidebars, and footer content  
- Remove social media buttons, comments sections, and related articles
- Keep only the core readable text that represents the main article
- Preserve paragraph breaks with double newlines
- Do not include any explanations or metadata, just the clean text

HTML content:
"""
    
    try:
        click.echo("Extracting text content using Ollama...")
        extracted_text = ollama_model.invoke(prompt + html_content[:50000])  # Limit HTML size
        return extracted_text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text using Ollama: {e}")


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

    click.echo(f"Analyzing {len(comments)} comments with AI (estimated {token_estimate:,} tokens)...")
    
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


@click.group()
def cli():
    """HN Profiles - Analyze Hacker News user profiles"""
    pass


@cli.command()
@click.argument("username", type=str)
@click.option("--nocache", is_flag=True, help="Skip cache and make fresh requests")
def about(username, nocache):
    """Tell me about this user"""
    use_cache = not nocache
    user = get_user(username, use_cache)
    comments = get_all_user_comments(user['submitted'], use_cache)
    
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
    token_count = count_tokens_for_input(total_input)
    
    if token_count > TOKEN_CONFIRM_THRESHOLD:
        estimated_cost = (token_count / 1_000_000) * 0.40
        click.echo(f"Token count: {token_count:,}")
        click.echo(f"Estimated cost: ${estimated_cost:.4f}")
        if not click.confirm("Continue with the request?"):
            click.echo("Request cancelled.")
            return
    
    infer_profile_from_comments(comments, use_cache)


@cli.command()
@click.argument("story_url", type=str)
@click.option("--nocache", is_flag=True, help="Skip cache and make fresh requests")
def analyze_story(story_url, nocache):
    """Analyze all comments on a Hacker News story"""
    try:
        story_id = extract_story_id(story_url)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return
    
    use_cache = not nocache
    # Fetch story and comments
    story = get_story_item(story_id, use_cache)
    comments_data = fetch_all_story_comments(story, use_cache)
    
    if comments_data['total_count'] == 0:
        click.echo("No comments to analyze.")
        return
    
    # Format for LLM
    formatted_text = format_comments_for_llm(story, comments_data)
    
    # Check token count
    prompt = """
    Analyze the following Hacker News story and its comments. Provide a concise overview that includes:
    
    1. **Main Topic**: What is the story about? 1-2 bullets
    2. **Key Themes**: What are the main themes/topics being discussed in the comments? 2-4 bullets
    3. **Popular Opinions**: What are the most common viewpoints or opinions expressed? 3 max, as bullets. For each, provide a link to the most popular comment supporting that point. To build the link, just add the comment id (comment['id']) to the end of the base URL `https://news.ycombinator.com/item?id=`.
    5. **Debate Points**: What are people debating or disagreeing about? Only include if there's repeated disagreement in the comments. Include 1 short bullet per side.
    6. **Overall Sentiment**: What's the general tone of the discussion? 5 words max
    """
    
    total_input = prompt + formatted_text
    token_count = count_tokens_for_input(total_input)
    
    if token_count > TOKEN_CONFIRM_THRESHOLD:
        estimated_cost = (token_count / 1_000_000) * 0.40
        click.echo(f"Token count: {token_count:,}")
        click.echo(f"Estimated cost: ${estimated_cost:.4f}")
        if not click.confirm("Continue with the analysis?"):
            click.echo("Analysis cancelled.")
            return
    
    click.echo(f"Analyzing {comments_data['total_count']} comments with AI (estimated {token_count:,} tokens)...")
    
    messages = [
        SystemMessage(prompt),
        HumanMessage(formatted_text),
    ]
    
    # Create cache key from input content
    cache_key = get_cache_key(prompt + formatted_text)
    result, cache_hit = cached_openai_request(messages, cache_key, use_cache)
    
    # Create aggregate-like object for compatibility
    class MockAggregate:
        def __init__(self, content, usage_metadata):
            self.content = content
            self.usage_metadata = usage_metadata
    
    aggregate = MockAggregate(result['content'], result['usage_metadata'])

    if cache_hit:
        console.print()
        console.print(Markdown(result['content']))
    
    # Create story summary panel
    story_info = Text()
    story_info.append(f"Story: {story.get('title', 'No title')}\n", style="bold cyan")
    story_info.append(f"Total comments analyzed: {comments_data['total_count']}\n", style="green")
    story_info.append(f"Pre-send token estimate: {token_count:,}\n", style="yellow")
    story_info.append(f"Token usage: {json.dumps(aggregate.usage_metadata, indent=2)}", style="dim")
    
    console.print()
    console.print(Panel(
        story_info,
        title="[bold green]Analysis Complete[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))


@cli.command()
@click.argument("story_input", type=str)
@click.option("--nocache", is_flag=True, help="Skip cache and make fresh requests")
def summarize_story(story_input, nocache):
    """Summarize the content of a Hacker News story URL"""
    use_cache = not nocache
    
    try:
        # Parse story input (URL or ID)
        story_id = parse_story_input(story_input)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    
    try:
        # Fetch story from HN API
        story = get_story_item(story_id, use_cache)
        
        # Check if story has URL
        if not story.get('url'):
            console.print(f"[red]Error:[/red] Story {story_id} does not have a URL to summarize")
            console.print(f"Story title: {story.get('title', 'No title')}")
            return
        
        story_url = story['url']
        console.print(f"[bold cyan]Story:[/bold cyan] {story.get('title', 'No title')}")
        console.print(f"[bold cyan]URL:[/bold cyan] {story_url}")
        console.print()
        
        # Fetch webpage content
        try:
            html_content = cached_web_request(story_url, use_cache)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        
        # Extract text using Ollama
        try:
            extracted_text = extract_text_from_html(html_content)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        
        # Create summarization prompt
        summary_prompt = """Please provide a concise one-paragraph summary of the following article. 
        Focus on the main points, key findings, and important details. Make it informative but readable.
        
        Article text:
        """
        
        total_input = summary_prompt + extracted_text
        token_estimate = count_tokens_for_input(total_input)
        
        # Check token count and cost
        if token_estimate > TOKEN_CONFIRM_THRESHOLD:
            estimated_cost = (token_estimate / 1_000_000) * 0.40
            console.print(f"[yellow]Warning:[/yellow] Large content detected")
            console.print(f"Token count: {token_estimate:,}")
            console.print(f"Estimated cost: ${estimated_cost:.4f}")
            if not click.confirm("Continue with summarization?"):
                console.print("Summarization cancelled.")
                return
        
        console.print(f"Generating summary... (estimated {token_estimate:,} tokens)")
        
        # Generate summary using OpenAI
        messages = [
            SystemMessage(summary_prompt),
            HumanMessage(extracted_text),
        ]
        
        cache_key = get_cache_key(total_input)
        result, cache_hit = cached_openai_request(messages, cache_key, use_cache)
        
        # Display results
        if cache_hit:
            console.print()
            console.print(Markdown(result['content']))
        
        # Create summary panel
        summary_info = Text()
        summary_info.append(f"Story: {story.get('title', 'No title')}\n", style="bold cyan")
        summary_info.append(f"Original URL: {story_url}\n", style="blue")
        summary_info.append(f"Token usage: {json.dumps(result['usage_metadata'], indent=2)}\n", style="dim")
        summary_info.append(f"Pre-send token estimate: {token_estimate:,}", style="yellow")
        
        console.print()
        console.print(Panel(
            summary_info,
            title="[bold green]Summary Complete[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


if __name__ == '__main__':
    cli()
