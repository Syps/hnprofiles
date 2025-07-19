import click
import requests
import json
import tiktoken
import re
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

HN_API_URL = "https://hacker-news.firebaseio.com/v0"
SUBMIT_LIMIT = 300

model = ChatOpenAI(model="gpt-4.1-mini")
enc = tiktoken.encoding_for_model("gpt-4o")


def count_tokens_for_input(input_str: str) -> int:
    return len(enc.encode(input_str))


def extract_story_id(url: str) -> int:
    """Extract story ID from HN URL"""
    match = re.search(r'id=([0-9]+)', url)
    if not match:
        raise ValueError(f"Could not extract story ID from URL: {url}")
    return int(match.group(1))


def get_story_item(story_id: int) -> dict:
    """Fetch story item from HN API"""
    click.echo(f"Fetching story {story_id}...")
    res = requests.get(f"{HN_API_URL}/item/{story_id}.json").json()
    if not res:
        raise ValueError(f"Story {story_id} not found")
    return res


def fetch_comment_recursively(comment_id: int, comments_dict: dict, depth: int = 0) -> dict:
    """Recursively fetch a comment and all its children"""
    res = requests.get(f"{HN_API_URL}/item/{comment_id}.json").json()
    
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
            child = fetch_comment_recursively(kid_id, comments_dict, depth + 1)
            if child:
                comment['children'].append(child)
    
    comments_dict[comment_id] = comment
    return comment


def fetch_all_story_comments(story: dict) -> dict:
    """Fetch all comments for a story recursively"""
    comments_dict = {}
    root_comments = []
    
    if 'kids' not in story:
        click.echo("No comments found for this story")
        return {'comments': [], 'total_count': 0}
    
    click.echo(f"Fetching {len(story['kids'])} top-level comments recursively...")
    
    with tqdm(total=len(story['kids']), desc="Fetching comment trees", unit="trees") as pbar:
        for comment_id in story['kids']:
            comment = fetch_comment_recursively(comment_id, comments_dict, 0)
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


def get_all_user_comments(submits: list[int]) -> list[dict]:
    comments = []
    total_comments = 0
    
    click.echo(f"Fetching up to {SUBMIT_LIMIT} comments from {len(submits)} submissions...")
    
    with tqdm(total=min(SUBMIT_LIMIT, len(submits)), desc="Fetching comments", unit="items") as pbar:
        for submitted in submits:
            if total_comments >= SUBMIT_LIMIT:
                break

            res = requests.get(f"{HN_API_URL}/item/{submitted}.json").json()
            if res and res.get('type') == 'comment':
                comments.append(res)
                total_comments += 1
                pbar.set_postfix({'comments': total_comments})
            
            pbar.update(1)

    click.echo(f"Found {len(comments)} comments")
    return comments


def get_user(username: str) -> dict:
    click.echo(f"Fetching user profile for '{username}'...")
    user_api_url = f"{HN_API_URL}/user/{username}.json"
    res = requests.get(user_api_url).json()
    if res:
        click.echo(f"Found user with {len(res.get('submitted', []))} total submissions")
    return res


def infer_profile_from_comments(comments: list[dict]):
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

    click.echo(f"Analyzing {len(comments)} comments with AI (estimated {token_estimate:,} tokens)...")
    
    messages = [
        SystemMessage(prompt),
        HumanMessage(json.dumps(comments)),
    ]

    aggregate = None
    for chunk in model.stream(messages, stream_usage=True):
        print(chunk.content, end="")
        aggregate = chunk if aggregate is None else aggregate + chunk

    print("\n-------------------")
    print("Usage:")
    print(json.dumps(aggregate.usage_metadata, indent=2))
    print("\n--------------------")
    print(f"Pre-send token count: {token_estimate}")


@click.group()
def cli():
    """HN Profiles - Analyze Hacker News user profiles"""
    pass


@cli.command()
@click.argument("username", type=str)
def about(username):
    """Tell me about this user"""
    user = get_user(username)
    comments = get_all_user_comments(user['submitted'])
    
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
    
    if token_count > 50000:
        estimated_cost = (token_count / 1_000_000) * 0.40
        click.echo(f"Token count: {token_count:,}")
        click.echo(f"Estimated cost: ${estimated_cost:.4f}")
        if not click.confirm("Continue with the request?"):
            click.echo("Request cancelled.")
            return
    
    infer_profile_from_comments(comments)


@cli.command()
@click.argument("story_url", type=str)
def analyze_story(story_url):
    """Analyze all comments on a Hacker News story"""
    try:
        story_id = extract_story_id(story_url)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return
    
    # Fetch story and comments
    story = get_story_item(story_id)
    comments_data = fetch_all_story_comments(story)
    
    if comments_data['total_count'] == 0:
        click.echo("No comments to analyze.")
        return
    
    # Format for LLM
    formatted_text = format_comments_for_llm(story, comments_data)
    
    # Check token count
    prompt = """
    Analyze the following Hacker News story and its comments. Provide a comprehensive overview that includes:
    
    1. **Main Topic**: What is the story about?
    2. **Key Themes**: What are the main themes/topics being discussed in the comments?
    3. **Popular Opinions**: What are the most common viewpoints or opinions expressed?
    4. **Notable Insights**: Any particularly interesting insights, anecdotes, or technical details shared?
    5. **Debate Points**: What are people debating or disagreeing about?
    6. **Overall Sentiment**: What's the general tone of the discussion?
    
    Make this overview concise but comprehensive enough that someone could understand the essence of the entire comment thread without reading all the individual comments.
    """
    
    total_input = prompt + formatted_text
    token_count = count_tokens_for_input(total_input)
    
    if token_count > 50000:
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
    
    aggregate = None
    for chunk in model.stream(messages, stream_usage=True):
        print(chunk.content, end="")
        aggregate = chunk if aggregate is None else aggregate + chunk
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Story: {story.get('title', 'No title')}")
    print(f"Total comments analyzed: {comments_data['total_count']}")
    print(f"Token usage: {json.dumps(aggregate.usage_metadata, indent=2)}")
    print(f"Pre-send token estimate: {token_count:,}")


if __name__ == '__main__':
    cli()
