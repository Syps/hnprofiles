import click
import json
from langchain_core.messages import HumanMessage, SystemMessage
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from .utils import (
    extract_story_id,
    get_story_item,
    fetch_all_story_comments,
    format_comments_for_llm,
    count_tokens_for_input,
    cached_openai_request,
    get_cache_key,
    console,
    TOKEN_CONFIRM_THRESHOLD
)


@click.command()
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