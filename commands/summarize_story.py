import click
import json
from langchain_core.messages import HumanMessage, SystemMessage
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from .utils import (
    parse_story_input,
    get_story_item,
    cached_web_request,
    extract_text_from_html,
    count_tokens_for_input,
    cached_openai_request,
    get_cache_key,
    console,
    TOKEN_CONFIRM_THRESHOLD
)


@click.command()
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