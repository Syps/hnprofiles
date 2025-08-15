from click import Command
import click
import json
from .utils import (
    get_user, 
    get_all_user_comments, 
    infer_profile_from_comments,
    count_tokens_for_input,
    TOKEN_CONFIRM_THRESHOLD
)


@click.command()
@click.argument("username", type=str)
@click.option("--nocache", is_flag=True, help="Skip cache and make fresh requests")
def about(username, nocache):
    """Tell me about this user"""
    use_cache = not nocache
    user = get_user(username, use_cache)
    comments = get_all_user_comments(user['submitted'], use_cache)
    
    prompt = """
    Based on the Hacker News comments, make an educated guess of the user's profile. Your output should be in the following format (make sure to add newline characters after eacch line):
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