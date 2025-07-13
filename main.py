import click
import requests
import json
import tiktoken
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

HN_API_URL = "https://hacker-news.firebaseio.com/v0"
SUBMIT_LIMIT = 300

model = ChatOpenAI(model="gpt-4.1-mini")
enc = tiktoken.encoding_for_model("gpt-4o")


def count_tokens_for_input(input_str: str) -> int:
    return len(enc.encode(input_str))


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


if __name__ == '__main__':
    cli()
