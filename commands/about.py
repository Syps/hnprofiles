import click
import json
from .utils import (
    get_user, 
    get_all_user_comments,
    scrape_hn_profile,
    detect_personal_blog,
    scrape_blog_content,
    infer_profile_from_structured_data,
    count_tokens_for_input,
    TOKEN_CONFIRM_THRESHOLD
)


@click.command()
@click.argument("username", type=str)
@click.option("--nocache", is_flag=True, help="Skip cache and make fresh requests")
@click.option("--debug", is_flag=True, help="Show prompt and ask for confirmation before sending to AI")
def about(username, nocache, debug):
    """Tell me about this user"""
    use_cache = not nocache
    
    # Step 1: Get user data from HN API
    click.echo(f"Fetching user data for '{username}'...")
    user = get_user(username, use_cache)
    
    # Step 2: Get user comments
    click.echo("Fetching user comments...")
    comments = get_all_user_comments(user['submitted'], use_cache)

    # Step 3: Scrape HN profile page
    click.echo("Scraping HN profile page...")
    profile_data = scrape_hn_profile(username, use_cache)

    print("\n" + "-"*80)
    print("Profile data:")
    print(profile_data)
    
    # Step 4: Detect personal blog from profile
    blog_content = None
    if profile_data.get('about'):
        blog_url = detect_personal_blog(profile_data['about'])
        
        # Step 5: Scrape blog content if found
        if blog_url:
            click.echo(f"Personal blog detected: {blog_url}")
            click.echo("Scraping blog content...")
            blog_content = scrape_blog_content(blog_url, use_cache)
        else:
            click.echo("No personal blog detected in profile")
    else:
        click.echo("No 'about' section found in profile")
    
    # Step 6: Structure all data with headings
    click.echo("Combining all data sources...")
    structured_data = "## Comments\n"
    structured_data += json.dumps(comments, indent=2)
    
    structured_data += "\n\n## HN Profile\n"
    if profile_data.get('created'):
        structured_data += f"Created: {profile_data['created']}\n"
    if profile_data.get('karma'):
        structured_data += f"Karma: {profile_data['karma']}\n"
    if profile_data.get('about'):
        structured_data += f"About: {profile_data['about']}\n"
    
    structured_data += "\n\n## Blog Content\n"
    print("\n" + "-"*80)
    print("Blog Content:")
    print(blog_content)
    if not click.confirm("Approve?"):
        click.echo("Cancelled.")
        return
    if blog_content:
        structured_data += blog_content
    else:
        structured_data += "No personal blog detected or content unavailable"
    
    # Step 7: Check token count before analysis
    token_count = count_tokens_for_input(structured_data)

    
    # Step 8: Run comprehensive analysis
    infer_profile_from_structured_data(structured_data, username, use_cache, debug)