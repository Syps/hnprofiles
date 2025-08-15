import click
from commands.about import about
from commands.analyze_story import analyze_story
from commands.summarize_story import summarize_story


@click.group()
def cli():
    """HN Profiles - Analyze Hacker News user profiles"""
    pass


cli.add_command(about)
cli.add_command(analyze_story)
cli.add_command(summarize_story)


if __name__ == '__main__':
    cli()
