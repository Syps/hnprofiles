# HN Profiles

A Python CLI tool for analyzing Hacker News users and discussions using AI.

## Features

- **User Analysis**: Analyze HN user profiles by examining their comment history to infer nationality, age, occupation, political leaning, and interests
- **Story Analysis**: Analyze all comments on a HN story to get key themes, popular opinions, and debate points

## Installation

```bash
make install
```

Requires an OpenAI API key set as an environment variable.

## Usage

### Analyze a user profile
```bash
python main.py about <username>
```

### Analyze a story discussion
```bash
python main.py analyze-story <story_url>
```

## How it works

- Fetches data from the Hacker News API
- Uses GPT-4 mini via LangChain to analyze comments
- Provides token count estimates and cost warnings for large analyses
- Limits user comment analysis to 300 items for cost control

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies managed via `uv`