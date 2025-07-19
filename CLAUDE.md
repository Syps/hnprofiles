# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python CLI tool that analyzes Hacker News user profiles by fetching their comments via the HN API and using OpenAI's GPT model to infer user characteristics (nationality, age, occupation, political leaning, interests).

## Architecture

- **Single module design**: The entire application is contained in `main.py`
- **Virtual environment**: Uses `hnprofiles/` venv with Python 3.11
- **Dependencies**: Managed via `uv` with `requirements.in` source and compiled `requirements.txt`
- **Key libraries**: 
  - `langchain-openai` for LLM integration
  - `requests` for HN API calls  
  - `click` for CLI interface
  - `tiktoken` for token counting

## Core Components

- `get_user()`: Fetches user data from HN API
- `get_all_user_comments()`: Retrieves user's comments (limited to 30)
- `infer_profile_from_comments()`: Uses GPT-4 mini to analyze comments and generate profile
- `count_tokens_for_input()`: Estimates token usage before API calls

## Development Commands

```bash
# Install dependencies
make install

# Update dependencies
make compile-deps

# Run the tool
python main.py <username>
```

## Environment Requirements

- Requires OpenAI API key (set as environment variable)
- Python 3.11+ with virtual environment in `hnprofiles/`
- Dependencies managed with `uv` package manager

## API Integration

- **Hacker News API**: `https://hacker-news.firebaseio.com/v0`
- **OpenAI**: Uses `gpt-4o-mini` model via langchain
- Rate limiting: Processes max 30 user submissions

## Caching

- **Cache Directory**: `.cache/` (created automatically)
- **HN API Cache**: `.cache/hn_api/` - Caches all API responses from Hacker News
- **OpenAI Cache**: `.cache/openai/` - Caches AI analysis results
- **Cache Keys**: MD5 hash of request content for deterministic caching
- **Benefits**: Reduces API costs and improves development speed during testing
- **Cache Control**: Use `--nocache` flag on commands to skip cache and make fresh requests