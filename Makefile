
compile-deps:
	uv pip compile -o requirements.txt requirements.in


install:
	uv pip install -r requirements.txt