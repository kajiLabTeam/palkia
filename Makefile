.PHONY: install test lint format clean run docs check

# poetryがインストールされていることを確認
POETRY := $(shell command -v poetry 2> /dev/null)

install:
ifndef POETRY
	@echo "Poetry is not installed. Please install it first."
	@exit 1
endif
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check src

lint-fix:
	poetry run ruff check src --fix

format:
	poetry run ruff format --check src
	
format-fix:
	poetry run ruff format src

pyright:
	poetry run pyright src

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache

run:
	cd src && \
	poetry run python main.py

release:
	poetry version patch
	git add pyproject.toml dist
	poetry build
	git commit -m "Bump up version"
	git push origin main
	poetry publish

docs:
	poetry run pdoc --html --output-dir docs src

ci: lint format pyright

# 開発環境のセットアップ
setup: install
	git config core.hooksPath .githooks

# 新しい機能ブランチの作成
feature:
	@read -p "Enter feature name: " name; \
	git checkout -b feature/$$name


# デプロイ（例：Herokuにデプロイする場合）
deploy:
	git push heroku main

# データベースのマイグレーション
migrate:
	poetry run python src/manage.py migrate

# 開発サーバーの起動
serve:
	poetry run python src/manage.py runserver