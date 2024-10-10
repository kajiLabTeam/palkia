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
	poetry run pylint src tests

format:
	poetry run black src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache

run:
	poetry run python src/main.py

docs:
	poetry run pdoc --html --output-dir docs src

check: lint test

# 開発環境のセットアップ
setup: install
	git config core.hooksPath .githooks

# 新しい機能ブランチの作成
feature:
	@read -p "Enter feature name: " name; \
	git checkout -b feature/$$name

# リリースブランチの作成
release:
	@read -p "Enter version number: " version; \
	git checkout -b release/$$version

# デプロイ（例：Herokuにデプロイする場合）
deploy:
	git push heroku main

# データベースのマイグレーション
migrate:
	poetry run python src/manage.py migrate

# 開発サーバーの起動
serve:
	poetry run python src/manage.py runserver