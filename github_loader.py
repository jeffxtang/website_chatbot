from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.document_loaders import GithubFileLoader

import os

os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')

def issues_files_load():
    # https://python.langchain.com/v0.2/docs/integrations/document_loaders/github/
    loader = GitHubIssuesLoader(repo="meta-llama/llama-recipes", state='all')
    docs = loader.load()

    # https://python.langchain.com/v0.2/docs/integrations/document_loaders/github/
    # Load github issues and PRs - API doc: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.github.GitHubIssuesLoader.html
    loader_issues = GitHubIssuesLoader(
        repo="meta-llama/llama-recipes",
        #include_prs=False,
        state='all'
    )

    docs_issues = loader_issues.load()

    # load all markdowns files in a repo - API doc: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.github.GithubFileLoader.html
    loader_files = GithubFileLoader(
        repo="meta-llama/llama-recipes",
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(
            ".md"
        ),  
    )
    docs_files = loader_files.load()

    all_docs = docs_issues + docs_files
    return all_docs
