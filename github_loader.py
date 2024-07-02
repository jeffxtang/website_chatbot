from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import GithubFileLoader
from langchain_community.document_loaders import GitHubIssuesLoader
import config
import os

os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')

# https://python.langchain.com/v0.2/docs/integrations/document_loaders/github/
# Load github issues and PRs - API doc: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.github.GitHubIssuesLoader.html
loader = GitHubIssuesLoader(
    repo=config.GITHUB_REPO_PATH,
    #include_prs=False,
    state='all'
)

doc_issues = loader.load()
print(f"number of issues: {len(doc_issues)}")

# load all markdowns files in a repo - API doc: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.github.GithubFileLoader.html
loader = GithubFileLoader(
    repo=config.GITHUB_REPO_PATH,
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(
        ".md"
    ),  
)
docs_files = loader.load()
print(f"number of MD files: {len(docs_files)}")

all_docs = doc_issues + docs_files

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
splits = text_splitter.split_documents(all_docs)
print(f"{len(splits)} splits")

print(f"indexing and saving to vector db at {config.VECTOR_DB_PATH}")

embeddings = HuggingFaceEmbeddings(model_name='hkunlp/instructor-large', #'sentence-transformers/all-MiniLM-L6-v2',                                    
                                   model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(splits, embeddings)
db.save_local(config.VECTOR_DB_PATH)
