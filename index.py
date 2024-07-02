import argparse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='device type') 
# change 'cpu' to 'cuda' on machine with GPU

# Create vector database
def create_vector_db(device):
    print(f'loading web site {config.WEB_SITE_URL}...')

    loader = RecursiveUrlLoader(
        config.WEB_SITE_URL, # "https://docs.python.org/3.9/",
        max_depth=2,
        # use_async=False,
        # extractor=None,
        # metadata_extractor=None,
        # exclude_dirs=(),
        # timeout=10,
        # check_response_status=True,
        # continue_on_failure=True,
        # prevent_outside=True,
        # base_url=None,
        # ...
    )

    documents = loader.load()
    print(f"{len(documents)} documents loaded, 1st doc: {documents[0]}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    print(f"{len(splits)} splits")

    docs =[]
    for idx, split in enumerate(splits):
        split.metadata.update({'id':idx})
        docs.append(split)

    print(f"indexing and saving to vector db at {config.VECTOR_DB_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name='hkunlp/instructor-large', 
                                        #'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': device})

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(config.VECTOR_DB_PATH)

if __name__ == "__main__":
    args = parser.parse_args()
    create_vector_db(args.device)
