# SiteChat
Building a chatbot to chat about any web site (documentation) or github repo

## Setting up the environment

First, set up a conda or virtual environment and install the required packages:
```
pip install -r requirements.txt
```

Then add two environment variables:
* Get a free `LANGCHAIN_API_KEY` from [LangChain](https://smith.langchain.com/), and:
```bash
export LANGCHAIN_API_KEY=<your_langchain_api_key>
```
* Get a free trial GROQ_API_KEY at [Groq.com](https://groq.com/), then:
```bash
export GROQ_API_KEY=<your_groq_api_key>
```

## Indexing the web site

First, set `WEB_SITE_URL` and `VECTOR_DB_PATH` in `config.py` such as:
```
WEB_SITE_URL = "https://github.com/meta-llama/llama-recipes"
VECTOR_DB_PATH = 'vectorstore/llama-recipes'
```

Then, build the index by running `python index.py`, with the default device type as `cpu`, or if you have a GPU, `python index.py --device cpu`.

## Running the Application

To start the chatbot:
```
streamlit run main.py
```
