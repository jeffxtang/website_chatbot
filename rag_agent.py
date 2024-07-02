import langchain

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, chain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together

from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter

import streamlit as st
import os
import config
import json
import datetime
from io import StringIO
from langchain_core.tools import tool
import re

from langchain.document_loaders import GithubFileLoader
from langchain_community.document_loaders import GitHubIssuesLoader

os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')

def issues_files_load():
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
    return all_docs

source_string = ""

@tool
def repo_statistics(stat):
    """Returns the statistics of a repo, including number of total issues, closed issues, open issues, total PRs, closed PRs, open PRs."""
    
    global source_string 
    source_string = ""
    if "number_of_total_issues" == stat:
        return st.session_state.github_stat['issues']
    elif "number_of_closed_issues" == stat:
        return st.session_state.github_stat['issues_closed']
    elif "number_of_open_issues" == stat:
        return st.session_state.github_stat['issues_open']
    elif "number_of_total_prs" == stat:
        return st.session_state.github_stat['prs']
    elif "number_of_closed_prs" == stat:
        return st.session_state.github_stat['prs_closed']
    elif "number_of_open_prs" == stat:
        return st.session_state.github_stat['prs_open']
    else:
        return -1

@tool
def rag_query(prompt):
    """Search the repo for specific content, such as info about agents, fine-tuning"""
    global source_string 

    print(f">>>chat_history: {st.session_state.chats[st.session_state.current_chatnum-1]}")
    answer = custom_chain.invoke(
        {
            "question": prompt,
            "chat_history": st.session_state.chats[st.session_state.current_chatnum-1]
        }
    )
    
    response = answer['response']
    docs = answer['source_document']

    sources = []
    for doc in docs:
        if 'source' in doc.metadata:
            if doc.metadata['source'] in sources:
                continue
            sources.append(doc.metadata['source'])

    source_string = ""
    if len(sources) == 1:
        source_string = "Source:\n"
    elif len(sources) > 1:
        source_string += "Sources:\n"    
    for source in sources:
        source_string += "\n> - " + source.replace("api.github.com", "github.com") + "\n"

    return(response)

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
First, use Thought to describe your thoughts about the question you have been asked, 
and generate Action in the following format:
Action: tool_name: tool_parameter

Then, return PAUSE.

Observation will be the result of running those tools. If the observation looks good as the answer, return it as the final answer.

Your available tool names with descrptions are as follows:
1. repo_statistics: useful for returning statistics about a repo, such as the number of total, closed or open issues or prs.
2. rag: useful for searching for specific content in a repo, such as info about agents, fine-tuning.

Examples to call the tools with tool name and parameter:
repo_statistics: number_of_closed_issues
repo_statistics: number_of_open_issues
repo_statistics: number_of_total_issues
repo_statistics: number_of_closed_prs
repo_statistics: number_of_open_prs
repo_statistics: number_of_total_prs
rag_query: original__question

""".strip()

known_actions = {
    "repo_statistics": repo_statistics,
    "rag_query": rag_query
}
action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action

# TODO: 1. show doc sources for RAG tool; 2. add chat history
def react(llm, question, max_turns=5):
    i = 0
    bot = Agent(llm, system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(f"***{result=}***")
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))

            # key to make the agent process fully automated:
            # programtically call the external func with arguments, with the info returned by LLM
            observation = known_actions[action](action_input) 

            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            # return substring starting after Answer:
            index = result.find("Answer: ")
            if index == -1:
                return "Sorry, I can't find any answers. Please rephrase your question."
            start_index = index + len("Answer: ")
            return result[start_index:]            

class Agent:
    def __init__(self, llm, system=""):
        self.llm = llm
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = self.llm.invoke(self.messages)
        return response.content



os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

embeddings = HuggingFaceEmbeddings(model_name='hkunlp/instructor-large', 
                                        #'sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu', }) # cuda, cpu

db = FAISS.load_local(config.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(
                search_type="similarity_score_threshold", # mmr # similarity_score_threshold
                search_kwargs={"k": 3, "score_threshold": 0.5}
)

from langchain_groq import ChatGroq
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

search_template = """
You are a helpful, respectful and honest assistant.
Given a user question, your task is to enrich the user question with more relevant information like explanation of any acronym.
You can also summarize the chat history to add relevant context to the user question.
Do not ask any follow up question to the users.

Chat history:
{chat_history}
User question:
{question}
"""
SEARCH_PROMPT = PromptTemplate.from_template(search_template)

answer_template = """
You are a helpful, respectful and honest assistant.
If question is related to the context, you should ONLY use the provided CONTEXT to answer the question. DO NOT USE INFORMATION NOT IN THE CONTEXT.
If question is not related to the context, respond like a general AI assistant.

<CONTEXT>:
{context}
</CONTEXT>

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{source}: {page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    print(f"{doc_strings=}")
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x['chat_history'][-6:])
    )
    | SEARCH_PROMPT
    | llm
    | StrOutputParser(),
)

chat_chain = (ANSWER_PROMPT | llm)

@chain
def custom_chain(inputs):
    refined_question = _inputs.invoke(inputs)     
    search_query = """
    Question: {question}
    {context_from_llm}
    """
    query_format = {"question": prompt, "context_from_llm": refined_question['standalone_question']}
    search_query = search_query.format(**query_format)
    print(search_query)
    docs = retriever.invoke(search_query)
        
    for doc in docs:
        if "url" in doc.metadata: 
            # github issues/prs return 'url' in metadata, markdowns return 'source'; 
            # but format_document in _combine_documents requires 'source' a doc's metadata 
            doc.metadata['source'] = doc.metadata['url']    

    context = _combine_documents(docs)

    answer = chat_chain.invoke({"context": context, "question": inputs['question']})
    print(f"{answer.content=}")
    return {'response': answer.content, 'source_document': docs}

st.title("RepoChat")
st.markdown("Chat about a Github Repo: the markdowns, issues, and PRs")

if "github_stat" not in st.session_state:
    st.session_state.github_stat = {}
    all_docs = issues_files_load()
    prs, prs_closed, prs_open, issues, issues_closed, issues_open, mds = 0, 0, 0, 0, 0, 0, 0
    for doc in all_docs:
        if 'is_pull_request' in doc.metadata:
            if doc.metadata['is_pull_request']:
                prs += 1            
                if doc.metadata['state'] == 'open':
                    prs_open += 1
                else:
                    prs_closed += 1
            else:
                issues += 1            
                if doc.metadata['state'] == 'open':
                    issues_open += 1
                else:
                    issues_closed += 1    
        else:
            mds += 1

    st.session_state.github_stat['prs'] = prs
    st.session_state.github_stat['prs_open'] = prs_open
    st.session_state.github_stat['prs_closed'] = prs_closed
    st.session_state.github_stat['issues'] = issues
    st.session_state.github_stat['issues_open'] = issues_open
    st.session_state.github_stat['issues_closed'] = issues_closed
    st.session_state.github_stat['mds'] = mds

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append([])

if 'chats' not in st.session_state:
    st.session_state.chats = []
    st.session_state.chats.append([])

# total number of chats
if "total_chatnum" not in st.session_state:
    st.session_state.total_chatnum = 1

if "current_chatnum" not in st.session_state:
    st.session_state.current_chatnum = 1

def new_chat(args):
    st.session_state.current_chatnum = st.session_state.total_chatnum
    # increase chatnum only if there's at least one QA
    show_chats(None)
    if len(st.session_state.chats) > 0 and len(st.session_state.chats[st.session_state.current_chatnum - 1]) >=2:
        st.session_state.total_chatnum += 1
        st.session_state.messages.append([])
        st.session_state.chats.append([])
    st.session_state.current_chatnum += 1

def show_chats(index):
    with st.sidebar:
        st.button("New Chat",
            args=("new_chat",),
            on_click=new_chat
        )

    chats = []
    for i in range(len(st.session_state.chats)):
        if len(st.session_state.chats[i]) > 0:
            chats.insert(0, st.session_state.chats[i][0].content)

    print(f"chats={chats}")

    if len(chats) == 0:
        return

    with st.sidebar:
        add_radio = st.radio(
            "Chats",
            chats,
            index if index==None else st.session_state.total_chatnum - index,
            key=f"chats_titles{st.session_state.total_chatnum}",
            on_change=change_chats
        )

def change_chats():
    key = f"chats_titles{st.session_state.total_chatnum}"
    # find the chatnum for the radio option key (title) st.session_state[key]
    for i in range(len(st.session_state.chats)):
        if len(st.session_state.chats[i]) > 0:
            if st.session_state.chats[i][0].content == st.session_state[key]:
                st.session_state.current_chatnum = i+1
                break

    show_chats(st.session_state.current_chatnum)

# Display chat messages (both prompts and answers) from history
for message in st.session_state.messages[st.session_state.current_chatnum-1]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask RepoChat")

if prompt:
    st.chat_message("user").markdown(prompt)
    answer = react(llm, prompt)
    # answer = custom_chain.invoke(
    #     {
    #         "question": prompt,
    #         "chat_history": st.session_state.chats[st.session_state.current_chatnum-1]
    #     }
    # )

    response = answer
    if source_string != "":
        response += "\n> " + source_string

    # response = answer['response']
    # docs = answer['source_document']

    # sources = []
    # if len(docs) > 0:
    #     # response += "\n> Source"
    #     # if len(docs) == 1:
    #     #     response += ":\n"
    #     # else:
    #     #     response += "s:\n"

    #     for doc in docs:
    #         if 'source' in doc.metadata:
    #             if doc.metadata['source'] in sources:
    #                 continue
    #             sources.append(doc.metadata['source'])
    #             response += "\n> - " + doc.metadata['source'].replace("api.github.com", "github.com") + "\n"
    # else:
    #    response = "Sorry, I can't find any relevant documents. Please rephrase your question."

    st.chat_message("assistant").markdown(response)

    st.session_state.messages[st.session_state.current_chatnum - 1].append({"role": "user", "content": prompt})
    st.session_state.messages[st.session_state.current_chatnum - 1].append({"role": "assistant", "content": response})

    st.session_state.chats[st.session_state.current_chatnum - 1].append(HumanMessage(content=prompt))
    st.session_state.chats[st.session_state.current_chatnum - 1].append(AIMessage(content=response))
    # st.session_state.chats[st.session_state.current_chatnum - 1].append(AIMessage(content=docs))

    # Any time something must be updated on the screen, Streamlit reruns your entire
    # Python script from top to bottom.
    # Whenever a callback is passed to a widget via the on_change (or on_click) parameter,
    # the callback will always run before the rest of your script.
    # Session State provides a dictionary-like interface where you can save information
    # that is preserved between script reruns.
    show_chats(st.session_state.current_chatnum)
