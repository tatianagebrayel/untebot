import os
import re
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import langid
from deep_translator import GoogleTranslator
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Setup logging
logging.basicConfig(level=logging.DEBUG)

OPENAI_API_TOKEN = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

# Retrieve the secret token from environment variables
hf_api_token = os.getenv('HF_API_TOKEN')

# Ensure the token is not None
if hf_api_token is None:
    raise ValueError("HF_API_TOKEN environment variable not set")

# Fixing random seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Function to translate text based on detected language
def translate_content(text):
    try:
        detected_lang = detect(text)
        if detected_lang == 'fr':
            return GoogleTranslator(source='fr', target='en').translate(text)
        elif detected_lang == 'en':
            return GoogleTranslator(source='en', target='fr').translate(text)
        else:
            return text
    except Exception as e:
        print(f"Error detecting language or translating: {e}")
        return text




# Function to chunk content
def chunk_content(content, chunk_size=2000, overlap=100):
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Initialize the list to store chunked documents
chunked_web_doc = []


# Load the fetched content from the text file
with open('fetched_contenttt.txt', 'r', encoding='utf-8') as f:
    fetched_content = f.read()


#web_contents = content.split("-" * 80 + "\n\n")
web_contents = fetched_content.split("-" * 80 + "\n\n")


for block in web_contents:
    if block.strip():
        lines = block.strip().splitlines()
        url = ""
        title = ""
        en_content = ""
        fr_content = ""
        language = None

        for i, line in enumerate(lines):
            if line.startswith("URL:"):
                url = line.split("URL:")[1].strip()
            elif line.startswith("Title:"):
                title = line.split("Title:")[1].strip()
            elif line == "English Content:":
                language = "en"
            elif line == "French Content:":
                 language = "fr"
            else:
                if language == "en":
                    en_content += line + "\n"
                elif language == "fr":
                    fr_content += line + "\n"

        if en_content.strip():
            en_chunks = chunk_content(en_content.strip())
            for chunk in en_chunks:
                chunked_web_doc.append({
                    "url": url,
                    "language": "en",
                    "chunk": chunk
                })

        if fr_content.strip():
            fr_chunks = chunk_content(fr_content.strip())
            for chunk in fr_chunks:
                chunked_web_doc.append({
                    "url": url,
                    "language": "fr",
                    "chunk": chunk
                })


documents = [
    Document(page_content=chunk['chunk'], metadata={"url": chunk['url'], "language": chunk['language']})
    for chunk in chunked_web_doc
]

model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs=model_kwargs
)



chroma_db = Chroma.from_documents(documents=documents,
                                  collection_name='rag_web_db',
                                  embedding=embeddings,
                                  collection_metadata={"hnsw:space": "cosine"},
                                  persist_directory="./web_db")

similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 3,
                                                                       "score_threshold": 0.3})


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


################ history_aware_retriever###################


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, similarity_threshold_retriever, contextualize_q_prompt
)


################ question_answer_chain#####################


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


################ rag_chain#####################


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []


def ask(question, chat_history):
    prepended_phrase = " in Moodle.USJ :"
    question = question.lower()
    modified_question = prepended_phrase + question 

    
    # Invoke the chain to get the response
    ai_message = rag_chain.invoke({"input": modified_question, "chat_history": chat_history})
    chat_history.append(("user", question))
    
    answer = ai_message["answer"]

    # Prepare document links if available
    document_links = []
    for doc in ai_message.get('context', []):
        if 'url' in doc.metadata:
            document_links.append(doc.metadata['url'])
    
    # Remove duplicate links by converting to a set and back to a list
    #unique_document_links = list(set(document_links))

    # Append the question and answer to the chat history (without sources)
    chat_history.append(("assistant", answer))

    # For display purposes, format the chat history without labels
    display_chat_history = []
    for role, content in chat_history:
        if role == "user":
            display_chat_history.append((None, content))  # User question on the right
        else:
            display_chat_history.append((content, None))  # Assistant answer on the left

    # Add sources to the last assistant message for display purposes only
    if document_links:
        document_links_text = "\n".join(document_links)
        display_chat_history[-1] = (display_chat_history[-1][0] + f"\nSources: {document_links_text}", None)

    # Return display history for the UI, and the actual chat history for internal use
    return display_chat_history, chat_history, ""

def copy_chat_history(chat_history):
    chat_text = "\n".join([f"{role}: {content}" for role, content in chat_history])
    return chat_text



# Initialize the Gradio interface
#with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
#    chatbot = gr.Chatbot()
#    question = gr.Textbox(placeholder="Ask me anything about Moodle...")
#    clear_button = gr.Button("Clear")
#    copy_button = gr.Button("copy Chat History")
#    chat_history = gr.State([])  # State to keep chat history
#    hidden_textbox = gr.Textbox(label="Copy Your Chat History", show_copy_button=True, visible=False)  # Hidden chat history

    # Ensure the ask function returns appropriate values
#    question.submit(ask, [question, chat_history], [chatbot, chat_history, question])
#    clear_button.click(lambda: ([], [], ""), None, [chatbot, chat_history, question], queue=False)

    # Button to download chat history as a text file
    #copy_button.click(copy_chat_history, [chat_history], [gr.Textbox(label="copy Your Chat History",show_copy_button=True)])
    # Button to allow copying chat history
    #copy_button.click(copy_chat_history, [chat_history], [hidden_textbox])

#    def show_copy_button_fn(chat_history):
#        return gr.update(value=copy_chat_history(chat_history), visible=True)

#    copy_button.click(show_copy_button_fn, [chat_history], [hidden_textbox])

# Define the title as a Markdown string
title = "# **Moodle assistant**"

# Modify the existing ask function to initialize with a system message
def initialize_chat():
    # Start the conversation with a system message
    initial_prompt = "Hello,how can i help you with Moodle."
    chat_history = [("assistant", initial_prompt)]
    display_chat_history = [(initial_prompt, None)]  # System message on the left
    return display_chat_history, chat_history, ""


with gr.Blocks(css=".title {display: flex; justify-content: center;}", theme=gr.themes.Soft()) as demo:
    # Add the title at the top with an id for custom CSS
    gr.Markdown(value=title, elem_id="title")
    
    chatbot = gr.Chatbot()
    question = gr.Textbox(placeholder="Ask me about Moodle..")
    chat_history = gr.State([])

    # Initialize the chat with the system prompt
    clear_button = gr.Button("Clear")
    
    # Load the initial prompt when the app launches
    demo.load(initialize_chat, None, [chatbot, chat_history])

    question.submit(ask, [question, chat_history], [chatbot, chat_history, question])
    
    # Clear button functionality
    clear_button.click(initialize_chat, None, [chatbot, chat_history], queue=False)

demo.queue()
demo.launch(share=False)

demo.queue()
demo.launch(share=True)


