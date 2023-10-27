from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone

import os
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

directory = './data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(len(documents))


def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# initialize pinecone
pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)

index_name = "faq" # your pinecone index name

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

print(index)

