from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from flask import Flask
import os
from api_key import api
os.environ["OPENAI_API_KEY"]=api
import streamlit as st
def pdf_loader(pathname):
  pdf=PyPDFLoader(f"./{pathname}")
  pages = pdf.load_and_split()
  db = Chroma.from_documents(pages, OpenAIEmbeddings())
  return db
def similarity_search(query,path):
  db = pdf_loader(path)
  docs = db.similarity_search(query, k=2)
  llm = ChatOpenAI()
  template=PromptTemplate(
    input_variables=["query","docs"],
    template="Act as pdf Reader and answer the {query} and take help of document {docs}"
  )
  llm_chain=LLMChain(llm=llm,prompt=template,verbose=True)
  result=llm_chain.run(query=query,docs=docs)
  return result
def main():
  st.title("PdfReaderChatGpt")
  pathname=loader_interface()
  if pathname:
    st.success("Pdf Uploaded Successfully")
  prompt=st.text_input("Enter the prompt here")
  if prompt:
    with st.expander("Response here"):
      result=similarity_search(prompt,pathname)
      st.write(result)
def  loader_interface():
  uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
  if uploaded_file is not None:
    save_path = Path("/Users/jaswanthsudha/Desktop/Langchain", uploaded_file.name)
    with open(save_path, mode='wb') as f:
        f.write(uploaded_file.getvalue())
    return uploaded_file.name
if __name__=="__main__":
  main()

