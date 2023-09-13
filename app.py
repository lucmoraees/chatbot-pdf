from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.callbacks import get_openai_callback

def main():
  load_dotenv()
  model = os.getenv('OPENAI_API_MODEL')

  # Loading the front-end
  st.set_page_config(page_title='Ask your PDF')
  st.header("Ask your PDF 💬")

  # Uploading the PDF file
  pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

  # Extracting the text from the PDF file
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    # Splitting the text into chunks
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
    )

    chunks = text_splitter.split_text(text)

    # Create the embeddings
    embeddings = OpenAIEmbeddings(model=model, chunk_size=1, max_retries=10)
    knowladge_base = FAISS.from_texts(chunks, embeddings)

    # Show user input
    user_question = st.text_input("Ask your question about your PDF")
    
    # Setting prompt
    prompt = "Responda em portugês com no máximo 50 palavras. "

    if user_question:
      doc = knowladge_base.similarity_search(user_question)
      llm = AzureOpenAI(deployment_name=model)
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=doc, question=prompt+user_question)
        print(cb)

      st.write(response)

if __name__ == '__main__':
  main()
  