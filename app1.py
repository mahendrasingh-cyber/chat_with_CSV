import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from CSV files
def get_csv_text(csv_files):
    text = ""
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        text += df.to_string(index=False)
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store for the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to define a conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "The answer is not available in the context." 
    Do not provide a wrong answer.

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide AI-generated responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat CSV", page_icon="📊")
    st.header("Chat with CSV Files using Gemini 📊💬")

    user_question = st.text_input("Ask a Question from the CSV Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        csv_files = st.file_uploader(
            "Upload your CSV Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["csv"],
        )
        if st.button("Submit & Process"):
            if csv_files:
                with st.spinner("Processing..."):
                    raw_text = get_csv_text(csv_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done!")
            else:
                st.warning("Please upload at least one CSV file.")

if __name__ == "__main__":
    main()
