import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer, util

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
DB_FAISS_PATH = 'vectorstore/db_faiss'
QUERY_RESPONSE_FILE = 'car_insurance_queries_responses.json'


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question in consise manner from the provided context, make sure to provide all the necessary details.Paraphrarse it such that user can able to understant it.,\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    # st.write("Reply: ", response["output_text"])
    return response["output_text"]


pdf_path = 'policy-booklet-0923.pdf'
raw_text = get_pdf_text([pdf_path])
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)


# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def evaluate_model():
    with open(QUERY_RESPONSE_FILE, 'r') as file:
        data = json.load(file)

    correct_responses = 0
    total_queries = len(data)

    for item in data:
        query = item['query']
        expected_response = item['response']
        actual_response = user_input(query)

        # Compute cosine similarity between expected_response and actual_response embeddings
        embeddings1 = model.encode([expected_response.strip()])
        embeddings2 = model.encode([actual_response.strip()])
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

        # Use a threshold to determine if the responses are sufficiently similar
        if similarity > 0.7:  # Adjust threshold as needed
            correct_responses += 1

    accuracy = correct_responses / total_queries
    return accuracy

st.title("Chat with PDF : Ask any query related to Car insurance ")  

# Initialize chat history only if it hasn't been initialized yet
if 'history' not in st.session_state:
    st.session_state['history'] = []
    st.session_state['generated'] = ["Hello! Ask me anything 🤗"]
    st.session_state['past'] = ["Hey! 👋"]
 

        # Container for the chat history
response_container = st.container()
        # Container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_question = st.text_input("Query:", placeholder="Talk to your PDF data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
            
    if submit_button and user_question:
        output = user_input(user_question)
                
        st.session_state['past'].append(user_question)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
