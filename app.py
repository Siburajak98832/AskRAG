import streamlit as st
import os
from io import BytesIO
import numpy as np
from docx import Document
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from huggingface_hub import login, InferenceClient
from dotenv import load_dotenv
import faiss

load_dotenv()
huggingface_api_key = os.getenv('huggingface_api_key')

@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    documents = []

    if input_type == "Link":
        for url in input_data:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        if not text.strip():
            st.error("The PDF contains no readable text. It may be a scanned image or image-based PDF.")
            return None
        documents = text
    elif input_type == "Text":
        documents = input_data
    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = text
    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
        documents = text
    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents) if isinstance(documents, list) else text_splitter.split_text(documents)
    if not texts:
        st.error("No valid text chunks could be created. Please upload a different document.")
        return None
    if isinstance(texts[0], str):
        processed_texts = texts
    else:
        processed_texts = [doc.page_content for doc in texts]

    hf_embeddings = load_embedding_model()
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(processed_texts)
    return vector_store

def answer_question(vectorstore, query):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Truncate context if it's too long
    max_tokens = 3500
    if len(context.split()) > max_tokens:
        context = " ".join(context.split()[:max_tokens])

    full_prompt = f"""You are a helpful and knowledgeable assistant.
        Based only on the context provided below, identify and explain up to 10 key challenges related to the user's question.

        Each point must include:

        A brief, clear title for the challenge

        A 1–2 sentence explanation of the issue, grounded in the context

        Respond using bullet points. Do not exceed 10 bullets.
        Do not list challenges without explanation.

        If no relevant information is found in the context, reply with:
        “I couldn't find that in the context.”


Context:
        
{context}

Question:
{query}

Answer:"""

    client = InferenceClient(provider="featherless-ai", api_key=huggingface_api_key)
    completion =client.chat_completion(
    model="HuggingFaceH4/zephyr-7b-beta",
    messages=[
        {"role": "system", "content": "You are a concise, helpful assistant. Use only the context provided."},
        {"role": "user", "content": full_prompt}
    ],
    max_tokens=500,
    temperature=0.7
)
    return completion.choices[0]["message"]["content"], context



def main():
    st.title("AskRAG")
    st.write( "Answers your questions by retrieving and reasoning over PDFs, links, text files, and more using Retrieval-Augmented Generation (RAG).")
    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    input_data = None
    if input_type == "Link":
        number_input = st.number_input("Enter number of Links", min_value=1, max_value=10, step=1)
        input_data = [st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(number_input)]
    elif input_type == "Text":
        input_data = st.text_area("Enter the text")
    else:
        accepted_types = {"PDF": "pdf", "TXT": "txt", "DOCX": ["docx", "doc"]}
        input_data = st.file_uploader(f"Upload a {input_type} file", type=accepted_types.get(input_type, []))

    if st.button("Proceed"):
        if not input_data:
            st.warning("Please provide a valid input.")
            return
        with st.spinner("Processing input and generating vector store..."):
            vectorstore = process_input(input_type, input_data)
            st.session_state["vectorstore"] = vectorstore
        st.success("Input processed successfully!")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            with st.spinner("Generating answer..."):
                answer, context = answer_question(st.session_state["vectorstore"], query)
                st.markdown(f"**Answer:**\n{answer}")
                if st.checkbox("Show retrieved context"):
                    st.text(context)

if __name__ == "__main__":
    main()




