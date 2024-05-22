from dotenv import load_dotenv
from src.dataloader import download_files_from_urls, load_data_from_files
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os

load_dotenv()

def get_embeddings(text):
     # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"))
    model = AutoModel.from_pretrained(os.getenv("MODEL_NAME"))
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def get_document_embeddings(input_text: str):
    doc_embeddings = [get_embeddings(text=text) for text in input_text]
    return doc_embeddings

def load_and_write_to_faiss(embeddings):
    # Convert list of embeddings to a numpy array
    embeddings_array = np.vstack(embeddings)

    # Define the dimension
    d = embeddings_array.shape[1]

    # Build the index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)

    # Save the index for later use
    faiss.write_index(index, 'src/embeddings/resume_index.faiss')

def embed_resume_texts():
    try:
        if os.path.exists('src/embeddings/resume_index.faiss'):
            print("Embeddings exist!")
            return True
        else:
            resume_texts = load_data_from_files()
            doc_embeddings = get_document_embeddings(resume_texts)
            load_and_write_to_faiss(doc_embeddings)
            return True
    except Exception as e:
        print(e)
        return False

def search(query, k=5):
    # Load the index
    index = faiss.read_index('src/embeddings/resume_index.faiss')

    query_embedding = get_embeddings(query)
    D, I = index.search(query_embedding, k)
    # return I  # return indices of top-k resumes
    resume_texts = load_data_from_files()
    search_output = {}
    for idx in I[0]:
        search_output[idx] = resume_texts[idx]

    return search_output