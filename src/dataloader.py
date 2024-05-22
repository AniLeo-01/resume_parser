import pandas as pd
import os
from google.cloud import storage
from urllib.parse import urlparse, unquote
from tqdm import tqdm
import PyPDF2

def download_files_from_bucket(bucket_name, file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)

    return True

def fetch_file_names_from_csv(csv_file_path: str):
    data = pd.read_csv(csv_file_path)
    urls = data['url'].to_list()
    filenames = [unquote(url.split("/")[-1]).split('.pdf')[0]+'.pdf' for url in urls]
    return filenames

def download_files_from_urls(csv_filepath: str):
    destination_file_name = "./resume_data/"
    filenames = fetch_file_names_from_csv(csv_filepath)
    for idx, filename in tqdm(enumerate(filenames), total=len(filenames)):
        try:
            download_files_from_bucket(os.getenv("BUCKET_NAME"), filename, destination_file_name + filename)
        except Exception as e:
            print(e)
            continue

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text.append(page.extract_text())
        return "\n".join(text)
    
def extract_resume_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")
    
def load_data_from_files():
    resume_texts = []
    for file_name in tqdm(os.listdir('resume_data')):
        file_path = os.path.join('./resume_data/', file_name)
        try:
            text = extract_resume_text(file_path)
        except Exception as e:
            print(e)
            continue
        resume_texts.append(text)

    return resume_texts