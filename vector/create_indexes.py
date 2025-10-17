from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
import tqdm
import yaml


def load_yaml_config():
    with open("project_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_yaml_config()

def load_model(
    model_name = config["embeddings"]["name"],
    model_kwargs = {"device": config["embeddings"]["device"]}):
    
    return HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

def loading_documents(directory: Path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config["TextSplitter"]["chunk_size"],
        chunk_overlap = config["TextSplitter"]["chunk_overlap"]
    )
    
    documents = []
    for document in tqdm.tqdm(directory.iterdir(), bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', colour='yellow'):
        loader = PyPDFLoader(document)
        documents.extend(loader.load_and_split(
            text_splitter=text_splitter
            )
        )
    return documents


def load_db(embedding_function, save_path=config["faiss_indexstore"]["save_path"],
            index_name=config["faiss_indexstore"]["index_name"]):
    db = FAISS.load_local(folder_path=save_path, index_name=index_name, embeddings=embedding_function)
    return db

def save_db(db, save_path=config["faiss_indexstore"]["save_path"], index_name=config["faiss_indexstore"]["index_name"]):
    db.save_local(save_path, index_name)
    print(f"Saved db into:\nsave path: {save_path}\nindex name: {index_name}")
