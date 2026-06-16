import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"loading files from {docs_path}...") #use f string to print out docs_path variable

    #check if the path even exist and do error handling
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} does not exist...")
    
    #initialize an instance (object) of the class directory_loader
    loader=DirectoryLoader(
        path=docs_path,
        loader_cls=TextLoader,
        glob="*.txt"
    )

def main():
    print("Starting ingestion pipeline...")
    load_documents()


if __name__ == "__main__":
    main()