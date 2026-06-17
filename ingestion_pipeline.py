import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
        glob="*.txt",
        loader_kwargs= {
            "encoding":"utf-8"#specify encoding otherwise it wont work
        }
    )

    #load the documents into documents
    documents=loader.load() #documents is a list [] with elements called documents(pagecontent,metadata)

    #check if files are there in documents
    if(len(documents)==0):
        raise FileNotFoundError(f"no .txt files exist in {docs_path}")
    
    #show a few contents present in the documents list
    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument{i+1}")
        print(f"Content Length: {len(doc.page_content)}")
        print(f"Content: {doc.page_content[0:100]}")
        print(f"Source: {doc.metadata["source"]}")

    return documents

def chunk_documents(documents,chunk_size=500,chunk_overlap=0):
    print("Preparing to chunk the documents...")

    #initialize the char text splitter
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks=text_splitter.split_documents(documents) #create chunks instance

    for i,chunk in enumerate(chunks[:5]):
        print(f"\nChunk{i+1}")
        print(f"Source: {chunk.metadata["source"]}")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Content: {chunk.page_content}")

    return chunks

def create_vector_embeddings_vectorstore(chunks,persist_directory="db/chroma_db"):
    print("setting up embeddings for chunks and storing in ChromaDB...")

    #select the embedding model
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #creating the chromaVECTOR STORE
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print(f"\nVector Created in {persist_directory}")

    return vector_store


def main():
    
    print("|STARTING INGESTION PIPELINE|")
    documents=load_documents(docs_path="docs") #1 Loading the Documents
    chunks=chunk_documents(documents)          #2 Chunking them
    create_vector_embeddings_vectorstore(chunks)#convert to vector embeddings and store in vectorStore

if __name__ == "__main__":
    main()