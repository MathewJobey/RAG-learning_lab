from langchain_chroma import Chroma 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
persist_directory="db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(
    collection_metadata={"hnsw:space":"cosine"},
    persist_directory=persist_directory,
    embedding_function=embedding_model
)
#to reopen the vector store we provide the location and tell which embedidng fxn to use to convert the user query to embeddings

query="In what year did Tesla began production of the car called Roadster?"

#creating the retriever
#retriever=db.as_retriever(search_kwargs={"k":3}) #retrieve top 3 scored similarity embeddings from the vector store
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.5 #return vectors with similarity threshold>=0.3,
        }
)

#use the query to invoke the db
relevant=retriever.invoke(query)

#print and show
print(f"\nUser Query: {query}")

print("---CONTEXT---")
for i,doc in enumerate(relevant,1): #start enumeration from 1 not 0
    print(f"\nDocument {i}: {doc.page_content}")