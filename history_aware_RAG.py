from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

# --------------------
# LLM
# --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# --------------------
# Embeddings + Chroma
# --------------------
persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------
# Reformulation Prompt
# --------------------
reformulation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Given the chat history and latest user question,
        rewrite the question so it can be understood
        without previous conversation.

        Return ONLY the rewritten question.
        """
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# --------------------
# Answer Prompt
# --------------------
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant.

        Use ONLY the provided context to answer.

        If the answer is not present in the context,
        say that you do not have enough information.

        Context:
        {context}
        """
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# --------------------
# Chat History
# --------------------
chat_history = []

# --------------------
# Chat Loop
# --------------------
while True:

    query = input("\nENTER MSG HERE: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Step 1: Reformulate question using history
    reformulated_prompt = reformulation_prompt.invoke({
        "chat_history": chat_history,
        "question": query
    })

    standalone_question = llm.invoke(
        reformulated_prompt
    ).content

    print(f"\nReformulated Query: {standalone_question}")

    # Step 2: Retrieve documents
    relevant_docs = retriever.invoke(
        standalone_question
    )

    # Step 3: Build context
    context = ""

    for doc in relevant_docs:
        context += doc.page_content
        context += "\n\n"

    # Step 4: Build answer prompt
    formatted_prompt = answer_prompt.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history
    })

    # Step 5: Generate answer
    response = llm.invoke(formatted_prompt)

    print("\n---ANSWER---")
    print(response.content)

    # Step 6: Save conversation
    chat_history.append(
        HumanMessage(content=query)
    )

    chat_history.append(
        AIMessage(content=response.content)
    )