import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Fetch API token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN is missing. Ensure it's set in the .env file.")

# Hugging Face model repo
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """Loads the LLM from HuggingFace."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Creates a custom prompt for the LLM."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}. Ensure vectorstore is created.")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

def main():
    """Runs the chatbot query system."""
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})  # Fixed key

    print("\nRESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:\n", response["source_documents"])

if __name__ == "__main__":
    main()

