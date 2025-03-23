import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    """Creates a custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Loads the LLM from Hugging Face."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def format_response(response):
    """Formats chatbot response with proper numbered points and unique sources."""
    import re

    result = response["result"]
    sources = response.get("source_documents", [])

    # Ensure numbered points start on a new line for better readability
    result = re.sub(r"(\d+)\.", r"\n\1.", result)

    # Store unique sources with associated page numbers
    unique_sources = {}
    for doc in sources:
        file_name = os.path.basename(doc.metadata.get("file_path", "Unknown Source"))
        page_number = doc.metadata.get("page", "Unknown Page")
        if file_name in unique_sources:
            unique_sources[file_name].add(page_number)
        else:
            unique_sources[file_name] = {page_number}

    # Format sources
    source_lines = [
        f"{file} (Pages {', '.join(map(str, sorted(pages)))})"
        for file, pages in unique_sources.items()
    ]
    source_text = "\n\n**Sources:** " + ", ".join(source_lines) if source_lines else ""

    return f"{result}{source_text}"

def main():
    """Main function for the Streamlit chatbot app."""
    st.markdown("""
        <h1 style='text-align: center;'>Ask Medical Chatbot!</h1>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            # Assistant on the LEFT
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(message['content'])
        else:
            # User on the RIGHT with emoji aligned properly
            st.markdown(
                f"<div style='display: flex; justify-content: flex-end; align-items: center;'>"
                f"<div style='margin-right: 10px;'>{message['content']}</div>"
                f"<div style='font-size: 24px;'>🧑</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        # Display user message on the RIGHT with emoji aligned properly
        st.markdown(
            f"<div style='display: flex; justify-content: flex-end; align-items: center;'>"
            f"<div style='margin-right: 10px;'>{prompt}</div>"
            f"<div style='font-size: 24px;'>🧑</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Only provide information from the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Pass the correct input format
            response = qa_chain.invoke({'query': prompt})

            # Format the response properly
            result_to_show = format_response(response)

            # Display assistant message on the LEFT
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
