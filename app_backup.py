import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriAssist AI",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌱 AI-Powered Smart Agriculture Assistance Platform")

st.write(
    "Ask questions about crops, diseases, fertilizers, "
    "irrigation, pesticides, and farming practices."
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_database():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="agriculture_knowledge_v2",
        embedding_function=embeddings,
        persist_directory="database/chroma_db_v2"
    )

    return vector_store


# --------------------------------------------------
# LOAD DATABASE
# --------------------------------------------------

vector_store = load_database()


# --------------------------------------------------
# FARMER QUESTION
# --------------------------------------------------

question = st.text_input(
    "👨‍🌾 Ask your agriculture question:",
    placeholder="Example: What fertilizer is suitable for rice?"
)


# --------------------------------------------------
# ASK AI BUTTON
# --------------------------------------------------

if st.button("🤖 Ask AI"):

    if question.strip() == "":
        st.warning("Please enter a question first.")

    else:

        with st.spinner("🌱 Searching agriculture knowledge..."):

            # Retrieve relevant documents
            documents = vector_store.similarity_search(
                question,
                k=3
            )

            # Create context
            context = "\n\n".join(
                document.page_content
                for document in documents
            )

        with st.spinner("🤖 AI is preparing your answer..."):

            prompt = f"""
You are an AI-powered agriculture assistance assistant.

Help farmers by providing clear and useful answers.

Use the agriculture knowledge below to answer the
farmer's question.

IMPORTANT:
- Use the provided knowledge as your main source.
- Do not invent facts.
- Do not invent pesticide prices, fertilizer prices,
  supplier names, or government benefits.
- If the information is not available, say that it
  is not available in the current knowledge base.
- For pesticide or chemical recommendations, advise
  farmers to follow product labels and local
  agricultural recommendations.
- Keep the answer simple and easy to understand.

AGRICULTURE KNOWLEDGE:
{context}

FARMER QUESTION:
{question}

ANSWER:
"""

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

        # Display answer
        st.subheader("🤖 AI Agriculture Assistant")

        st.write(answer)


        # Display sources
        with st.expander("📚 Knowledge Sources"):

            for document in documents:

                st.write(
                    document.metadata.get(
                        "source",
                        "Unknown source"
                    )
                )