import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AgriAssist AI",
    page_icon="🌱",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🌱 AI-Powered Smart Agriculture Assistance Platform")

st.write(
    "Your AI assistant for crops, diseases, fertilizers, "
    "irrigation, pesticides, and farming practices."
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("🌾 AgriAssist AI")

    st.write(
        "Ask agriculture-related questions and get answers "
        "from the agriculture knowledge base."
    )

    st.divider()

    st.subheader("💡 Example Questions")

    st.write("• What are the symptoms of rice blast?")
    st.write("• What are irrigation methods?")
    st.write("• What is PM-KISAN?")
    st.write("• What fertilizer is suitable for crops?")

    st.divider()

    st.info(
        "Important: Agricultural recommendations can vary "
        "depending on crop, location, soil, weather, and "
        "local regulations."
    )

# ==========================================================
# LOAD CHROMADB
# ==========================================================

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


vector_store = load_database()


# ==========================================================
# CHAT MEMORY
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================================
# DISPLAY PREVIOUS CHAT
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================================
# FARMER QUESTION
# ==========================================================

question = st.chat_input(
    "👨‍🌾 Ask your agriculture question..."
)


# ==========================================================
# PROCESS QUESTION
# ==========================================================

if question:

    # Add farmer question to memory
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display farmer question
    with st.chat_message("user"):

        st.markdown(question)


    # ------------------------------------------------------
    # Retrieve agriculture knowledge
    # ------------------------------------------------------

    with st.spinner("🔎 Searching agriculture knowledge..."):

        documents = vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )


    # ------------------------------------------------------
    # Create conversation history
    # ------------------------------------------------------

    conversation_history = ""

    for message in st.session_state.messages[-6:]:

        conversation_history += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # ------------------------------------------------------
    # Create AI prompt
    # ------------------------------------------------------

    prompt = f"""
You are AgriAssist AI, an agriculture assistance chatbot.

You help farmers understand agriculture-related topics
using the provided knowledge base.

AGRICULTURE KNOWLEDGE:
{context}

RECENT CONVERSATION:
{conversation_history}

CURRENT FARMER QUESTION:
{question}

IMPORTANT RULES:

1. Use the agriculture knowledge as your main source.
2. Use the conversation history to understand follow-up questions.
3. Do not invent agricultural facts.
4. Do not invent pesticide prices.
5. Do not invent fertilizer prices.
6. Do not invent supplier names or locations.
7. Do not invent government scheme benefits.
8. If the requested information is not available, clearly say:
   "I don't have enough information in my current knowledge base."
9. For pesticides and chemicals, advise farmers to follow product
   labels and local agricultural recommendations.
10. Keep the answer simple and easy for farmers to understand.

Give the best answer possible using the provided knowledge.
"""


    # ------------------------------------------------------
    # Generate AI answer
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 Preparing your answer..."):

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

        st.markdown(answer)


    # ------------------------------------------------------
    # Save AI answer to memory
    # ------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # ------------------------------------------------------
    # Sources
    # ------------------------------------------------------

    with st.expander("📚 Knowledge Sources"):

        sources = set()

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown source"
            )

            sources.add(source)

        for source in sources:

            st.write("📄", source)  