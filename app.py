# import os
# import streamlit as st
# from langchain_community.graphs import Neo4jGraph
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.chains import GraphCypherQAChain

# # Load environment variables from .env file
# load_dotenv()

# # Set Neo4j credentials
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# # Set Groq API key
# groq_api_key = os.getenv("GROQ_API_KEY")

# # Initialize the Neo4j graph
# graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # Load the LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# # Create the Graph Cypher QA Chain
# chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# # Define the Streamlit app layout
# st.title("Movie Encyclopedia")

# # Input field for the user's query
# user_query = st.text_input("Enter your question about movies:")

# # Run the query when the user submits
# if st.button("Ask"):
#     if user_query:
#         with st.spinner("Processing..."):
#             response = chain.invoke({"query": user_query})
#             st.write(response)

# # Optionally, display the graph schema
# if st.checkbox("Show Graph Schema"):
#     graph.refresh_schema()
#     st.write(graph.schema)

import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain

# Load environment variables
load_dotenv()

# Set up Streamlit app with custom title and layout
st.set_page_config(
    page_title="Movie Encyclopedia",
    page_icon="🎬",
    layout="centered",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .container {
        background-image: url("https://cdn.pixabay.com/animation/2023/06/26/03/02/03-02-03-917_512.gif");
        background-size: cover;
        margin: 0;
        padding: 50px;
        border-radius: 5px;
        border: 1px solid #ddd;
        position: relative;
        overflow: hidden;
    }

    .container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 100%;
        background-color: #F1D4E5;
        transition: width 0.5s ease;
        z-index: 0;
    }

    .container:hover::before {
        width: 100%;
    }

    .container h4,
    .container p {
        position: relative;
        z-index: 1;
        color: #fff;
        transition: color 0.5s ease;
    }

    .container:hover h4,
    .container:hover p {
        color: #000;
    }
    </style>
    
    <div class="container">
        <h4>🎬 Movie Encyclopedia</h4>
        <p>Ask questions about movies using state-of-the-art language models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings and API key input
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if groq_api_key:
    # Initialize the Neo4j graph
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    # Load the LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Create the Graph Cypher QA Chain
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

    # Session ID input
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    # Input for user questions
    user_query = st.chat_input("Enter your question about movies:")

    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    if user_query:
        st.chat_message("user").markdown(user_query)
        with st.spinner("Processing..."):
            response = chain.invoke({"query": user_query})
            st.chat_message("assistant").markdown(response['result'])
            st.session_state.history.append({"user": user_query, "assistant": response['result']})

    # Display chat history in the right-hand sidebar
    with st.sidebar.expander("Chat History", expanded=False):
        for entry in st.session_state.history:
            st.write(f"**You:** {entry['user']}")
            st.write(f"**Assistant:** {entry['assistant']}")

    # Optionally, display the graph schema
    if st.checkbox("Show Graph Schema"):
        graph.refresh_schema()
        st.write(graph.schema)

else:
    st.warning("Please enter the Groq API Key")
