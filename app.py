import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain

# Load environment variables from .env file
load_dotenv()

# Set Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Load the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Create the Graph Cypher QA Chain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# Define the Streamlit app layout
st.title("Chat-Mate...Filmography version")

# Input field for the user's query
user_query = st.text_input("Enter your question about movies:")

# Run the query when the user submits
if st.button("Ask"):
    if user_query:
        with st.spinner("Processing..."):
            response = chain.invoke({"query": user_query})
            st.write(response)

# Optionally, display the graph schema
if st.checkbox("Show Graph Schema"):
    graph.refresh_schema()
    st.write(graph.schema)
