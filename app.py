# app.py

import streamlit as st
import google.generativeai as genai
import duckdb
import pandas as pd
import os

st.title("My App is Running!") # <--- ADD THIS LINE

# --- 1. APP CONFIGURATION & SETUP ---

# Set up the Streamlit page with a title and icon
st.set_page_config(
    page_title="DuckDB Chat",
    page_icon="🦆",
    layout="wide"
)

# --- CUSTOM STYLING (CSS) ---
# This is a new section to customize the look of the app.
# You can change colors and other styles here.
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Chat message containers */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* User's chat message */
    div[data-testid="chat-bubble-container"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: #e1f5fe;
    }

    /* Assistant's chat message */
    div[data-testid="chat-bubble-container"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
            
    /* Main title style */
    h1 {
        color: #1e3a8a; /* A darker blue for the title */
    }
</style>
""", unsafe_allow_html=True)


st.title("💬 Chat with your DuckDB Database")
st.caption("Powered by Google Gemini & Streamlit")

# --- PASTE YOUR API KEY HERE ---
# For local testing, it's easiest to paste your key directly.
# Replace "YOUR_API_KEY_HERE" with your actual Gemini API key.

api_key = "AIzaSyAU191le7IAh70uGsycPutuxjCZfk3l4AA"  # <--- PASTE YOUR KEY IN THE QUOTES

# ---------------------------------

try:
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        st.warning("Please add your Gemini API key to the `app.py` file.")
        api_key = None  # Set to None to indicate it's missing
    else:
        genai.configure(api_key=api_key)

except Exception as e:
    st.error(f"Error configuring the Gemini API: {e}")
    st.stop()


# Connect to DuckDB using Streamlit's resource caching for efficiency
@st.cache_resource
def get_db_connection():
    """Establishes a connection to the DuckDB database."""
    try:
        # Replace 'move_key_df' with the actual name of your database file
        con = duckdb.connect(database='move_key_df.duckdb', read_only=True)
        return con
    except duckdb.IOException as e:
        st.error(f"❌ Error connecting to database file 'move_key_df.duckdb': {e}")
        st.info("Please ensure the database file is in the same directory as this app and no other program is using it.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during database connection: {e}")
        return None

con = get_db_connection()


# Your SQL execution function, adapted for Streamlit feedback
def execute_sql_query(query: str) -> str:
    """Executes a SQL query on the database and returns results as markdown."""
    if con is None:
        return "Database connection is not available."
    try:
        # Use st.spinner to show a loading message while the query runs
        with st.spinner(f"Running query: `{query}`..."):
            result_df = con.execute(query).df()
        if result_df.empty:
            return "Query returned no results."
        # Convert the pandas DataFrame to a markdown string for display
        return result_df.to_markdown(index=False)
    except Exception as e:
        return f"An error occurred while executing the query: {e}"

# --- NEW FUNCTION TO LOAD SYSTEM INSTRUCTION ---
def load_system_instruction(filepath="system_prompt.txt"):
    """Loads the system instruction from a text file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"Warning: '{filepath}' not found. Using a default system instruction.")
        # Return a default instruction if the file is missing
        return """You are an expert SQL assistant. You are working with a DuckDB database that contains a table named 'main_data'. Your task is to receive questions from a user, convert them into accurate SQL queries, and use the provided `execute_sql_query` tool to get the answer. Always use the exact column names from the data dictionary. Do not guess column names."""

# --- 2. GEMINI MODEL AND TOOL DEFINITION ---

# Initialize model variable to None
model = None

# Only define the model if the API key is available
if api_key:
    # Load the system instruction from the new text file
    system_instruction_from_file = load_system_instruction()

    # Define the function schema for the Gemini tool
    execute_sql_function = genai.protos.FunctionDeclaration(
        name="execute_sql_query",
        description="Executes a SQL query on the 'main_data' table in the DuckDB database. Use this for any questions that require retrieving, aggregating, or analyzing data from the database.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "query": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="The complete and valid SQL query to be executed on the 'main_data' table."
                )
            },
            required=["query"]
        )
    )

    # Create the tool that Gemini can use
    sql_tool = genai.protos.Tool(function_declarations=[execute_sql_function])

    # Initialize the generative model using the instruction from the file
    model = genai.GenerativeModel(
        'gemini-1.5-pro-latest',
        tools=[sql_tool],
        system_instruction=system_instruction_from_file
    )


# --- 3. STREAMLIT CHAT INTERFACE LOGIC ---

# Add a defensive check to ensure the model is initialized before starting the chat
if not model:
    st.info("Please add your Gemini API key to continue.")
    st.stop()


# Initialize chat history in session state if it doesn't exist
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Display past messages from the chat history
for message in st.session_state.chat.history:
    role = "assistant" if message.role == "model" else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# Main chat input and response loop
if prompt := st.chat_input("Ask a question about your data..."):
    # First, check if the database connection is valid
    if con is None:
        st.error("Cannot proceed: Database connection is not established.")
    else:
        # Display the user's message in the chat
        st.chat_message("user").markdown(prompt)

        try:
            # Send the user's prompt to the Gemini model
            response = st.session_state.chat.send_message(prompt)

            # Check if the model decided to call the SQL tool
            if response.parts and hasattr(response.parts[0], 'function_call'):
                function_call = response.parts[0].function_call
                function_name = function_call.name
                arguments = dict(function_call.args)

                if function_name == "execute_sql_query":
                    # Execute the SQL query returned by the model
                    query_result = execute_sql_query(arguments['query'])

                    # Send the query result back to the model for analysis
                    final_response = st.session_state.chat.send_message(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_name,
                                response={"result": query_result}
                            )
                        )
                    )
                    # Display the model's final, user-friendly answer
                    st.chat_message("assistant").markdown(final_response.text)
            else:
                # If no tool was called, just display the model's text response
                st.chat_message("assistant").markdown(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")