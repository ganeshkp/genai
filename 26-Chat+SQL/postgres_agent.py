from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
import os


# Environment variables or Django settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PG_URI = os.getenv(
    "POSTGRES_URI"
)  # like: postgresql+psycopg2://user:pass@localhost:5432/dbname


llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Set up database
engine = create_engine(PG_URI)
db = SQLDatabase(engine)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)


def run_query(query: str) -> str:
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    query = "what is lowest price product?"
    response = run_query(query)
    print(response)
