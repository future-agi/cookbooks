"""Module for the Text to SQL Agent that converts natural language to SQL queries."""

import time
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

from .database import DatabaseManager
from .prompts import get_text2sql_prompt


class TextToSQLAgent:
    """Agent that converts natural language queries to SQL and executes them against a database."""

    def __init__(self,
                 db_uri=None,
                 model_name="gpt-4o-mini",
                 complexity="default"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.db_manager = DatabaseManager(db_uri)
        self.prompt = get_text2sql_prompt(complexity)
        self.agent_executor = None

    def create_sql_agent(self):
        """Creates an SQL agent that can execute SQL queries against the database."""
        db = self.db_manager.get_langchain_db()
        schema_info = db.get_table_info()
        print(f"\nDatabase schema info:\n{schema_info}\n")

        self.agent_executor = create_sql_agent(llm=self.llm,
                                               db=db,
                                               agent_type="tool-calling",
                                               verbose=True)

        return self.agent_executor

    def query(self, question):
        """Executes a natural language query against the database and returns the result."""
        start_time = time.time()
        try:
            if self.agent_executor is None:
                self.create_sql_agent()
            agent_result = self.agent_executor.invoke({"input": question})

            end_time = time.time()
            latency = end_time - start_time

            sql_query = self._extract_sql_from_steps(
                agent_result.get("intermediate_steps", []))

            return {
                "question": question,
                "generated_sql": sql_query,
                "execution_success": True,
                "result": agent_result["output"],
                "error": "",
                "latency": latency
            }

        except (ValueError, KeyError) as e:
            # Handle specific data/key errors
            error_msg = f"Data error: {str(e)}"
            return {
                "question": question,
                "generated_sql": "Error: Could not extract SQL query",
                "execution_success": False,
                "result": "",
                "error": error_msg,
                "latency": time.time() - start_time
            }
        except AttributeError as e:
            error_msg = f"Missing attribute error: {str(e)}"
            return {
                "question": question,
                "generated_sql": "Error: Could not extract SQL query",
                "execution_success": False,
                "result": "",
                "error": error_msg,
                "latency": time.time() - start_time
            }

    def _extract_sql_from_steps(self, steps):
        """Extracts the SQL query from the agent's intermediate steps."""
        for step in steps:
            if isinstance(step[0].tool_input, str) and any(
                    keyword in step[0].tool_input.upper()
                    for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                return step[0].tool_input
        return ""

    def create_tables_from_schema(self, schema):
        """Creates tables from the database schema."""
        self.db_manager.create_tables_from_schema(schema)

    def insert_sample_data(self, data_dict):
        """Inserts sample data into the database."""
        self.db_manager.insert_sample_data(data_dict)

    def direct_sql_query(self, sql_query):
        """Executes a direct SQL query against the database and returns the result."""
        return self.db_manager.direct_sql_query(sql_query)

    def get_schema_description(self):
        """Returns the database schema description."""
        return self.db_manager.get_schema_description()
