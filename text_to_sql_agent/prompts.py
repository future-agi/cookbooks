"""Prompts for the Text to SQL Agent."""

from langchain_core.prompts import PromptTemplate

DEFAULT_TEXT2SQL_TEMPLATE = """You are an expert SQL query generator.
Given the following database schema:

{schema}

Generate a SQL query to answer the following question:
{question}

Make sure your query is correct and efficient. Return only the SQL query without any explanations.
"""

text2sql_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=DEFAULT_TEXT2SQL_TEMPLATE
)

ADVANCED_TEXT2SQL_TEMPLATE = """You are an expert SQL query generator 
for an e-commerce database with deep knowledge of database optimization.
Given the following database schema:

{schema}

Generate a SQL query to answer the following question:
{question}

Requirements:
1. Ensure optimal query performance with proper joins and indexes
2. Handle edge cases and NULL values appropriately
3. Use appropriate aggregate functions and GROUP BY clauses when needed
4. Follow SQL best practices for readability and maintainability

Return only the SQL query without any explanations.
"""

advanced_text2sql_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=ADVANCED_TEXT2SQL_TEMPLATE
)

SQL_EXPLANATION_TEMPLATE = """Given the following SQL query:

{sql_query}

Explain in simple terms what this query does, step by step. Include:
1. What tables are being queried
2. What conditions are being applied
3. How the data is being filtered, grouped, or sorted
4. What the expected results would be
"""

sql_explanation_prompt = PromptTemplate(
    input_variables=["sql_query"],
    template=SQL_EXPLANATION_TEMPLATE
)


def get_text2sql_prompt(complexity="default"):
    """Returns the appropriate text2sql prompt based on the complexity level."""
    if complexity.lower() == "advanced":
        return advanced_text2sql_prompt
    return text2sql_prompt
