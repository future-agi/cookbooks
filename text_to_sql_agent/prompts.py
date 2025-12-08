"""Prompts for the Text to SQL Agent."""

from langchain_core.prompts import PromptTemplate
import optimized_prompt


# Prompt variations for text-to-SQL generation
prompt_templates = {
    "basic": """You are an expert SQL query generator.
Given the following database schema:

{schema}

Generate a SQL query to answer the following question:
{question}

Make sure your query is correct and efficient. Return only the SQL query without any explanations.
""",

    "advanced": """You are an expert SQL query generator 
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
""",

    "verbose": """You are an advanced SQL query generator with expertise in e-commerce data modeling.
Given the following database schema:

{schema}

Generate a comprehensive SQL query to answer the following question:
{question}

Your query should:
1. Include comments explaining your approach
2. Prioritize readability over brevity
3. Use appropriate table aliases for clarity
4. Handle potential edge cases and NULL values
5. Use appropriate JOINs and subqueries when beneficial

Return only the SQL query with inline comments.
""",

    "analytical": """You are a data analyst specializing in e-commerce analytics.
Given the following database schema:

{schema}

Generate an analytical SQL query to answer the following question:
{question}

Your query should:
1. Use advanced analytical functions when appropriate (window functions, CTEs, etc.)
2. Optimize for performance with large datasets
3. Follow dimensional modeling best practices
4. Include appropriate aggregations and groupings
5. Handle seasonality and time-based analysis when relevant

Return only the SQL query without any explanations.
""",

    "explanation": """Given the following SQL query:

{sql_query}

Explain in simple terms what this query does, step by step. Include:
1. What tables are being queried
2. What conditions are being applied
3. How the data is being filtered, grouped, or sorted
4. What the expected results would be
""",

    # Add the optimized prompt from our optimization process
    "optimized": optimized_prompt.optimized_prompt
}

# Create PromptTemplates for each variation
text2sql_prompts = {}
for name, template in prompt_templates.items():
    if name == "explanation":
        text2sql_prompts[name] = PromptTemplate(
            input_variables=["sql_query"],
            template=template
        )
    else:
        text2sql_prompts[name] = PromptTemplate(
            input_variables=["schema", "question"],
            template=template
        )

# For backward compatibility
text2sql_prompt = text2sql_prompts["basic"]
advanced_text2sql_prompt = text2sql_prompts["advanced"]
sql_explanation_prompt = text2sql_prompts["explanation"]


def get_text2sql_prompt(variation="basic"):
    """
    Returns the appropriate text2sql prompt based on the variation name.
    
    Args:
        variation (str): The prompt variation to use. Options include:
            - "basic": Simple SQL query generation
            - "advanced": Optimized queries with best practices
            - "verbose": Queries with explanatory comments
            - "analytical": Queries using advanced analytical functions
            - "explanation": For explaining existing SQL queries
    
    Returns:
        PromptTemplate: The requested prompt template
    """
    variation = variation.lower()
    if variation in text2sql_prompts:
        return text2sql_prompts[variation]
    else:
        # Default to basic if variation not found
        print(f"Variation '{variation}' not found. Using 'basic' variation instead.")
        return text2sql_prompts["basic"]
