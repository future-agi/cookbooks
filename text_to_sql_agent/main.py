"""Main entry point for the Text to SQL Agent application."""

import random
from dotenv import load_dotenv
from .agent import TextToSQLAgent
from .schema import ECOMMERCE_SCHEMA, SAMPLE_DATA, EXAMPLE_QUERIES

load_dotenv()


def main():
    """Main entry point for the Text to SQL Agent application."""
    print("------------------------------------ Creating Text to SQL Agent... ------------------------------------")
    # Create an agent with default in-memory SQLite
    agent = TextToSQLAgent()

    print("------------------------------------ Setting up database schema... ------------------------------------")
    # Create tables from schema
    agent.create_tables_from_schema(ECOMMERCE_SCHEMA)

    print("------------------------------------ Loading sample data... ------------------------------------")
    # Insert sample data
    agent.insert_sample_data(SAMPLE_DATA)

    print("------------------------------------ Creating SQL agent with populated database... ------------------------------------")
    # Create SQL agent now that the database is populated
    agent.create_sql_agent()

    # Print the schema description
    print("\n------------------------------------ Database Schema: ------------------------------------")
    print(agent.get_schema_description())

    # Pick a random query from our examples or run all of them
    run_all = True

    if run_all:
        for i, query in enumerate(EXAMPLE_QUERIES):
            print(
                f"\n------------------------------------ Query {i+1}: {query} ------------------------------------"
            )
            result = agent.query(query)
            print(f"Generated SQL: {result['generated_sql']}")
            print(f"Result: {result['result']}")
            print(f"Execution Time: {result['latency']:.2f} seconds")
    else:
        # Pick a random query
        query = random.choice(EXAMPLE_QUERIES)
        print(f"\n------------------------------------ Query: {query} ------------------------------------")

        # Execute the query
        result = agent.query(query)

        # Print the results
        print(f"------------------------------------ Generated SQL: {result['generated_sql']} ------------------------------------")
        print(f"------------------------------------ Result: {result['result']} ------------------------------------")
        print(f"------------------------------------ Execution Time: {result['latency']:.2f} seconds ------------------------------------")


if __name__ == "__main__":
    main()
