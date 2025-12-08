"""Database manager module for Text to SQL Agent that handles database operations."""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from langchain_community.utilities import SQLDatabase


class DatabaseManager:
    """Manager for database operations including schema creation, data insertion, and direct SQL queries."""

    def __init__(self, db_uri=None):
        """Initializes the database manager with an optional database URI or in-memory SQLite database."""
        if db_uri:
            self.engine = create_engine(db_uri)
        else:
            self.engine = create_engine("sqlite:///:memory:")

        self.db = SQLDatabase(engine=self.engine)

    def create_tables_from_schema(self, schema):
        """Creates tables from the provided database schema."""
        create_statements = schema.strip().split(';')
        with self.engine.connect() as conn:
            for statement in create_statements:
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()

    def insert_sample_data(self, data_dict):
        """Inserts sample data into the database."""
        with self.engine.connect() as conn:
            for table_name, rows in data_dict.items():
                if not rows or not isinstance(rows, list) or len(rows) == 0:
                    continue

                columns = list(rows[0].keys())

                for row in rows:
                    params = {col: row[col] for col in columns}

                    # Build the insert query with named parameters
                    placeholders = ', '.join([f":{col}" for col in columns])
                    column_str = ', '.join(columns)
                    insert_query = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"

                    # Execute with parameters as a dictionary
                    conn.execute(text(insert_query), params)

                conn.commit()

    def direct_sql_query(self, sql_query):
        """Executes a direct SQL query against the database and returns the result."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = result.keys()
                data = result.fetchall()
                return pd.DataFrame(data, columns=columns)
        except (SQLAlchemyError, ProgrammingError, OperationalError) as e:
            error_msg = f"Database error: {str(e)}"
            print(f"Error executing SQL query: {error_msg}")
            return None

    def get_schema_description(self):
        """Returns the database schema description."""
        return self.db.get_table_info()

    def get_langchain_db(self):
        """Returns the LangChain SQLDatabase wrapper for the database."""
        db = SQLDatabase(engine=self.engine)
        with self.engine.connect() as conn:
            tables_result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [table[0] for table in tables_result]
            print(f"Available tables in database: {', '.join(tables)}")

        return db
