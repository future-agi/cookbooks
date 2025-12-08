# Text to SQL Agent

A powerful conversational agent that translates natural language questions into SQL queries for e-commerce data analysis.

## Overview

The Text to SQL Agent is an AI-powered tool that allows you to interact with e-commerce database schemas using natural language. Instead of writing complex SQL queries, users can ask questions in plain English, and the agent will:

1. Translate the natural language to SQL
2. Execute the query against the database
3. Return the results in a readable format
4. Provide explanations of the SQL when requested

## Features

- **Natural Language Interface**: Query your database using plain English
- **Multiple Prompt Variations**: Choose from different prompt strategies for various use cases:
  - `basic`: Simple, concise prompting
  - `advanced`: More detailed prompting with additional context
  - `verbose`: Extensive reasoning and step-by-step query generation
  - `analytical`: Business-focused prompting for analytical queries
  - `explanation`: Designed to explain SQL queries in plain English
  - `optimized`: AI-optimized prompt for better SQL generation
- **Interactive Mode**: Continuously query the database in a terminal interface
- **Multiple Model Support**: Works with various OpenAI models (default: gpt-4o-mini)
- **Instrumentation**: Built-in tracing and evaluation using Future Insights instrumentation
- **Database Schema**: Pre-configured e-commerce database schema with sample data
- **SQL Explanation**: Capability to explain the generated SQL in plain English

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd text_to_sql_agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   # Create a .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Basic Usage

```bash
python main.py
```
This will start the interactive mode where you can type questions about the e-commerce database.

### Command Line Options

```bash
python main.py --prompt [basic|advanced|verbose|analytical|explanation|optimized] --model MODEL_NAME --question "Your question here"
```

#### Arguments

- `--prompt`: Choose the prompt variation (default: basic)
  - Options: basic, advanced, verbose, analytical, explanation, optimized
- `--model`: Specify the OpenAI model to use (default: gpt-4o-mini)
- `--question`: Specify a direct question to query the database
- `--run-all`: Run all example queries with all prompt variations
- `--interactive`: Run in interactive mode (this is the default if no question is provided)
- `--mode`: Choose between "experiment" (with version tracking and evaluation) or "observe" mode (default)

### Interactive Mode Commands

While in interactive mode, you can use the following special commands:

- `exit`, `quit`, or `q`: Exit the interactive mode
- `prompt:NAME`: Change the prompt variation (e.g., `prompt:advanced`)
- `explain:SQL`: Get an explanation of an SQL query (e.g., `explain:SELECT * FROM customers`)
- `example`: Run a random example query from the predefined set

### Example Questions

```
How many customers do we have?
What are the top 5 best-selling products?
Which customer spent the most money?
What is the average order value?
How many orders were placed in the last month?
Which products have less than 10 items in stock?
```

## Project Structure

```
text_to_sql_agent/
├── agent.py               # Core Text to SQL Agent implementation
├── database.py            # Database connection and utilities
├── main.py                # Main entry point and CLI interface
├── prompts.py             # Prompt templates for different variations
├── requirements.txt       # Python dependencies
├── schema.py              # E-commerce database schema and sample data
├── optimized_prompt.py    # AI-optimized prompt template
├── README.md              # This documentation file
```

## Evaluation

The agent includes built-in evaluation tags for tracking performance metrics:
- Completeness
- Groundedness
- Hallucination detection
- Tool calling accuracy
- Text-to-SQL quality
- Business context quality

## Development

### Adding New Prompt Variations

To add a new prompt variation:

1. Create a new template in `prompts.py`
2. Add the variation name to the `PROMPT_VARIATIONS` dictionary
3. Update the argument parser in `main.py` to include the new variation

### Extending the Schema

To extend or modify the database schema:

1. Update the schema definitions in `schema.py`
2. Add sample data for new tables if needed
3. Update any references in prompt templates if necessary

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
