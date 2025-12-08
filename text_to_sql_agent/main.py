"""Main entry point for the Text to SQL Agent application."""

import random
import argparse
from dotenv import load_dotenv
from agent import TextToSQLAgent
from schema import ECOMMERCE_SCHEMA, SAMPLE_DATA, EXAMPLE_QUERIES
# The optimized_prompt is imported in prompts.py

load_dotenv()

from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    ProjectType,
    EvalName,
    EvalTag,
    EvalTagType,
    EvalSpanKind,
    ModelChoices
)
from traceai_langchain import LangChainInstrumentor

eval_tags = [
    EvalTag(type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.AGENT,
            eval_name=EvalName.COMPLETENESS,
            config={},
            mapping={
                "input": "raw.input",
                "output": "raw.output"
            },
            custom_eval_name="Completeness",
            model=ModelChoices.TURING_LARGE),
    EvalTag(type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.AGENT,
            eval_name=EvalName.GROUNDEDNESS,
            config={},
            mapping={
                "input": "raw.input",
                "output": "raw.output"
            },
            custom_eval_name="Groundedness",
            model=ModelChoices.TURING_LARGE),
    EvalTag(type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.AGENT,
            eval_name=EvalName.DETECT_HALLUCINATION,
            config={},
            mapping={
                "input": "raw.input",
                "output": "raw.output"
            },
            custom_eval_name="Hallucination",
            model=ModelChoices.TURING_LARGE),
    EvalTag(type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.TOOL,
            eval_name=EvalName.EVALUATE_FUNCTION_CALLING,
            config={},
            mapping={
                "input": "raw.input",
                "output": "tool.name"
            },
            custom_eval_name="Tool_Calling",
            model=ModelChoices.TURING_LARGE),
    EvalTag(type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.AGENT,
            eval_name=EvalName.TEXT_TO_SQL,
            config={},
            mapping={
                "input": "raw.input",
                "output": "raw.output"
            },
            custom_eval_name="Text_To_SQL",
            model=ModelChoices.TURING_LARGE),
        EvalTag(type=EvalTagType.OBSERVATION_SPAN,
                value=EvalSpanKind.AGENT,
                eval_name='business_context_quality',
                config={},
                mapping={
                    "query": "raw.input",
                    "response": "raw.output"
                },
                custom_eval_name="Business_Context_Quality",
                model=ModelChoices.TURING_LARGE)
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text to SQL Agent with customizable prompts")
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["basic", "advanced", "verbose", "analytical", "explanation", "optimized"],
        default="basic",
        help="Which prompt variation to use: basic, advanced, verbose, analytical, explanation, or optimized"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all example queries instead of just one random query"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Custom question to query the database"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive mode where you can ask multiple questions"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["experiment", "observe"],
        default="observe",
        help="Mode to run the agent in: experiment (with version tracking) or observe (default)"
    )
    return parser.parse_args()

# We'll register the trace provider inside main() after arguments are parsed



def main():
    """Main entry point for the Text to SQL Agent application."""
    args = parse_args()

    # Map deprecated 'default' value to 'basic' for backward compatibility
    prompt_variation = args.prompt
    if prompt_variation == "default":
        prompt_variation = "basic"
        
    # Create a version name based on prompt type and model
    version_name = f"{prompt_variation}_{args.model.replace('-', '_')}"
    
    # Register trace provider based on the selected mode
    if args.mode == "experiment":
        # For experiment mode, include version name and evaluation tags
        trace_provider = register(
            project_type=ProjectType.EXPERIMENT,
            project_name="Text-to-SQL",
            project_version_name=version_name,
            eval_tags=eval_tags
        )
    else:  # observe mode (default)
        trace_provider = register(
            project_type=ProjectType.OBSERVE,
            project_name="text_to_sql_agent_obs"
        )
    
    # Instrument LangChain/LangGraph
    LangChainInstrumentor().instrument(tracer_provider=trace_provider)

    print(f"------------------------------------ Creating Text to SQL Agent with {prompt_variation} prompt... ------------------------------------")
    print(f"------------------------------------ Using version: {version_name} ------------------------------------")
    print(f"------------------------------------ Running in {args.mode} mode ------------------------------------")
    # Create an agent with specified prompt variation and model
    agent = TextToSQLAgent(complexity=prompt_variation, model_name=args.model)

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

    # Print available prompt variations
    print("\n------------------------------------ Available Prompt Variations ------------------------------------")
    print(f"Available prompt variations: {agent.get_available_prompt_variations()}")
    print(f"Current prompt variation: {args.prompt}")
    
    # Highlight the new optimized prompt option if available
    if "optimized" in agent.get_available_prompt_variations():
        print("✨ NEW! Try the AI-optimized prompt with --prompt optimized for improved SQL generation!")

    # Determine which query to run based on command line arguments
    if args.interactive:
        # Interactive mode - keep terminal open for multiple questions
        print("\n------------------------------------ INTERACTIVE MODE ------------------------------------")
        print("Type your questions about the e-commerce database. Type 'exit', 'quit', or 'q' to exit.")
        print("Type 'prompt:NAME' to change prompt variation (basic, advanced, verbose, analytical).")
        print("Type 'explain:SQL' to get an explanation of an SQL query.")
        
        while True:
            # Reset to the original prompt variation after explanations
            agent.set_prompt_variation(prompt_variation)
            
            # Get user question
            user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
            
            # Check for exit command
            if user_input.lower() in ('exit', 'quit', 'q'):
                print("Exiting interactive mode...")
                break
                
            # Check for prompt change command
            if user_input.lower().startswith('prompt:'):
                new_variation = user_input.split(':', 1)[1].strip()
                if new_variation in agent.get_available_prompt_variations():
                    prompt_variation = new_variation
                    agent.set_prompt_variation(prompt_variation)
                    print(f"Prompt variation changed to: {prompt_variation}")
                else:
                    print(f"Invalid prompt variation. Available options: {agent.get_available_prompt_variations()}")
                continue
                
            # Check for explain command
            if user_input.lower().startswith('explain:'):
                sql_to_explain = user_input.split(':', 1)[1].strip()
                print("\n------------------------------------ SQL Explanation ------------------------------------")
                agent.set_prompt_variation("explanation")
                explanation_result = agent.query(f"Explain this SQL: {sql_to_explain}")
                print(explanation_result['result'])
                continue
            
            # Process normal question
            if user_input:
                print(f"\n------------------------------------ User Question: {user_input} ------------------------------------")
                
                # Execute the query
                result = agent.query(user_input)
                
                # Print the results
                print(f"------------------------------------ Generated SQL ({result['prompt_variation']} prompt): {result['generated_sql']} ------------------------------------")
                print(f"------------------------------------ Result: {result['result']} ------------------------------------")
                print(f"------------------------------------ Execution Time: {result['latency']:.2f} seconds ------------------------------------")
    
    elif args.question:
        # Use the user-provided question
        query = args.question
        print(f"\n------------------------------------ User Question: {query} ------------------------------------")
        
        # Execute the query with the specified prompt variation
        result = agent.query(query)
        
        # Print the results
        print(f"------------------------------------ Generated SQL ({result['prompt_variation']} prompt): {result['generated_sql']} ------------------------------------")
        print(f"------------------------------------ Result: {result['result']} ------------------------------------")
        print(f"------------------------------------ Execution Time: {result['latency']:.2f} seconds ------------------------------------")
        
        # Optionally, demonstrate how to explain the generated SQL
        if result['generated_sql'] and result['generated_sql'] != "Error: Could not extract SQL query":
            print("\n------------------------------------ SQL Explanation ------------------------------------")
            agent.set_prompt_variation("explanation")
            explanation_result = agent.query(f"Explain this SQL: {result['generated_sql']}")
            print(explanation_result['result'])
            
    # Use the run-all argument to decide whether to run all example queries or just one
    elif args.run_all:
        # Run each query with each prompt variation
        for i, query in enumerate(EXAMPLE_QUERIES):
            print(
                f"\n------------------------------------ Query {i+1}: {query} ------------------------------------"
            )

            for variation in agent.get_available_prompt_variations():
                if variation != "explanation":  # Skip explanation prompt for generation
                    # Set the prompt variation for this query
                    agent.set_prompt_variation(variation)

                    # Execute the query with this prompt variation
                    result = agent.query(query)

                    print(f"--- {variation.upper()} PROMPT ---")
                    print(f"Generated SQL: {result['generated_sql']}")
                    print(f"Result: {result['result']}")
                    print(f"Execution Time: {result['latency']:.2f} seconds\n")
    else:
        # Interactive mode by default
        print("\n------------------------------------ INTERACTIVE MODE ------------------------------------")
        print("Type your questions about the e-commerce database. Type 'exit', 'quit', or 'q' to exit.")
        print("Type 'prompt:NAME' to change prompt variation (basic, advanced, verbose, analytical).")
        print("Type 'explain:SQL' to get an explanation of an SQL query.")
        print("Type 'example' to run a random example query.")
        
        while True:
            # Reset to the original prompt variation after explanations
            agent.set_prompt_variation(prompt_variation)
            
            # Get user question
            user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
            
            # Check for exit command
            if user_input.lower() in ('exit', 'quit', 'q'):
                print("Exiting interactive mode...")
                break
                
            # Check for example command
            if user_input.lower() == 'example':
                query = random.choice(EXAMPLE_QUERIES)
                print(f"\n------------------------------------ Example Query: {query} ------------------------------------")
                result = agent.query(query)
                print(f"------------------------------------ Generated SQL ({result['prompt_variation']} prompt): {result['generated_sql']} ------------------------------------")
                print(f"------------------------------------ Result: {result['result']} ------------------------------------")
                print(f"------------------------------------ Execution Time: {result['latency']:.2f} seconds ------------------------------------")
                continue
                
            # Check for prompt change command
            if user_input.lower().startswith('prompt:'):
                new_variation = user_input.split(':', 1)[1].strip()
                if new_variation in agent.get_available_prompt_variations():
                    prompt_variation = new_variation
                    agent.set_prompt_variation(prompt_variation)
                    print(f"Prompt variation changed to: {prompt_variation}")
                else:
                    print(f"Invalid prompt variation. Available options: {agent.get_available_prompt_variations()}")
                continue
                
            # Check for explain command
            if user_input.lower().startswith('explain:'):
                sql_to_explain = user_input.split(':', 1)[1].strip()
                print("\n------------------------------------ SQL Explanation ------------------------------------")
                agent.set_prompt_variation("explanation")
                explanation_result = agent.query(f"Explain this SQL: {sql_to_explain}")
                print(explanation_result['result'])
                continue
            
            # Process normal question
            if user_input:
                print(f"\n------------------------------------ User Question: {user_input} ------------------------------------")
                
                # Execute the query
                result = agent.query(user_input)
                
                # Print the results
                print(f"------------------------------------ Generated SQL ({result['prompt_variation']} prompt): {result['generated_sql']} ------------------------------------")
                print(f"------------------------------------ Result: {result['result']} ------------------------------------")
                print(f"------------------------------------ Execution Time: {result['latency']:.2f} seconds ------------------------------------")


if __name__ == "__main__":
    main()
