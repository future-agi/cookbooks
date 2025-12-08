# @title
# CELL 2 — Imports and logging configuration
import logging
import os
import pandas as pd
from typing import List, Dict, Any

# --- Framework Imports ---
from fi.opt.datamappers import BasicDataMapper
from fi.opt.base.evaluator import Evaluator
from fi.opt.utils import setup_logging
from fi.opt.optimizers import GEPAOptimizer

# --- Evaluator Imports ---
from fi.evals.metrics import CustomLLMJudge
from fi.evals.llm import LiteLLMProvider

# Configure logging
setup_logging(level=logging.INFO, log_to_console=True, log_to_file=True, log_file="agent-opt.log")
logger = logging.getLogger(__name__)

print("✅ All components imported and logging is configured.")


# @title Load dataset from CSV
def load_dataset_from_csv() -> List[Dict[str, Any]]:
    '''Loads dataset from the CSV file.'''
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    df = pd.read_csv(csv_path)
    
    # SQL datasets typically have table, question, sql columns
    print("Columns: ", df.columns)
    print("Head: ", df.head(2))
    
    return df.to_dict("records")

dataset = load_dataset_from_csv()
print("✅ Dataset loaded from CSV successfully. Here are the first two examples:")
for item in dataset[:2]:
    print(item)



# LLM provider used by the judge
provider = LiteLLMProvider()

correctness_judge_config = {
    "name": "sql_correctness_judge",
    "grading_criteria": '''You are evaluating an AI's SQL query against a reference SQL query. The score must be 1.0 if the 'response'
is functionally equivalent to the 'expected_response' (the ground truth) - meaning both queries would return the same results.
The score should be 0.0 if they would return different results or if there are syntax errors.
Partial credit is acceptable for queries that are close but have minor differences that would slightly affect results.
Focus on correctness of joins, conditions, aggregations, and overall query logic, not stylistic differences.''',
}

# Instantiate the judge and evaluator wrapper
correctness_judge = CustomLLMJudge(provider, config=correctness_judge_config)
evaluator = Evaluator(metric=correctness_judge)

# Data mapper connects model outputs to the judge expectations
data_mapper = BasicDataMapper(
    key_map={
        "response": "generated_output",
        "expected_response": "sql"  # Using SQL queries as the expected response
    }
)

print("✅ Evaluation strategy defined using a Custom LLM-as-a-Judge.")



# @title Prompt Setup
INITIAL_PROMPT = "Table: {table}\nQuestion: {question}\nSQL:" # @param {"type":"string"}
GENERATOR_MODEL = "gpt-4.1-nano-2025-04-14" # @param {"type":"string"}
TEACHER_MODEL = "gpt-5" # @param {"type":"string"}

print(f"✅ Ready to optimize! We will improve `{INITIAL_PROMPT}`")


optimizer = GEPAOptimizer(reflection_model=TEACHER_MODEL,
                          generator_model=GENERATOR_MODEL)

# @title
results = optimizer.optimize(
    evaluator=evaluator,
    data_mapper=data_mapper,
    dataset=dataset,
    initial_prompts=[INITIAL_PROMPT],
    max_metric_calls=100  # Since our dataset is small and isn't too complex, a lower limit should suffice.
)


# @title Best Prompt Found
print("Best Prompt Template:")
print(f"{results.best_generator.get_prompt_template()}")

# @title Final Score
print("Final Score:")
print(f"{results.final_score}")


# @title Iteration History
for idx, hist in enumerate(results.history):
    print(f"---- Iteration {idx+1} ----")
    print(f"===PROMPT===\n{hist.prompt}")
    print(f"\n\n===AVERAGE SCORE===\n{hist.average_score}")