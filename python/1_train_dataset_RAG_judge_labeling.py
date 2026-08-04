import os
import re
import json
import argparse
import asyncio
import logging
from typing import List, Dict

# Third-party libraries
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tqdm.asyncio import tqdm as async_tqdm

# ----------------- Command-Line Interface (CLI) Setup -----------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel Gemini evaluation for medical QA solutions."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Directory to save the final labeled output."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Your Google AI (Gemini) API key."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="Maximum number of concurrent requests to the Gemini API."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The Gemini model to use for evaluation."
    )
    return parser.parse_args()


# ----------------- Gemini Model Setup -----------------
def setup_gemini_model(api_key: str, model_name: str, system_instruction: str = None) -> genai.GenerativeModel:
    """Configure and create a Gemini model instance."""
    genai.configure(api_key=api_key)
    
    if system_instruction:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
    else:
        model = genai.GenerativeModel(
            model_name=model_name
        )
    return model

# ----------------- Core Logic (Updated Functions) -----------------
def construct_full_question(
    question_text: str,
    correct_answer: str,
    related_docs: List[str] = None
) -> str:
    """Build a single prompt string from question components."""
    parts = []
    
    if related_docs:
        docs_block = "\n\n".join(f"Doc {i+1}:\n{doc}" for i, doc in enumerate(related_docs))
        parts.append(docs_block)
    
    parts.append(f"Question:\n{question_text}")
    parts.append(f"Correct Answer: ({correct_answer})")
    
    return "\n\n".join(parts)

def get_system_instruction() -> str:
    """Get system instruction for wrong solutions evaluation."""
    return (
        """You are an evaluator responsible for assessing the quality of **wrong solutions** to medical questions in a stepwise manner.
Each question is accompanied by relevant documents, a question, and the correct answer, and the quality of reasoning at each step must be evaluated.
Give a score of 0 if the response lacks logical coherence or is not based on medical evidence, and 1 if this is not the case.
Please note that if the explanation does not match the provided ground truth, it must be scored as 0.
Critically assess the reasoning at each step.
At the end of your evaluation, you must include a final summary of the scores in the following format:
## Step 1: 0 or 1
## Step 2: 0 or 1
## Step 3: 0 or 1
..."""
    )

def extract_steps_from_solution(sol: dict) -> int:
    """Get the number of steps by counting 'ки' occurrences in prm_processed_solution."""
    prm_processed_solution = sol.get('prm_processed_solution', '')
    if prm_processed_solution:
        # Determine the number of steps by counting 'ки' occurrences.
        step_count = prm_processed_solution.count('ки')
        return step_count if step_count > 0 else 1
    
    # Fall back to the previous method.
    if isinstance(sol, dict) and 'PRM_score_list_soft' in sol and sol['PRM_score_list_soft']:
        return len(sol['PRM_score_list_soft'])
    
    solution_text = sol.get('solution', '')  # CHANGED from 'solution_text' to 'solution'
    pattern = r"(?i)(^|\n)\s*Step\s+(\d+)\s*:"
    matches = re.findall(pattern, solution_text)
    return len(matches) if matches else 1

def parse_gemini_scores(gemini_response: str, expected_steps: int = None) -> list[int]:
    """
    Parse scores in ``## Step X: Y`` format from a Gemini response.
    If ``expected_steps`` is provided and the result contains twice that many
    scores, return only the first ``expected_steps`` values.
    """
    pattern = r"^##\s*Step\s+(\d+)\s*:\s*([0-1])"
    lines = gemini_response.splitlines()
    
    scores = []
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            scores.append(int(match.group(2)))
    
    # Keep the first half when the response contains duplicate scores.
    if expected_steps is not None and len(scores) == 2 * expected_steps:
        scores = scores[:expected_steps]
    
    return scores

# ----------------- Asynchronous Gemini API Call -----------------
def sync_get_gemini_response(model: genai.GenerativeModel, prompt: str) -> str:
    """Synchronous wrapper for the Gemini API call."""
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.0)
        )
        return response.text.strip() if response and response.text else ""
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}\nPrompt that caused error:\n{prompt[:500]}...")
        return ""

async def get_gemini_response_async(model: genai.GenerativeModel, prompt: str) -> str:
    """Run the synchronous Gemini API call in a separate thread."""
    return await asyncio.to_thread(sync_get_gemini_response, model, prompt)


# ----------------- Asynchronous Worker Task -----------------
async def process_solution_async(
    solution: Dict, 
    full_question_prompt: str,
    api_key: str,
    model_name: str,
    semaphore: asyncio.Semaphore
):
    """
    Process a single solution asynchronously. This function acquires the semaphore,
    calls the Gemini API, parses the result, and updates the solution dictionary.
    """
    async with semaphore:
        # Check the ORM label.
        is_correct = solution.get("orm_label", 0) == 1
        
        if is_correct:
            # For a correct solution, mark every step as correct.
            n_steps = extract_steps_from_solution(solution)
            solution["prm_gemini_label"] = [1] * n_steps
            logging.info(f"Q_ID: {solution.get('question_id', 'N/A')}, Sol_ID: {solution.get('solution_id', 'N/A')} -> Correct solution, all scores set to 1")
        else:
            # For an incorrect solution, ask the Gemini API to score each step.
            n_steps = extract_steps_from_solution(solution)
            sol_text = solution.get("solution", "")  # CHANGED from 'solution_text' to 'solution'
            final_prompt = f"{full_question_prompt}\n\nSolution: {sol_text}"
            
            # Set the system instruction used for incorrect solutions.
            system_instruction = get_system_instruction()
            
            # Create the model.
            model = setup_gemini_model(api_key, model_name, system_instruction)
            
            response_text = await get_gemini_response_async(model, final_prompt)
            
            # Parse and store the scores.
            parsed_scores = parse_gemini_scores(response_text, expected_steps=n_steps)
            
            if not parsed_scores:
                logging.error(f"Failed to parse scores for Q_ID: {solution.get('question_id', 'N/A')}, Sol_ID: {solution.get('solution_id', 'N/A')}")
                logging.error(f"Expected {n_steps} steps, but parsing failed. Raw response: {response_text[:200]}...")
                solution["prm_gemini_label"] = []  # Store an empty list on failure.
            else:
                # Compare against PRM_score_list_soft and adjust the length.
                PRM_score_list_soft = solution.get('PRM_score_list_soft', [])
                if PRM_score_list_soft and len(parsed_scores) == 2 * len(PRM_score_list_soft):
                    # If the result is twice as long, use the first half.
                    parsed_scores = parsed_scores[:len(PRM_score_list_soft)]
                    logging.info(f"Q_ID: {solution.get('question_id', 'N/A')}, Sol_ID: {solution.get('solution_id', 'N/A')} - Using first half of parsed scores (2x length)")
                elif PRM_score_list_soft and len(parsed_scores) != len(PRM_score_list_soft):
                    # Warn when the lengths differ by anything other than 2x.
                    logging.warning(f"Q_ID: {solution.get('question_id', 'N/A')}, Sol_ID: {solution.get('solution_id', 'N/A')} - Length mismatch: expected {len(PRM_score_list_soft)}, got {len(parsed_scores)}")
                
                solution["prm_gemini_label"] = parsed_scores
            
            logging.info(f"Q_ID: {solution.get('question_id', 'N/A')}, Sol_ID: {solution.get('solution_id', 'N/A')} -> Scores: {parsed_scores}")

# ----------------- Main Execution Logic (IMPROVED) -----------------
async def main():
    """Main async function to coordinate the entire process."""
    args = parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the combined log file.
    log_file = os.path.join(output_dir, "complete_processing.log")
    
    # Configure logging so all messages are written to one file.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),  # File handler
            logging.StreamHandler()  # Console handler
        ],
        force=True  # Override any existing logger configuration.
    )
    
    logging.info("=" * 80)
    logging.info("GEMINI RAG LABELING PROCESS STARTED")
    logging.info("=" * 80)
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Concurrency: {args.concurrency}")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 80)

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
        logging.info(f"Successfully loaded {len(all_entries)} questions from input file")
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    
    logging.info(f"Starting processing of {len(all_entries)} questions...")
    
    # --- Improved processing logic ---

    # 1. Create one task list containing every solution for every question.
    all_tasks = []
    for question_data in all_entries:
        question_id = question_data.get('question_id', 'unknown')
        
        full_question_prompt = construct_full_question(
            question_data.get("question", ""),
            question_data.get("correct_answer", "")
        )
        
        for s_idx, sol in enumerate(question_data.get("solutions", [])):
            sol['question_id'] = question_id
            sol['solution_id'] = s_idx
            
            task = process_solution_async(
                sol, full_question_prompt, args.api_key, args.model_name, semaphore
            )
            all_tasks.append(task)
            
    logging.info(f"Created {len(all_tasks)} tasks for all solutions. Starting concurrent processing...")
    
    # 2. Track overall progress with tqdm and run all tasks with asyncio.gather.
    #    This lets the semaphore control concurrency across the complete task pool.
    pbar = async_tqdm(total=len(all_tasks), desc="Processing Solutions")
    
    async def task_wrapper(task, pbar_instance):
        try:
            await task
        except Exception as e:
            # Errors can be logged in process_solution_async or supplemented here.
            # Keep an individual task failure from stopping the full run.
            logging.error(f"A task failed and was caught in wrapper: {e}")
        finally:
            pbar_instance.update(1)

    # Wrap and run every task.
    await asyncio.gather(*(task_wrapper(task, pbar) for task in all_tasks))
    
    pbar.close()

    # --- End improved processing logic ---

    # 3. Save the final output once all processing is complete.
    final_output_path = os.path.join(args.output_dir, "gemini_labeled_complete_all_questions.json")
    logging.info(f"\nSaving final output to: {final_output_path}")
    
    try:
        with open(final_output_path, "w", encoding="utf-8") as f:
            # all_entries now contains the populated prm_gemini_label values.
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        logging.info(f"✓ Final output saved successfully")
    except Exception as e:
        logging.error(f"✗ Failed to save final output: {e}")
        
    # Log the final summary.
    successful_solutions = sum(1 for q in all_entries for s in q.get('solutions', []) if 'prm_gemini_label' in s and s['prm_gemini_label'])
    failed_solutions = len(all_tasks) - successful_solutions
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Total solutions processed: {len(all_tasks)}")
    logging.info(f"Successful: {successful_solutions}")
    logging.info(f"Failed: {failed_solutions}")
    logging.info(f"{'='*80}")
    logging.info("PROCESSING COMPLETED")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main()) 
