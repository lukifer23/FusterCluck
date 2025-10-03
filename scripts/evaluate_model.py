#!/usr/bin/env python3
"""Comprehensive evaluation harness for FusterCluck STEM capabilities."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FusterCluckEvaluator:
    """Evaluate FusterCluck model on STEM tasks."""

    def __init__(self, model_path: Optional[Path] = None, tokenizer_path: Optional[Path] = None):
        self.model = None
        self.tokenizer = None
        self.mock_mode = not HAS_TRANSFORMERS
        self.device = "cpu"  # Default to CPU string for mock mode

        if HAS_TRANSFORMERS and torch:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if model_path and tokenizer_path and HAS_TRANSFORMERS:
            self.load_model(model_path, tokenizer_path)
        elif not HAS_TRANSFORMERS:
            logger.warning("Transformers library not available - using mock responses for testing")
            self.mock_mode = True

    def load_model(self, model_path: Path, tokenizer_path: Path):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Loading tokenizer from {tokenizer_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if self.device.type != "mps" else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate response from model (or mock response)."""
        if self.mock_mode:
            return self._generate_mock_response(prompt)
        elif not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        # Use torch.no_grad() context manager
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses for testing without a model."""
        # Simple mock responses based on prompt content
        if "Natalia sold clips" in prompt:
            return "72"
        elif "rope is cut into 4 equal pieces" in prompt:
            return "12"
        elif "15 trees" in prompt:
            return "6"
        elif "3 cars in the parking lot" in prompt:
            return "5"
        elif "32 chocolates" in prompt:
            return "39"
        elif "chemical symbol for gold" in prompt:
            return "Au"
        elif "NOT a fundamental force" in prompt:
            return "Friction"
        elif "derivative of sin(x)" in prompt:
            return "cos(x)"
        elif "len() function" in prompt:
            return "Returns the length of a list"
        elif "binary search" in prompt:
            return "O(log n)"
        elif "starting position" in prompt:
            return "e4 is a good move for white in the opening."
        elif "e4, black played e5" in prompt:
            return "Nf3 would be a solid developing move."
        elif "factorial" in prompt:
            return """```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```"""
        elif "prime" in prompt:
            return """```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```"""
        else:
            return "This is a mock response for testing purposes."


class GSM8KEvaluator:
    """Evaluate on GSM8K math problems."""

    def __init__(self, evaluator: FusterCluckEvaluator):
        self.evaluator = evaluator
        # Load GSM8K test set (subset for demo)
        self.problems = [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "answer": "72"
            },
            {
                "question": "A rope is cut into 4 equal pieces. Each piece is 3 feet long. What is the original length of the rope?",
                "answer": "12"
            },
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "6"
            },
            {
                "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "answer": "5"
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "39"
            }
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Run GSM8K evaluation."""
        logger.info("Running GSM8K evaluation...")

        correct = 0
        total = len(self.problems)

        for i, problem in enumerate(self.problems):
            question = problem["question"]
            expected_answer = problem["answer"]

            prompt = f"<sys>You are a STEM assistant.</sys>\nInstruction: {question}\nAssistant: Let me solve this step by step.\n\n<reasoning>"

            try:
                response = self.evaluator.generate_response(prompt, max_new_tokens=256)

                # Extract final answer (look for numbers in the response)
                numbers = re.findall(r'\b\d+\b', response)
                predicted_answer = numbers[-1] if numbers else "0"

                is_correct = predicted_answer == expected_answer

                if is_correct:
                    correct += 1

                logger.info(f"GSM8K {i+1}/{total}: {'✅' if is_correct else '❌'} "
                           f"Expected: {expected_answer}, Got: {predicted_answer}")

            except Exception as e:
                logger.error(f"Error evaluating problem {i+1}: {e}")

        accuracy = correct / total if total > 0 else 0

        return {
            "task": "GSM8K",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": f"{correct}/{total} correct"
        }


class MMLUSTEMEvaluator:
    """Evaluate on MMLU STEM questions."""

    def __init__(self, evaluator: FusterCluckEvaluator):
        self.evaluator = evaluator
        # Sample MMLU STEM questions
        self.questions = [
            {
                "question": "What is the chemical symbol for gold?",
                "options": ["Au", "Ag", "Fe", "Cu"],
                "correct": "Au"
            },
            {
                "question": "Which of the following is NOT a fundamental force in physics?",
                "options": ["Gravity", "Electromagnetism", "Strong nuclear", "Friction"],
                "correct": "Friction"
            },
            {
                "question": "What is the derivative of sin(x)?",
                "options": ["cos(x)", "-sin(x)", "tan(x)", "sec(x)"],
                "correct": "cos(x)"
            },
            {
                "question": "In Python, what does 'len()' function do?",
                "options": ["Returns the length of a list", "Returns the last element", "Sorts the list", "Reverses the list"],
                "correct": "Returns the length of a list"
            },
            {
                "question": "What is the time complexity of binary search?",
                "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
                "correct": "O(log n)"
            }
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Run MMLU STEM evaluation."""
        logger.info("Running MMLU STEM evaluation...")

        correct = 0
        total = len(self.questions)

        for i, q in enumerate(self.questions):
            question = q["question"]
            options = q["options"]
            expected = q["correct"]

            prompt = f"<sys>You are a STEM assistant.</sys>\nInstruction: {question}\nOptions: {', '.join(options)}\nAssistant:"

            try:
                response = self.evaluator.generate_response(prompt, max_new_tokens=128)

                # Check if expected answer appears in response
                is_correct = expected.lower() in response.lower()

                if is_correct:
                    correct += 1

                logger.info(f"MMLU {i+1}/{total}: {'✅' if is_correct else '❌'} "
                           f"Expected: {expected}")

            except Exception as e:
                logger.error(f"Error evaluating question {i+1}: {e}")

        accuracy = correct / total if total > 0 else 0

        return {
            "task": "MMLU_STEM",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": f"{correct}/{total} correct"
        }


class ChessEvaluator:
    """Evaluate chess position analysis."""

    def __init__(self, evaluator: FusterCluckEvaluator):
        self.evaluator = evaluator
        # Simple chess positions
        self.positions = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "question": "What is the best move for white in the starting position?",
                "expected_keywords": ["e4", "d4", "Nf3", "Nc3"]
            },
            {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
                "question": "White just played e4, black played e5. What should white do next?",
                "expected_keywords": ["Nf3", "Bc4", "d4", "Qh5"]
            }
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Run chess evaluation."""
        logger.info("Running chess evaluation...")

        correct = 0
        total = len(self.positions)

        for i, pos in enumerate(self.positions):
            question = pos["question"]
            keywords = pos["expected_keywords"]

            prompt = f"<sys>You are a STEM assistant.</sys>\nInstruction: {question}\nAssistant:"

            try:
                response = self.evaluator.generate_response(prompt, max_new_tokens=256)

                # Check if any expected keyword appears in response
                has_keyword = any(keyword.lower() in response.lower() for keyword in keywords)

                if has_keyword:
                    correct += 1

                logger.info(f"Chess {i+1}/{total}: {'✅' if has_keyword else '❌'} "
                           f"Expected keywords: {keywords}")

            except Exception as e:
                logger.error(f"Error evaluating position {i+1}: {e}")

        accuracy = correct / total if total > 0 else 0

        return {
            "task": "Chess_Analysis",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": f"{correct}/{total} correct"
        }


class CodeEvaluator:
    """Evaluate code generation capabilities."""

    def __init__(self, evaluator: FusterCluckEvaluator):
        self.evaluator = evaluator
        self.test_cases = [
            {
                "prompt": "Write a Python function to calculate factorial",
                "test_code": """
def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1
    assert factorial(1) == 1
    print("All tests passed!")
""",
                "validation": "factorial"
            },
            {
                "prompt": "Write a Python function to check if a number is prime",
                "test_code": """
def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(4) == False
    assert is_prime(17) == True
    print("All tests passed!")
""",
                "validation": "is_prime"
            }
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Run code evaluation."""
        logger.info("Running code evaluation...")

        correct = 0
        total = len(self.test_cases)

        for i, test_case in enumerate(self.test_cases):
            prompt = test_case["prompt"]
            test_code = test_case["test_code"]
            function_name = test_case["validation"]

            full_prompt = f"<sys>You are a STEM assistant.</sys>\nInstruction: {prompt}\nAssistant: Here is the Python code:\n\n```python\n"

            try:
                response = self.evaluator.generate_response(full_prompt, max_new_tokens=256)

                # Extract code from response (look for ```python blocks)
                code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1)
                else:
                    # Try to extract code directly
                    generated_code = response.strip()

                # Create test script
                test_script = f"{generated_code}\n\n{test_code}"

                # Test the code
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(test_script)
                    temp_file = f.name

                try:
                    result = subprocess.run(
                        ['python3', temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    is_correct = result.returncode == 0 and "All tests passed!" in result.stdout

                    if is_correct:
                        correct += 1

                    logger.info(f"Code {i+1}/{total}: {'✅' if is_correct else '❌'}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"Code {i+1}/{total}: Timeout")
                except Exception as e:
                    logger.error(f"Error testing code {i+1}: {e}")
                finally:
                    Path(temp_file).unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Error generating code for test {i+1}: {e}")

        accuracy = correct / total if total > 0 else 0

        return {
            "task": "Code_Generation",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": f"{correct}/{total} correct"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FusterCluck model on STEM tasks")
    parser.add_argument("--model-path", type=Path, help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=Path, help="Path to tokenizer")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_results.json"),
                       help="Output file for results")
    parser.add_argument("--tasks", nargs="+",
                       choices=["gsm8k", "mmlu", "chess", "code", "all"],
                       default=["all"],
                       help="Evaluation tasks to run")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = FusterCluckEvaluator()

    if args.model_path and args.tokenizer_path:
        evaluator.load_model(args.model_path, args.tokenizer_path)
    else:
        logger.warning("No model specified - running evaluation with mock responses for testing")

    # Initialize evaluators
    evaluators = {
        "gsm8k": GSM8KEvaluator(evaluator),
        "mmlu": MMLUSTEMEvaluator(evaluator),
        "chess": ChessEvaluator(evaluator),
        "code": CodeEvaluator(evaluator)
    }

    # Determine which tasks to run
    tasks_to_run = evaluators.keys() if "all" in args.tasks else args.tasks

    # Run evaluations
    results = {}
    total_score = 0
    task_count = 0

    logger.info("🚀 Starting FusterCluck evaluation...")
    logger.info(f"📋 Tasks: {', '.join(tasks_to_run)}")

    for task_name in tasks_to_run:
        if task_name in evaluators:
            result = evaluators[task_name].evaluate()
            results[task_name] = result
            total_score += result["accuracy"]
            task_count += 1

            logger.info(".1%")

    # Overall results
    if task_count > 0:
        overall_score = total_score / task_count
        results["overall"] = {
            "average_accuracy": overall_score,
            "tasks_evaluated": task_count,
            "summary": f"{overall_score:.1%} average across {task_count} tasks"
        }

        logger.info("=" * 60)
        logger.info("📊 FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(".1%")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"💾 Results saved to {args.output}")

    # Return overall score for automation
    if "overall" in results:
        print(".3f")


if __name__ == "__main__":
    main()
