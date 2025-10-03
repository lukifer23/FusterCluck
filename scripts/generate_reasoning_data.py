#!/usr/bin/env python3
"""Generate high-quality reasoning/CoT data with proper formatting."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ReasoningDataGenerator:
    """Generate reasoning traces for STEM problems."""

    def __init__(self):
        # Math problems with solutions
        self.math_problems = [
            {
                "problem": "Solve for x: 2x + 3 = 7",
                "solution": "Subtract 3 from both sides: 2x = 4. Divide by 2: x = 2",
                "topic": "algebra"
            },
            {
                "problem": "What is the derivative of x² + 3x + 1?",
                "solution": "Using power rule: d/dx[x²] = 2x, d/dx[3x] = 3, d/dx[1] = 0. So derivative is 2x + 3",
                "topic": "calculus"
            },
            {
                "problem": "Calculate the integral of 2x dx from 0 to 1",
                "solution": "Antiderivative is x². Evaluate: [x²] from 0 to 1 = 1² - 0² = 1",
                "topic": "calculus"
            },
            {
                "problem": "What is the probability of rolling a 6 on a fair die?",
                "solution": "A die has 6 faces, only one shows 6. Probability = 1/6 ≈ 0.167",
                "topic": "probability"
            },
            {
                "problem": "Find the area of a circle with radius 5",
                "solution": "Area formula: A = πr². A = π(5)² = 25π ≈ 78.54",
                "topic": "geometry"
            }
        ]

        # Physics problems
        self.physics_problems = [
            {
                "problem": "A car accelerates from 0 to 60 mph in 5 seconds. What is its average acceleration?",
                "solution": "Convert to SI units: 60 mph = 26.8 m/s. Time = 5 s. Acceleration = Δv/Δt = (26.8 - 0)/5 = 5.36 m/s²",
                "topic": "kinematics"
            },
            {
                "problem": "What is the gravitational force between two 1kg masses 1 meter apart?",
                "solution": "F = G m₁ m₂ / r². G = 6.67×10⁻¹¹. F = 6.67×10⁻¹¹ × 1 × 1 / 1² = 6.67×10⁻¹¹ N",
                "topic": "gravity"
            },
            {
                "problem": "Calculate the energy of a photon with wavelength 500nm",
                "solution": "E = hc/λ. h = 6.63×10⁻³⁴ J⋅s, c = 3×10⁸ m/s, λ = 500×10⁻⁹ m = 5×10⁻⁷ m. E = (6.63×10⁻³⁴ × 3×10⁸) / 5×10⁻⁷ = 3.98×10⁻¹⁹ J",
                "topic": "quantum"
            }
        ]

        # Code reasoning problems
        self.code_problems = [
            {
                "problem": "Debug this Python code: x = [1,2,3]; print(x[3])",
                "solution": "IndexError: list index out of range. Valid indices are 0, 1, 2. x[3] doesn't exist",
                "topic": "debugging"
            },
            {
                "problem": "What does this regex match: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$",
                "solution": "Email address validation. ^ anchors start, [a-zA-Z0-9._%+-]+ matches username chars, @ literal, domain part, \\. escapes dot, [a-zA-Z]{2,} matches TLD",
                "topic": "regex"
            }
        ]

    def format_reasoning_trace(self, problem: str, solution: str, topic: str) -> str:
        """Format a single reasoning example with proper tags."""
        return f"""<sys>You are a STEM assistant.</sys>
Instruction: {problem}
Assistant: Let me solve this step by step.

<reasoning>
Topic: {topic}
Problem analysis: {problem}

Step-by-step solution:
{solution}

Final answer: {self.extract_final_answer(solution)}
</reasoning>

{self.extract_final_answer(solution)}"""

    def extract_final_answer(self, solution: str) -> str:
        """Extract or generate final answer from solution."""
        # Simple extraction - in real implementation, this would be more sophisticated
        sentences = solution.split('.')
        if sentences:
            return sentences[-1].strip()
        return solution.strip()

    def generate_math_reasoning(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate math reasoning examples."""
        examples = []
        for _ in range(count):
            problem_data = random.choice(self.math_problems)
            trace = self.format_reasoning_trace(
                problem_data["problem"],
                problem_data["solution"],
                problem_data["topic"]
            )
            examples.append({
                "instruction": problem_data["problem"],
                "response": trace,
                "topic": problem_data["topic"],
                "type": "math_reasoning"
            })
        return examples

    def generate_physics_reasoning(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate physics reasoning examples."""
        examples = []
        for _ in range(count):
            problem_data = random.choice(self.physics_problems)
            trace = self.format_reasoning_trace(
                problem_data["problem"],
                problem_data["solution"],
                problem_data["topic"]
            )
            examples.append({
                "instruction": problem_data["problem"],
                "response": trace,
                "topic": problem_data["topic"],
                "type": "physics_reasoning"
            })
        return examples

    def generate_code_reasoning(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate code reasoning examples."""
        examples = []
        for _ in range(count):
            problem_data = random.choice(self.code_problems)
            trace = self.format_reasoning_trace(
                problem_data["problem"],
                problem_data["solution"],
                problem_data["topic"]
            )
            examples.append({
                "instruction": problem_data["problem"],
                "response": trace,
                "topic": problem_data["topic"],
                "type": "code_reasoning"
            })
        return examples

    def generate_complex_reasoning(self, count: int = 200) -> List[Dict[str, Any]]:
        """Generate more complex multi-step reasoning examples."""
        examples = []

        complex_problems = [
            {
                "problem": "A ball is thrown upward with initial velocity 20 m/s. How high does it go? (g=10 m/s²)",
                "solution": "Use v² = u² + 2as. At max height v=0, u=20 m/s, a=-10 m/s², s=? 0 = 400 + 2(-10)s → 20 = s. So maximum height = 20m",
                "topic": "kinematics"
            },
            {
                "problem": "Solve the system: 2x + 3y = 7, x - y = 1",
                "solution": "From second equation: x = y + 1. Substitute: 2(y+1) + 3y = 7 → 2y + 2 + 3y = 7 → 5y = 5 → y = 1. Then x = 1 + 1 = 2",
                "topic": "algebra"
            },
            {
                "problem": "Find the area under y = x² from 0 to 2 using integration",
                "solution": "∫x² dx = (1/3)x³. Evaluate from 0 to 2: (1/3)(8) - (1/3)(0) = 8/3 ≈ 2.667",
                "topic": "calculus"
            }
        ]

        for _ in range(count):
            problem_data = random.choice(complex_problems)
            # Add more detailed reasoning steps
            enhanced_solution = self.enhance_reasoning_steps(problem_data["solution"])
            trace = self.format_complex_reasoning(
                problem_data["problem"],
                enhanced_solution,
                problem_data["topic"]
            )
            examples.append({
                "instruction": problem_data["problem"],
                "response": trace,
                "topic": problem_data["topic"],
                "type": "complex_reasoning"
            })

        return examples

    def enhance_reasoning_steps(self, solution: str) -> str:
        """Add more detailed reasoning steps."""
        # This would be enhanced with more sophisticated step generation
        return solution

    def format_complex_reasoning(self, problem: str, solution: str, topic: str) -> str:
        """Format complex reasoning with more detailed traces."""
        return f"""<sys>You are a STEM assistant.</sys>
Instruction: {problem}
Assistant: This requires careful step-by-step reasoning.

<reasoning>
Topic: {topic}
Problem type: {self.classify_problem_type(problem)}

Step 1: Understand the problem
- What is being asked: {self.extract_question(problem)}
- What information is given: {self.extract_given_info(problem)}

Step 2: Select appropriate method
- For this {topic} problem, I need: {self.select_method(topic)}

Step 3: Apply the method
{solution}

Step 4: Verify the result
- Check units and reasonableness
- Consider edge cases

Final verification: The solution appears correct.
</reasoning>

{self.extract_final_answer(solution)}"""

    def classify_problem_type(self, problem: str) -> str:
        """Classify the type of problem."""
        if any(word in problem.lower() for word in ["solve", "find x", "equation"]):
            return "equation solving"
        elif any(word in problem.lower() for word in ["derivative", "integral", "calculus"]):
            return "calculus"
        elif any(word in problem.lower() for word in ["force", "velocity", "acceleration"]):
            return "physics"
        else:
            return "general STEM"

    def extract_question(self, problem: str) -> str:
        """Extract the question part."""
        return problem

    def extract_given_info(self, problem: str) -> str:
        """Extract given information."""
        return "See problem statement"

    def select_method(self, topic: str) -> str:
        """Select appropriate solution method."""
        methods = {
            "algebra": "equation manipulation and substitution",
            "calculus": "derivative/integral rules",
            "kinematics": "kinematic equations",
            "geometry": "geometric formulas",
            "probability": "probability rules"
        }
        return methods.get(topic, "appropriate mathematical methods")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning/CoT training data")
    parser.add_argument("--output", type=Path, default=Path("data/sft_reasoning"))
    parser.add_argument("--math-count", type=int, default=500, help="Number of math reasoning examples")
    parser.add_argument("--physics-count", type=int, default=300, help="Number of physics reasoning examples")
    parser.add_argument("--code-count", type=int, default=200, help="Number of code reasoning examples")
    parser.add_argument("--complex-count", type=int, default=500, help="Number of complex reasoning examples")

    args = parser.parse_args()

    generator = ReasoningDataGenerator()

    logger.info("🧠 Generating reasoning training data...")

    # Generate different types of reasoning data
    all_examples = []

    logger.info("Generating math reasoning examples...")
    all_examples.extend(generator.generate_math_reasoning(args.math_count))

    logger.info("Generating physics reasoning examples...")
    all_examples.extend(generator.generate_physics_reasoning(args.physics_count))

    logger.info("Generating code reasoning examples...")
    all_examples.extend(generator.generate_code_reasoning(args.code_count))

    logger.info("Generating complex reasoning examples...")
    all_examples.extend(generator.generate_complex_reasoning(args.complex_count))

    # Shuffle to mix different types
    random.shuffle(all_examples)

    # Save to JSONL format
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "reasoning_training.jsonl"

    logger.info(f"💾 Saving {len(all_examples)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Save statistics
    stats = {
        "total_examples": len(all_examples),
        "math_examples": args.math_count,
        "physics_examples": args.physics_count,
        "code_examples": args.code_count,
        "complex_examples": args.complex_count,
        "output_file": str(output_file)
    }

    stats_file = args.output / "generation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info("✅ Reasoning data generation complete!")
    logger.info(f"📊 Generated {len(all_examples)} total examples")


if __name__ == "__main__":
    main()
