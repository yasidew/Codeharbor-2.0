import ast
import json
import os
import re
from collections import Counter
# import textstat
from readability.readability import Readability



def load_guidelines():
    """Load complexity metric guidelines from a JSON file."""
    try:
        # Get the absolute path of the JSON file
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        json_path = os.path.join(base_dir, "complexity_guidelines.json")

        with open(json_path, "r") as file:
            guidelines = json.load(file)
            print(f"✅ Loaded Complexity Guidelines: {guidelines}")  # ✅ Debug Output
            return guidelines
    except Exception as e:
        print(f"❌ Error loading guidelines: {e}")  # ✅ Debug if file isn't found
        return {}


def count_lines_of_code(code):
    """Calculate total and effective lines of code."""
    lines = code.split("\n")
    total_lines = len(lines)
    effective_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    return total_lines, effective_lines

def count_functions_and_length(code):
    """Count the number of functions and their average length."""
    try:
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        function_lengths = [node.end_lineno - node.lineno + 1 for node in functions if hasattr(node, "end_lineno")]

        num_functions = len(functions)
        avg_function_length = sum(function_lengths) / num_functions if num_functions > 0 else 0
        return num_functions, avg_function_length
    except SyntaxError:
        return 0, 0

def count_duplicate_code_percentage(code):
    """Calculate the percentage of duplicate lines in the code."""
    lines = code.split("\n")
    duplicates = [item for item, count in Counter(lines).items() if count > 1 and item.strip()]
    return (len(duplicates) / len(lines)) * 100 if lines else 0

def calculate_comment_density(code):
    """Compute comment density (code-to-comment ratio)."""
    lines = code.split("\n")
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))

    return code_lines / comment_lines if comment_lines > 0 else 0

def calculate_readability_score(code):
    """Estimate the readability of the code's comments."""
    comments = "\n".join([line.strip() for line in code.split("\n") if line.strip().startswith("#")])
    if not comments:
        return 0
    try:
        r = Readability(comments)
        return r.flesch_kincaid().score
    except:
        return 0

def calculate_complexity_score(loc, functions, duplication):
    """Calculate a simple composite complexity score."""
    return (loc * 0.3) + (functions * 0.4) + (duplication * 0.3)

def analyze_code_complexity(code):
    """Analyze code complexity using various metrics."""
    guidelines = load_guidelines()

    loc, eloc = count_lines_of_code(code)
    num_functions, avg_function_length = count_functions_and_length(code)
    duplicate_percentage = count_duplicate_code_percentage(code)
    comment_density = calculate_comment_density(code)
    readability_score = calculate_readability_score(code)
    complexity_score = calculate_complexity_score(loc, num_functions, duplicate_percentage)

    return {
        "lines_of_code": loc,
        "effective_lines_of_code": eloc,
        "num_functions": num_functions,
        "avg_function_length": avg_function_length,
        "duplicate_code_percentage": duplicate_percentage,
        "comment_density": comment_density,
        "readability_score": readability_score,
        "complexity_score": complexity_score,
        "rating": {
            "lines_of_code": categorize_value(loc, guidelines["lines_of_code"]),
            "comment_density": categorize_value(comment_density, guidelines["code_density"]),
            "function_length": categorize_value(avg_function_length, guidelines["function_length"]),
            "duplicate_code": categorize_value(duplicate_percentage, guidelines["duplicate_code"]),
            "num_functions": categorize_value(num_functions, guidelines["num_functions"]),
            "complexity_score": categorize_value(complexity_score, guidelines["complexity_score"]),
        }
    }

def categorize_value(value, thresholds):
    """Categorize a value based on predefined thresholds."""
    if value <= thresholds["low"]:
        return "Low"
    elif value <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"


