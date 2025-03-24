import ast
import json
import os
import re
from collections import Counter

import javalang
import textstat
# from readability.readability import Readability



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

def find_duplicate_code(code, block_size=3):
    """
    Find duplicate multi-line code snippets in the input code.
    Uses a sliding window approach to detect repeated blocks of 'block_size' lines.
    """
    lines = code.split("\n")
    block_counts = Counter()
    duplicate_map = {}

    # Create a dictionary to track occurrences of code blocks
    for i in range(len(lines) - block_size + 1):
        block = "\n".join(lines[i:i + block_size])  # Extract multi-line block
        block_counts[block] += 1

        if block_counts[block] > 1:  # ✅ Only store if it's a duplicate
            if block not in duplicate_map:
                duplicate_map[block] = []
            duplicate_map[block].append(i + 1)  # Store line number of first occurrence

    return duplicate_map


# def calculate_comment_density(code):
#     """Compute comment density (ratio of comments to code) for Python code."""
#     lines = code.split("\n")
#
#     # Initialize counters
#     comment_lines = 0
#     code_lines = 0
#     inside_docstring = False
#
#     for line in lines:
#         stripped = line.strip()
#
#         # ✅ Handle multi-line docstrings properly (''' or """ in the line)
#         if stripped.startswith(('"""', "'''")) and stripped.endswith(('"""', "'''")) and len(stripped) > 3:
#             comment_lines += 1  # Single-line docstring
#             continue
#         elif stripped.startswith(('"""', "'''")):
#             inside_docstring = True
#             comment_lines += 1
#             continue
#         elif stripped.endswith(('"""', "'''")):
#             inside_docstring = False
#             comment_lines += 1
#             continue
#
#         if inside_docstring:
#             comment_lines += 1
#             continue  # ✅ Skip to next line
#
#         # ✅ Count single-line comments
#         if stripped.startswith("#"):
#             comment_lines += 1
#             continue  # ✅ Move to next line
#
#         # ✅ Detect inline comments (code followed by #)
#         if "#" in stripped:
#             before_comment, after_comment = stripped.split("#", 1)
#             if after_comment.strip():  # Ensure there's actual comment content
#                 comment_lines += 1
#                 if before_comment.strip():  # If code is before comment, count it
#                     code_lines += 1
#                 continue
#
#         # ✅ Count non-comment, non-empty lines as code
#         if stripped:
#             code_lines += 1
#
#     total_lines = code_lines + comment_lines
#     return round(comment_lines / total_lines, 2) if total_lines > 0 else 0  # ✅ Prevent division by zero

def calculate_comment_density(code):
    """Count the number of comment lines in Python code."""
    lines = code.split("\n")
    comment_lines = 0
    inside_docstring = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Handle multi-line docstrings
        if stripped.startswith(('"""', "'''")):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                comment_lines += 1
                continue
            else:
                inside_docstring = not inside_docstring
                comment_lines += 1
                continue

        if inside_docstring:
            comment_lines += 1
            continue

        # Count standalone comments
        if stripped.startswith("#"):
            comment_lines += 1
            continue

        # Count inline comments
        if "#" in stripped:
            quote_count = stripped.count('"') + stripped.count("'")
            if quote_count % 2 == 0:
                comment_lines += 1

    return comment_lines





def calculate_readability_score(code):
    """Estimate readability of Python comments using textstat, ensuring no negative scores."""
    comment_lines = [line.strip().split("#", 1)[1] for line in code.split("\n") if "#" in line]
    comments = " ".join(comment_lines)

    if not comments.strip():
        return 0  # No comments = readability score of 0

    try:
        readability_score = textstat.flesch_reading_ease(comments)

        # ✅ Normalize scores (Clamp to a minimum of 0)
        return max(round(readability_score, 2), 5)
    except Exception as e:
        print(f"❌ Readability Calculation Error: {e}")
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
    duplicate_code_details = find_duplicate_code(code)  # ✅ Get duplicate lines & locations
    comment_density = calculate_comment_density(code)
    readability_score = calculate_readability_score(code)
    complexity_score = calculate_complexity_score(loc, num_functions, duplicate_percentage)

    return {
        "lines_of_code": loc,
        "effective_lines_of_code": eloc,
        "num_functions": num_functions,
        "avg_function_length": avg_function_length,
        "duplicate_code_percentage": duplicate_percentage,
        "duplicate_code_details": duplicate_code_details,  # ✅ Include duplicate details
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



#################### java ############################

def java_load_guidelines():
    """Load Java complexity metric guidelines from a JSON file."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "complexity_guidelines_java.json")

        with open(json_path, "r") as file:
            guidelines = json.load(file)
        return guidelines
    except Exception as e:
        print(f"❌ Error loading Java guidelines: {e}")
        return {}


def java_count_lines_of_code(code):
    """Calculate total and effective lines of code in Java code."""
    lines = code.split("\n")
    total_lines = len(lines)
    effective_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("//"))
    return total_lines, effective_lines


def java_count_classes_and_methods(code):
    """Count the number of classes and methods in Java code."""
    try:
        tree = javalang.parse.parse(code)
        classes = list(tree.filter(javalang.tree.ClassDeclaration))
        methods = list(tree.filter(javalang.tree.MethodDeclaration))
    except Exception:
        return 0, 0, 0  # Return zeros if parsing fails

    num_classes = len(classes)
    num_methods = len(methods)
    method_lengths = [len(m.body) if m.body else 0 for _, m in methods]
    avg_method_length = sum(method_lengths) / num_methods if num_methods > 0 else 0

    return num_classes, num_methods, avg_method_length


# def java_calculate_cyclomatic_complexity(code):
#     """Estimate cyclomatic complexity based on control structures in Java."""
#     complexity_keywords = ["if", "for", "while", "case", "catch", "&&", "||", "?"]
#     complexity_score = sum(len(re.findall(fr"\b{kw}\b", code)) for kw in complexity_keywords) + 1
#     return complexity_score


def java_calculate_nesting_depth(code):
    """Estimate the maximum nesting depth of Java control structures."""
    max_depth = 0
    current_depth = 0
    for line in code.split("\n"):
        stripped_line = line.strip()
        if re.match(r"^(if|for|while|try|catch|switch)\b", stripped_line):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        if stripped_line == "}":
            current_depth = max(0, current_depth - 1)
    return max_depth


def java_count_duplicate_code_percentage(code):
    """Calculate the percentage of duplicate lines in Java code."""
    lines = code.split("\n")
    duplicates = [item for item, count in Counter(lines).items() if count > 1 and item.strip()]
    return (len(duplicates) / len(lines)) * 100 if lines else 0

def java_find_duplicate_code(code, block_size=3):
    """
    Improved duplicate code detection in Java.
    Uses structured splitting and filters incomplete statements.
    """
    lines = code.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]  # ✅ Remove blank lines

    block_counts = Counter()
    duplicate_map = {}

    # ✅ Track extracted duplicates to prevent redundancy
    extracted_blocks = set()

    for i in range(len(cleaned_lines) - block_size + 1):
        block = "\n".join(cleaned_lines[i:i + block_size])  # Extract multi-line block

        # ✅ Ignore tiny fragments (e.g., single braces, empty lines)
        if len(re.findall(r'\w+', block)) < 2:  # ✅ Ensure at least 2 meaningful words
            continue

        block_counts[block] += 1

        if block_counts[block] > 1 and block not in extracted_blocks:  # ✅ Avoid redundant detections
            if block not in duplicate_map:
                duplicate_map[block] = []
            duplicate_map[block].append(i + 1)  # Store line number of first occurrence
            extracted_blocks.add(block)  # ✅ Store extracted block to prevent duplicates

    return duplicate_map


def java_calculate_comment_density(code):
    """Compute the ratio of comments to code lines in Java."""
    lines = code.split("\n")

    # Java single-line (`//`) and multi-line (`/* ... */`) comments
    single_line_comments = sum(1 for line in lines if line.strip().startswith("//"))
    multi_line_comments = re.findall(r"/\*[\s\S]*?\*/", code, re.MULTILINE)
    multi_line_comment_lines = sum(comment.count("\n") + 1 for comment in multi_line_comments)

    total_comment_lines = single_line_comments + multi_line_comment_lines
    code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("//"))

    return total_comment_lines / code_lines if code_lines > 0 else 0


def java_calculate_readability_score(code):
    """Estimate the readability of Java comments using textstat."""
    # Extract single-line comments (`//`)
    comment_lines = [line.strip()[2:] for line in code.split("\n") if line.strip().startswith("//")]

    # Extract multi-line comments (`/* ... */`)
    multi_line_comments = re.findall(r"/\*([\s\S]*?)\*/", code, re.MULTILINE)

    # Combine all comments into one text block
    comments = "\n".join(comment_lines + multi_line_comments)

    # If there are no comments, return 0
    if not comments.strip():
        return 0

    try:
        # Calculate readability score using textstat (Flesch-Kincaid readability test)
        readability_score = textstat.flesch_kincaid_grade(comments)
        return round(readability_score, 2)
    except Exception as e:
        print(f"❌ Readability Calculation Error: {e}")
        return 0

# def java_calculate_readability_score(code):
#     """Estimate the readability of Java comments using readability metrics."""
#     comment_lines = [line.strip() for line in code.split("\n") if line.strip().startswith("//")]
#     multi_line_comments = re.findall(r"/\*[\s\S]*?\*/", code, re.MULTILINE)
#     comments = "\n".join(comment_lines + multi_line_comments)
#
#     if not comments:
#         return 0
#
#     try:
#         r = Readability(comments)
#         return r.flesch_kincaid().score
#     except Exception:
#         return 0


def java_calculate_complexity_score(loc, num_methods, duplication):
    """Calculate a composite complexity score for Java code."""
    return (loc * 0.2) + (num_methods * 0.3) + (duplication * 0.2)


def java_analyze_code_complexity(code):
    """Analyze Java code complexity using various metrics."""
    guidelines = java_load_guidelines()

    loc, eloc = java_count_lines_of_code(code)
    num_classes, num_methods, avg_method_length = java_count_classes_and_methods(code)
    # cyclomatic_complexity = java_calculate_cyclomatic_complexity(code)
    nesting_depth = java_calculate_nesting_depth(code)
    duplicate_percentage = java_count_duplicate_code_percentage(code)
    duplicate_code_details = java_find_duplicate_code(code)
    comment_density = java_calculate_comment_density(code)
    readability_score = java_calculate_readability_score(code)
    complexity_score = java_calculate_complexity_score(loc, num_methods, duplicate_percentage)

    return {
        "lines_of_code": loc,
        "effective_lines_of_code": eloc,
        "num_classes": num_classes,
        "num_methods": num_methods,
        "avg_method_length": avg_method_length,
        # "cyclomatic_complexity": cyclomatic_complexity,
        "nesting_depth": nesting_depth,
        "duplicate_code_percentage": duplicate_percentage,
        "duplicate_code_details": duplicate_code_details,
        "comment_density": comment_density,
        "readability_score": readability_score,
        "complexity_score": complexity_score,
        "rating": {
            "lines_of_code": java_categorize_value(loc, guidelines["lines_of_code"]),
            "comment_density": java_categorize_value(comment_density, guidelines["code_density"]),
            "method_length": java_categorize_value(avg_method_length, guidelines["function_length"]),
            "duplicate_code": java_categorize_value(duplicate_percentage, guidelines["duplicate_code"]),
            "num_methods": java_categorize_value(num_methods, guidelines["num_functions"]),
            # "cyclomatic_complexity": java_categorize_value(cyclomatic_complexity, guidelines["cyclomatic_complexity"]),
            "complexity_score": java_categorize_value(complexity_score, guidelines["complexity_score"]),
        }
    }


def java_categorize_value(value, thresholds):
    """Categorize a value based on predefined Java thresholds."""
    if value <= thresholds["low"]:
        return "Low"
    elif value <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"

