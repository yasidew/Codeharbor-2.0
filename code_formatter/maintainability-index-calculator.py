import re
import math

def calculate_complexity(code):
    """Calculate Cyclomatic Complexity."""
    complexity_keywords = [
        r'\bif\b', r'\belif\b', r'\belse\b',
        r'\bfor\b', r'\bwhile\b',
        r'\btry\b', r'\bexcept\b',
        r'\bbreak\b', r'\bcontinue\b',
        r'\breturn\b', r'\braise\b',
        r'\bswitch\b', r'\bcase\b'
    ]
    complexity_pattern = '|'.join(complexity_keywords)
    matches = re.findall(complexity_pattern, code)
    return 1 + len(matches)  # Complexity starts at 1

def count_operators_and_operands(code):
    """Counts operators and operands."""
    operators = re.findall(r"[+\-*/%=<>!&|^~]", code)  # Operators
    operands = re.findall(r"[a-zA-Z_]\w*", code)  # Variable names

    unique_operators = len(set(operators))
    unique_operands = len(set(operands))

    return len(operators), len(operands), unique_operators, unique_operands

def calculate_halstead_volume(operators_count, operands_count, unique_operators, unique_operands):
    """Calculate Halstead Volume."""
    if unique_operators + unique_operands == 0:
        return 0  # Prevent log(0)
    return (operators_count + operands_count) * math.log2(unique_operators + unique_operands)

def calculate_maintainability_index(cyclomatic_complexity, halstead_volume, loc):
    """Compute Maintainability Index."""
    if loc == 0 or halstead_volume <= 0:
        return 0
    mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(loc)
    return max(0, min(100, mi))  # Scale between 0-100

def calculate_loc(code):
    """Calculate Lines of Code (LOC)."""
    return len([line for line in code.split('\n') if line.strip() != ''])

# 🟢 Example Usage
# code = """
# if x > 0:
#     print(x)
# else:
#     x = 2 + 3
# """
#
# complexity = calculate_complexity(code)
# operators_count, operands_count, unique_operators, unique_operands = count_operators_and_operands(code)
# halstead_volume = calculate_halstead_volume(operators_count, operands_count, unique_operators, unique_operands)
# loc = calculate_loc(code)
# maintainability = calculate_maintainability_index(complexity, halstead_volume, loc)

# print(f"LOC: {loc}")
# print(f"Complexity: {complexity}")
# print(f"Halstead Volume: {halstead_volume:.2f}")
# print(f"Maintainability Index: {maintainability:.2f}")
