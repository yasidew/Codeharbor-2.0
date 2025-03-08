import re

def calculate_complexity(code):
    """
    Calculate cyclomatic complexity of the given code.
    """
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
    complexity = 1 + len(matches)
    return complexity

def calculate_loc(code):
    """Calculate Lines of Code (LOC)."""
    return len([line for line in code.split('\n') if line.strip() != ''])

def calculate_readability(code):
    """Calculate readability score using a basic algorithm."""
    sentences = code.count('.') + code.count(';')
    words = len(code.split())
    if sentences == 0:
        return 0
    return round(206.835 - 1.015 * (words / sentences) - 84.6, 2)
