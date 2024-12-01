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
