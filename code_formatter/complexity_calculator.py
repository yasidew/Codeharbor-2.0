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
    """
    Calculate Lines of Code (LOC), excluding empty lines, comments, and import statements.

    Steps:
    - Remove empty lines.
    - Exclude single-line comments (`//` in Java, `#` in Python).
    - Exclude multi-line comments (`/* ... */` in Java, `''' ... '''` in Python).
    - Ignore import statements (`import` in Python, `import ...;` in Java).

    Returns:
    - The total number of meaningful lines of code.
    """
    # Remove multi-line comments (Python and Java)
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?"""|/\*[\s\S]*?\*/', '', code, flags=re.MULTILINE)

    # Remove single-line comments (Java `// ...` and Python `# ...`)
    code = re.sub(r'//.*|#.*', '', code)

    # Remove import statements (Python & Java)
    code = re.sub(r'^\s*import .*', '', code, flags=re.MULTILINE)

    # Count non-empty, meaningful lines
    return len([line for line in code.split('\n') if line.strip() != ''])


def calculate_readability(code):
    """
    Calculate a basic readability score for code.

    Readability formula (modified Flesch-Kincaid-based method):
    - A higher score means better readability.
    - Factors considered:
      - **Total words**: All tokens in the code.
      - **Logical Units** (instead of "sentences"):
        - Code lines ending in `;`, `{}`, or indentation changes are considered logical units.
        - Function/method calls, loops, and conditionals contribute to readability.

    Formula:
        Readability Score = 206.835 - 1.015 * (Words per Logical Unit) - 84.6

    Returns:
    - A numerical readability score (higher is better).
    """

    # Identify logical code "units" (instead of treating periods `.` as sentence endings)
    logical_units = code.count(';') + code.count('{') + code.count('}') + code.count('\n')

    # Extract words/tokens (excluding punctuation)
    words = len(re.findall(r'\b\w+\b', code))

    # Avoid division by zero
    if logical_units == 0:
        return 0

    # Readability calculation (simplified)
    return round(206.835 - 1.015 * (words / logical_units) - 84.6, 2)
