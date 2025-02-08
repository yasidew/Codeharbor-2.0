import logging
import re

import joblib
import matplotlib
import matplotlib.pyplot as plt

import clr
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pygments.lexers import CSharpLexer
from pygments.token import Token
from pygments import lex

# Load dataset
dataset = pd.read_csv("media/c#_code_features.csv")

# Drop non-numeric columns
dataset = dataset.drop(columns=["file_name"], errors="ignore")

# Ensure dataset contains labels
if "cbo_label" not in dataset.columns:
    raise ValueError("The dataset must contain a 'cbo_label' column!")

# Split features and labels
X = dataset.drop(columns=["cbo_label"])
y = dataset["cbo_label"]

# Handle missing values
X.fillna(0, inplace=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Save trained model
joblib.dump(rf_model, "random_forest_cbo_model.pkl")

# Evaluate model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("✅ Model saved as 'random_forest_cbo_model.pkl'")

# Load the trained model
rf_model = joblib.load("random_forest_cbo_model.pkl")

# Global dictionary for tracking inheritance depth
inheritance_depth = {}

# Global flag to detect when the class declaration ends
class_declaration_ended = False


# Function to remove comments (both single-line and multi-line)
def remove_comments(line):
    # Remove single-line comments (//)
    line = re.sub(r'//.*', '', line)
    # Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*[\s\S]*?\*/', '', line)
    return line


# Function to calculate the size of a single line (token count)
def calculate_size(line):
    global class_declaration_ended

    line = remove_comments(line)

    # Check for class declaration and skip until it's completed
    if not class_declaration_ended:
        # Detect the class declaration pattern
        if re.search(r'\b(class|struct)\s+\w+', line):
            class_declaration_ended = True
            return 0, []  # Do not count tokens until after class declaration
        return 0, []  # Still in class declaration, so ignore

    # Remove annotations like `[Obsolete]`
    line = re.sub(r'\[\w+\]', '', line)

    # Exclude access modifiers and function parameters
    line = re.sub(r'\b(public|private|protected|internal|abstract|static|sealed|readonly|virtual|override|unsafe|async|extern|else)\b', '', line)
    line = re.sub(r'\(\s*\w+\s*\w*\s*\)', '()', line)  # Ignore function parameters

    # Token patterns based on WCC rules
    token_pattern = r'''
            "[^"]*"                 # Strings inside double quotes
            | '[^']*'               # Strings inside single quotes
            | \+\+|--                # Pre and post increment/decrement (++i, --i, i++, i--)
            | \b(?:if|for|while|switch|case|default|catch)\b\s*\(  # Control structures with brackets
            | \b(?:int|float|double|char|boolean|long|short|byte|void)\b  # Data types
            | &&|\|\|                # Logical operators (&&, ||) as single tokens
            | [\+\*/%=&|<>!~^]       # Operators (except for - which is handled separately)
            | ==|===|>=|<=|!=         # Relational operators considered as one token
            | -?\d+                  # Negative or positive numbers as a single token
            | \.                     # Dot operator treated as a separate token
            | \d+                     # Numerical values
            | [a-zA-Z_]\w*            # Identifiers
        '''

    # Remove ignored tokens like return, try, and ;
    line = re.sub(r'\b(return|try)\b', '', line)
    line = re.sub(r';', '', line)

    # Tokenize the statement based on WCC rules
    tokens = re.findall(token_pattern, line, re.VERBOSE)

    return len(tokens), tokens


def get_code_lines(csharp_code):
    """ Convert C# code into a list of lines for reference. """
    return {i + 1: line.strip() for i, line in enumerate(csharp_code.split("\n"))}


def calculate_control_structure_complexity(lines):
    """
    Calculates control structure complexity for a given C# code.

    Parameters:
        - csharp_code (str): C# code as a string.

    Returns:
        - dict: A dictionary with line numbers as keys and assigned complexity weights as values.
        - int: Total complexity of the code.
    """

    # Split C# code into lines
    # lines = csharp_code.split("\n")
    total_weight = 0
    line_weights = {}

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        # Ignore empty lines
        if not stripped_line:
            continue

        # ✅ Branching Statements (if, else if, else) → Weight = 1
        if re.match(r"^\s*if\b", stripped_line):
            weight = 1
        elif re.match(r"^\s*\}?\s*else\s*if\b", stripped_line):  # Ensure 'else if' is correctly captured
            weight = 1

        # ✅ Loops (for, while, do-while) → Weight = 2
        elif re.match(r"^\s*(for|while|do|foreach)\b", stripped_line):
            weight = 2

        # ✅ Switch-Case Complexity → Weight = Number of cases
        elif re.match(r"^\s*switch\b", stripped_line):
            case_count = 0
            for subsequent_line in lines[line_number:]:
                subsequent_line = subsequent_line.strip()
                if re.match(r"^\s*(case|default)\b", subsequent_line):
                    case_count += 1
                if subsequent_line == "}":  # End of switch block
                    break
            weight = case_count

        else:
            weight = 0  # Default weight for lines without control structures

        # ✅ Store results
        line_weights[line_number] = {
            "line_content": stripped_line,
            "weight": weight
        }
        total_weight += weight

    return line_weights, total_weight


def calculate_nesting_level(java_code):
    """
    Dynamically calculates the nesting level of control structures (if-else, switch-case, loops, etc.)
    in Java-like code line by line, and assigns appropriate weights without hardcoding specific cases.
    """
    # Remove comments from the Java code
    java_code = remove_comments(java_code)
    lines = java_code.splitlines()

    # State variables to track nesting level
    current_nesting = 0
    control_structure_stack = []  # Stack to track nesting levels
    nesting_levels = []
    line_weights = {}

    # Regular expressions for detecting control structures and braces
    control_regex = re.compile(r'\b(if|else if|else|for|while|do|switch|case|default|foreach)\b')
    open_brace_regex = re.compile(r'\{')  # Opening brace
    close_brace_regex = re.compile(r'\}')  # Closing brace

    # Default weight for any control structure
    default_weight = 1

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        # Skip empty lines
        if not stripped_line:
            nesting_levels.append((line_no, stripped_line, current_nesting, 0))
            continue

        # Check if the line contains a control structure
        control_match = control_regex.search(stripped_line)
        if control_match:
            control_type = control_match.group()

            # Assign weight dynamically based on nesting
            weight = default_weight
            if control_type in ['case', 'default']:
                # Special handling for case and default (keep at the same nesting level)
                line_weights[line_no] = weight
            else:
                # Increment nesting level for all other control structures
                current_nesting += 1
                control_structure_stack.append(current_nesting)
                line_weights[line_no] = weight

        # Record the current nesting level and weight for the line
        nesting_levels.append((line_no, stripped_line, current_nesting, line_weights.get(line_no, 0)))

        # Adjust nesting level based on braces
        opening_braces = len(open_brace_regex.findall(stripped_line))
        closing_braces = len(close_brace_regex.findall(stripped_line))

        if closing_braces > 0:
            for _ in range(closing_braces):
                if control_structure_stack:
                    control_structure_stack.pop()
                    current_nesting = len(control_structure_stack)

    return nesting_levels

# Function to calculate inheritance level complexity (Wi)
def calculate_inheritance_level(line, current_inheritance):
    if re.search(r'class\s+\w+\s*:\s*\w+', line):  # Detect inheritance
        current_inheritance += 1
    elif re.search(r'class\s+\w+', line):  # Detect class declaration without inheritance
        current_inheritance = 1
    return current_inheritance


# Function to calculate compound condition complexity
def calculate_compound_condition_weight(line):
    """
        Calculate the complexity of C# compound conditions in a line of code.
        Formula: Complexity = Base Weight + (Number of Subconditions - 1)
        """

    complexity = 0

    # Patterns to detect C# compound conditions
    compound_condition_pattern = r'\(.*?(?:&&|\|\||\?.*:).*?\)'
    logical_operator_pattern = r'&&|\|\|'  # Count logical operators

    # Find all compound conditions in the line
    compound_conditions = re.findall(compound_condition_pattern, line)

    for condition in compound_conditions:
        logical_operators = len(re.findall(logical_operator_pattern, condition))
        subconditions = logical_operators + 1  # Subconditions count
        condition_complexity = 1 + (subconditions - 1)
        complexity += condition_complexity

    # If no compound condition is found, check for simple conditions
    if not compound_conditions:
        simple_condition_pattern = r'\b(if|while|for|switch|return)\s*\(.*?\)'
        simple_conditions = re.findall(simple_condition_pattern, line)
        if simple_conditions:
            complexity += 1  # Assign a base complexity weight

    return complexity

def calculate_try_catch_weight(java_code):
    """
    Calculates the weight of nesting levels specifically for try-catch-finally blocks in Java code.
    - Increment nesting level for `try`.
    - Assign weights line by line for `catch` and `finally` based on nesting level.
    """
    # Remove comments from the Java code
    java_code = remove_comments(java_code)
    lines = java_code.splitlines()

    # State variables
    current_nesting = 0
    nesting_levels = []
    line_weights = {}

    # Regular expressions for try, catch, and finally
    control_regex = re.compile(r'\b(try|catch|finally)\b')

    # Weights for `catch` based on nesting levels
    catch_weights = {0: 1, 1: 1, 2: 3, 3: 4, 4: 5}
    finally_weight = 2

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        if not stripped_line:
            # Skip empty lines
            nesting_levels.append((line_no, stripped_line, current_nesting, 0))
            continue

        # Check for try, catch, or finally
        control_match = control_regex.search(stripped_line)
        if control_match:
            control_type = control_match.group()

            if control_type == 'try':
                # Increment nesting level for `try`
                current_nesting += 1

            elif control_type == 'catch':
                # Assign weight for `catch` based on nesting level
                weight = catch_weights.get(current_nesting, 5)
                line_weights[line_no] = weight

            elif control_type == 'finally':
                # Assign fixed weight for `finally`
                line_weights[line_no] = finally_weight

        # Append the current line and its weight
        nesting_levels.append((line_no, stripped_line, current_nesting, line_weights.get(line_no, 0)))

        # Adjust nesting level for closing braces
        if stripped_line.endswith('}'):
            current_nesting = max(0, current_nesting - 1)

    return nesting_levels, line_weights

def calculate_thread_weight(java_code):
    """
    Calculates the complexity weight for thread-related constructs in Java code.
    Assigns weights for:
    - Thread creation (e.g., `new Thread` or `Runnable`) with weight 2.
    - Thread synchronization (`synchronized`) with weight 3.
    """

    java_code = remove_comments(java_code)
    lines = java_code.splitlines()

    # Regular expressions
    thread_creation_regex = re.compile(r'\bnew\s+Thread\b|\bRunnable\b')
    thread_sync_regex = re.compile(r'\bsynchronized\b')

    # Weights
    thread_creation_weight = 2
    thread_sync_weight = 4

    # Result variables
    line_weights = {}

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        if not stripped_line:
            # Skip empty lines
            continue

        # Initialize weight for the current line
        weight = 0

        # Check for thread creation
        if thread_creation_regex.search(stripped_line):
            weight += thread_creation_weight

        # Check for thread synchronization
        if thread_sync_regex.search(stripped_line):
            weight += thread_sync_weight

        # Store weight if it's greater than 0
        if weight > 0:
            line_weights[line_no] = weight

    return line_weights


def extract_class_references(csharp_code):
    """
    Extracts class references from C# code and assigns weights based on the type of dependency.
    """

    # Regular expressions for detecting class references
    instantiation_pattern = r'new\s+([A-Z][\w]*)\s*\('  # Object instantiation
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'  # Constructor parameters (dependency injection)
    setter_injection_pattern = r'(public|private)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'  # Setter injection
    static_usage_pattern = r'([A-Z][\w]*)\.\w+\s*\('  # Static method call
    static_variable_pattern = r'([A-Z][\w]*)\.\w+'  # Static variable usage
    field_declaration_pattern = r'([A-Z][\w]*)\s+\w+\s*;'  # Field declaration

    excluded_classes = [
        'System', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
        'Math', 'Object', 'Thread', 'Runtime', 'Optional', 'Task', 'Action', 'Func'
    ]

    # Dictionary to store class references per class
    class_references = {}

    # Split code into lines and process each line
    lines = csharp_code.splitlines()
    recommendations = []
    declared_fields = set()

    current_class = None
    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', stripped_line)
        if class_declaration:
            current_class = class_declaration.group(1)
            if current_class not in class_references:
                class_references[current_class] = {}

        if current_class:
            # Detect field declarations (aggregation coupling)
            field_declaration_match = re.search(field_declaration_pattern, stripped_line)
            if field_declaration_match:
                field_class = field_declaration_match.group(1)
                declared_fields.add(field_class)
                class_references[current_class][field_class] = class_references[current_class].get(field_class, 0) + 1

            # Detect instantiation (composition coupling)
            instantiations = re.findall(instantiation_pattern, stripped_line)
            for instantiated_class in instantiations:
                class_references[current_class][instantiated_class] = 3

            # Detect constructor injection
            constructor_matches = re.findall(constructor_pattern, stripped_line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    class_references[current_class][param_class] = class_references[current_class].get(param_class, 0) + 1

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, stripped_line)
            for setter_class in setter_matches:
                class_references[current_class][setter_class] = 1

            # Detect static method usage
            static_methods = re.findall(static_usage_pattern, stripped_line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class, 0) + 2

            # Detect static variable usage
            static_variables = re.findall(static_variable_pattern, stripped_line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class, 0) + 1

    return class_references


def calculate_cbo(class_references):
    """
    Calculates Coupling Between Objects (CBO) for each class.
    """
    cbo_results = {}

    # Sum the weights of class references per class
    for class_name, references in class_references.items():
        cbo_results[class_name] = sum(references.values())

    return cbo_results

def extract_class_references_with_lines(csharp_code):
    """
    Extracts class references from C# code line by line for detailed CBO calculations.
    """
    instantiation_pattern = r'new\s+([A-Z][\w]*)\s*\('
    field_declaration_pattern = r'([A-Z][\w]*)\s+\w+\s*;'
    static_usage_pattern = r'([A-Z][\w]*)\.\w+\s*\('
    static_variable_pattern = r'([A-Z][\w]*)\.\w+'
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'
    setter_injection_pattern = r'(public|private)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'

    excluded_classes = [
        'Console', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
        'Math', 'Object', 'Thread', 'Runtime', 'Optional', 'Task', 'Action', 'Func'
    ]

    line_references = []
    lines = csharp_code.splitlines()
    declared_fields = set()
    current_class = None

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        line_data = {"line": line_no, "code": stripped_line, "weights": {}}

        class_declaration = re.search(r'class\s+([A-Z][\w]*)', stripped_line)
        if class_declaration:
            current_class = class_declaration.group(1)

        if current_class:
            weights = {}

            field_declaration_match = re.search(field_declaration_pattern, stripped_line)
            if field_declaration_match:
                field_class = field_declaration_match.group(1)
                declared_fields.add(field_class)
                weights[field_class] = 1

            for field_class in declared_fields:
                if f"= new {field_class}" in stripped_line:
                    weights[field_class] = 3

            instantiations = re.findall(instantiation_pattern, stripped_line)
            for instantiated_class in instantiations:
                weights[instantiated_class] = 3

            constructor_matches = re.findall(constructor_pattern, stripped_line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    weights[param_class] = 1

            setter_matches = re.findall(setter_injection_pattern, stripped_line)
            for setter_class in setter_matches:
                weights[setter_class] = 1

            static_methods = re.findall(static_usage_pattern, stripped_line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    weights[static_class] = weights.get(static_class, 0) + 2

            static_variables = re.findall(static_variable_pattern, stripped_line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    weights[static_class] = weights.get(static_class, 0) + 1

            line_data["weights"] = weights
        line_references.append(line_data)

    return line_references

def calculate_cbo_line_by_line(line_references):
    """
    Calculates CBO line by line.
    """
    cbo_results = []

    for line_data in line_references:
        total_weight = sum(line_data["weights"].values())
        cbo_results.append({
            "line": line_data["line"],
            "code": line_data["code"],
            "total_weight": total_weight,
            "weights": line_data["weights"]
        })

    return cbo_results

def calculate_code_complexity_multiple(file_contents):
    results = {}

    # Iterate through each file content
    for filename, content in file_contents.items():
        # Extract class references line by line
        line_references = extract_class_references_with_lines(content)
        cbo_line_results = calculate_cbo_line_by_line(line_references)

        # Add results to the file's analysis
        results[filename] = cbo_line_results

    return results


def extract_message_passing(csharp_code):
    """
    Extracts message passing interactions for calculating Message Passing Complexity (MPC) in C# code.
    """
    # Patterns for detecting different types of message passing
    simple_message_pattern = r'\b([a-zA-Z_]\w*)\.\s*([a-zA-Z_]\w*)\s*\('  # Simple method calls
    complex_message_pattern = r'new\s+([A-Z][\w]*)\('  # Object instantiation
    async_message_pattern = r'\b([a-zA-Z_]\w*)\.\s*([a-zA-Z_]\w*)\s*\('  # Asynchronous calls (e.g., Task.Run)
    fluent_pattern = r'\b([a-zA-Z_]\w*)\.\s*([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\('  # Fluent APIs (chained calls)
    linq_pattern = r'\b([a-zA-Z_]\w*)\s*\.\s*(Where|Select|OrderBy|GroupBy|ToList|First|FirstOrDefault)\s*\('  # LINQ queries
    exceptional_pattern = r'\b([a-zA-Z_]\w*)\.\s*(Try|Catch|Finally)\s*\('  # Exceptional handling (e.g., try-catch-finally)
    callback_pattern = r'([a-zA-Z_]\w*)\.\s*(thenApply|thenAccept|thenRun|whenComplete)\s*\('  # Callbacks

    # Dictionary to store the message passing weights per class
    message_passing = {}
    current_class = None

    # Split code into lines and process each line
    lines = csharp_code.splitlines()
    for line in lines:
        line = remove_comments(line)

        # Detect current class declaration
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            if current_class not in message_passing:
                message_passing[current_class] = {}
            logging.info(f"Detected class: {current_class}")

        if current_class:
            # Detect simple method calls
            simple_messages = re.findall(simple_message_pattern, line)
            for method_call in simple_messages:
                method_name = method_call[1]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 1

            # Detect complex message passing (e.g., object instantiation)
            complex_messages = re.findall(complex_message_pattern, line)
            for method_call in complex_messages:
                message_passing[current_class][method_call] = message_passing[current_class].get(method_call, 0) + 2

            # Detect asynchronous calls
            async_messages = re.findall(async_message_pattern, line)
            for async_call in async_messages:
                method_name = async_call[1]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3

            # Detect fluent API calls
            fluent_messages = re.findall(fluent_pattern, line)
            for fluent_call in fluent_messages:
                method_name = fluent_call[2]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3

            # Detect LINQ queries
            linq_calls = re.findall(linq_pattern, line)
            for linq_call in linq_calls:
                method_name = linq_call[1]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 2

            # Detect exceptional handling patterns
            exceptional_calls = re.findall(exceptional_pattern, line)
            for exceptional_call in exceptional_calls:
                method_name = exceptional_call[1]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3

            # Detect callback handling
            callback_messages = re.findall(callback_pattern, line)
            for callback_call in callback_messages:
                method_name = callback_call[1]
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3

    return message_passing


def calculate_mpc(message_passing):
    """
    Calculates Message Passing Complexity (MPC) from extracted interactions.
    """
    mpc_results = {}
    for class_name, messages in message_passing.items():
        mpc_results[class_name] = sum(messages.values())
        logging.info(f"MPC for {class_name}: {mpc_results[class_name]}")
    return mpc_results


def extract_message_passing_with_lines_csharp(csharp_code):
    """
    Extracts message passing interactions for calculating Message Passing Complexity (MPC) in C# code,
    associating weights with specific lines.
    """
    # Patterns to match different types of message passing
    simple_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\('  # Simple method calls
    complex_message_pattern = r'new\s+([A-Z][\w]*)\('  # Object instantiation
    async_message_pattern = r'\bTask\s*\.\s*([a-zA-Z_]\w+)\s*\('  # Asynchronous calls (e.g., Task.Run)
    linq_pattern = r'\b([a-zA-Z_]\w*)\s*\.\s*(Where|Select|OrderBy|GroupBy|ToList|First|FirstOrDefault)\s*\('  # LINQ queries
    fluent_pattern = r'\b([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\.\s*([a-zA-Z_]\w*)\('  # Fluent APIs (chained calls)
    exceptional_pattern = r'try\s*\{|catch\s*\(|finally\s*\{'  # Exception handling patterns

    # Methods to ignore, such as Console.WriteLine
    ignore_methods = {"WriteLine", "ReadLine", "ToString"}

    # Dictionary to store the message passing weights per line
    message_passing_lines = {}

    # Split code into lines and process each line
    lines = csharp_code.splitlines()

    current_class = None
    for line_number, line in enumerate(lines, start=1):
        # Detect current class declaration
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)

        # Initialize the line weight
        if line_number not in message_passing_lines:
            message_passing_lines[line_number] = 0

        # Detect simple method calls
        simple_messages = re.findall(simple_message_pattern, line)
        for method_call in simple_messages:
            method_name = method_call[1]
            if method_name not in ignore_methods:
                message_passing_lines[line_number] += 1  # Weight = 1
                logging.info(f"Simple message passing at line {line_number}: {method_name} (Weight = 1)")

        # Detect complex message passing (e.g., object instantiation)
        complex_messages = re.findall(complex_message_pattern, line)
        for method_call in complex_messages:
            message_passing_lines[line_number] += 2  # Weight = 2
            logging.info(f"Complex message passing at line {line_number}: {method_call} (Weight = 2)")

        # Detect asynchronous calls
        async_messages = re.findall(async_message_pattern, line)
        for async_call in async_messages:
            message_passing_lines[line_number] += 3  # Weight = 3
            logging.info(f"Asynchronous message passing at line {line_number}: {async_call} (Weight = 3)")

        # Detect LINQ queries
        linq_calls = re.findall(linq_pattern, line)
        for linq_call in linq_calls:
            method_name = linq_call[1]
            message_passing_lines[line_number] += 2  # Weight = 2
            logging.info(f"LINQ query at line {line_number}: {method_name} (Weight = 2)")

        # Detect fluent API calls
        fluent_calls = re.findall(fluent_pattern, line)
        for fluent_call in fluent_calls:
            method_name = fluent_call[2]
            message_passing_lines[line_number] += 3  # Weight = 3
            logging.info(f"Fluent API call at line {line_number}: {method_name} (Weight = 3)")

        # Detect exceptional handling patterns
        exceptional_calls = re.findall(exceptional_pattern, line)
        if exceptional_calls:
            message_passing_lines[line_number] += 3  # Weight = 3
            logging.info(f"Exceptional handling at line {line_number} (Weight = 3)")

    return message_passing_lines


def calculate_mpc_line_by_line_csharp(message_passing_lines):
    """
    Aggregates MPC values line by line for C# code.
    """
    mpc_line_results = {}
    for line_number, weight in message_passing_lines.items():
        mpc_line_results[line_number] = weight
        logging.info(f"MPC at line {line_number}: {weight}")
    return mpc_line_results


def calculate_mpc_for_csharp_code(file_contents):
    """
    Processes multiple C# files and calculates line-by-line MPC for each.
    """
    mpc_results = {}

    for filename, csharp_code in file_contents.items():
        # Extract message passing weights with line numbers for the current file
        message_passing_lines = extract_message_passing_with_lines_csharp(csharp_code)

        # Calculate MPC values line by line
        mpc_results[filename] = calculate_mpc_line_by_line_csharp(message_passing_lines)

    return mpc_results

# Function to track inheritance depth across multiple files
def track_inheritance_depth_across_files(file_contents):
    global inheritance_depth  # Reference the global inheritance_depth
    # First pass: record all class declarations (including those with extends)
    for filename, content in file_contents.items():
        lines = content.splitlines()
        for line in lines:
            line = remove_comments(line)
            # Detect class declaration with inheritance
            match_inheritance = re.search(r'class\s+(\w+)\s*:\s*(\w+)', line)
            match_base_class = re.search(r'class\s+(\w+)', line)

            if match_inheritance:
                class_name = match_inheritance.group(1)
                base_class_name = match_inheritance.group(2)

                # Record the inheritance relationship
                inheritance_depth[class_name] = inheritance_depth.get(base_class_name, 1) + 1  # Derived class depth
            elif match_base_class and not re.search(r':', line):
                class_name = match_base_class.group(1)
                if class_name not in inheritance_depth:
                    inheritance_depth[class_name] = 1  # Base class gets depth 1


# Function to retrieve the inheritance level
def calculate_inheritance_level2(class_name):
    return inheritance_depth.get(class_name, 1)

# Load or initialize dataset
data_file = "media/synthetic_training_data_c#_1000.csv"
if os.path.exists(data_file):
    dataset = pd.read_csv(data_file)
else:
    dataset = pd.DataFrame(columns=["control_structure_complexity", "nesting_level", "compound_condition_weight", "try_catch_weight", "current_inheritance", "label"])


# Function to clean and convert dataset
def clean_and_convert_dataset(data):
    numerical_columns = ["control_structure_complexity", "nesting_level", "compound_condition_weight",
                         "try_catch_weight", "current_inheritance"]
    for col in numerical_columns:
        if col in data.columns:
            data[col] = data[col].astype(int)

    if "label" in data.columns:
        data["label"] = data["label"].str.replace('"', '', regex=False)

    return data


dataset = dataset.drop_duplicates()

dataset = clean_and_convert_dataset(dataset)


def train_model(data):
    X = data[["control_structure_complexity", "nesting_level", "compound_condition_weight", "try_catch_weight", "current_inheritance"]]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score Report:")
    print(classification_report(y_test, y_pred))
    return model

model = train_model(dataset)

def update_dataset_and_model(new_data):
    global dataset, model

    # Clean the new data
    new_data = clean_and_convert_dataset(new_data)

    dataset = pd.concat([dataset, new_data], ignore_index=True).drop_duplicates()
    dataset.to_csv(data_file, index=False)
    model = train_model(dataset)

def recommend_action(metrics):
    control_structure_complexity, nesting_level, compound_condition_weight, try_catch_weight, current_inheritance = metrics

    if control_structure_complexity == 1 and nesting_level >= 3:
        if compound_condition_weight >= 3:
            return "Critical refactor: High complexity and if nesting & reduce compound conditional statement"
        return "Critical refactor: High complexity and if nesting"

    if control_structure_complexity == 2 and nesting_level >= 3:
        if compound_condition_weight >= 3:
            return "Critical refactor: High complexity and for loop nesting & reduce compound conditional statement"
        return "Critical refactor: High complexity and for loop nesting"

    if control_structure_complexity >= 3 and nesting_level >= 3:
        return "Critical refactor: High complexity and switch case nesting"

    if try_catch_weight > 5:
        return "Critical refactor: High complexity due to try-catch nesting"

    if current_inheritance > 5:
        return "Critical refactor: Deep inheritance hierarchy, consider flattening the design"

    if current_inheritance > 3:
        return "Consider reducing inheritance levels to improve maintainability"

    if compound_condition_weight > 3:
        if compound_condition_weight > 5:
            return "Critical refactor: Excessive compound conditions, simplify conditional logic"
        return "Consider simplifying compound conditions to improve readability"

    if try_catch_weight > 3 and try_catch_weight <= 5:
        return "Moderate complexity: Try-catch nesting is acceptable but consider flattening for clarity"

    if control_structure_complexity >= 2 and nesting_level > 2:
        return "Moderate complexity: Control structures are manageable but keep them simple"

    return "No action needed"


def ai_recommend_refactoring(new_data):
    recommendations = []
    for line in new_data:
        metrics = [
            line['control_structure_complexity'],
            line['nesting_level'],
            line['compound_condition_weight'],
            line['try_catch_weight'],
            line['inheritance_level'],
        ]

        prediction = model.predict([metrics])[0]

        recommendations.append({
            'line_number': line['line_number'],
            'line_content': line['line_content'],
            'recommendation': prediction
        })

    return recommendations


# Refined method pattern for accurate method detection
method_pattern = re.compile(
    r'^\s*(public|private|protected)?\s*(static\s+)?'  # Access modifiers and 'static' keyword
    r'(\w+\s+)?'  # Optional return type (including void)
    r'(\w+)\s*\([^)]*\)\s*\{'  # Method name and parameter list
)

# Keywords to ignore to prevent detecting control structures as methods
control_keywords = {'if', 'for', 'while', 'switch', 'catch'}

# Function to calculate complexity for each method in a C# file
def calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data, line_weights):
    methods = {}
    method_name = None
    method_lines = []

    nesting_levels = calculate_nesting_level(content)
    nesting_level_dict = {line[0]: line[2] for line in nesting_levels}

    try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)

    thread_weights = calculate_thread_weight(content)

    # Split content by line and analyze each one
    for line in content.splitlines():
        # Detect C# method declarations
        match = re.search(r'\b(?:public|private|protected)?\s*(?:static)?\s*(?:\w+<.*?>|\w+)\s+(\w+)\s*\(.*\)\s*{', line)
        if match:
            method_name = match.group(1)
            if method_name in ["if", "for", "while", "switch", "Thread", "run"]:  # Ignore control structures
                continue
            # If we're already in a method, calculate its complexity
            if method_name:
                methods[method_name] = calculate_complexity_for_method(method_lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights)
            # Start a new method
            method_name = match.group(1)
            method_lines = [line]
        elif method_name:
            # Append lines within the current method
            method_lines.append(line)

    # Final method complexity calculation
    if method_name:
        methods[method_name] = calculate_complexity_for_method(method_lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights)

    return methods


# Helper function to calculate complexity for a C# method based on lines of code
def calculate_complexity_for_method(lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights):
    size = 0
    control_structure_complexity = 0
    nesting_level = 0
    compound_condition_weight = 0
    try_catch_weight = 0
    thread_weight = 0
    current_inheritance_sum = 0
    current_class = class_name
    total_nesting = 0
    total_try_catch_weight = 0
    total_thread_weight = 0
    total_cbo = 0
    total_mpc = 0

    current_nesting = 0
    control_structure_stack = []

    # Analyze each line within the method
    for line_number, line in enumerate(lines, start=1):
        # Calculate size (token count) for this line
        line_size, tokens = calculate_size(line)

        if line_size == 0:
            continue  # Skip this line if size is 0

        total_inheritance = method_inheritance.get(current_class, 0)
        nesting_level = nesting_level_dict.get(line_number, 0)
        total_nesting += nesting_level

        try_catch_weight = try_catch_weight_dict.get(line_number, 0)
        total_try_catch_weight += try_catch_weight

        thread_weight = thread_weights.get(line_number, 0)
        total_thread_weight += thread_weight

        size += line_size
        current_inheritance_sum += total_inheritance

        print("line weight cc >>>>>>>>>>>>>", line_weights)
        control_structure_complexity += line_weights.get(line_number, {}).get("weight", 0)
        print("consrol_structure??????????????", control_structure_complexity)
        cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
        total_cbo_weight = sum(cbo_weights.values())
        total_cbo += total_cbo_weight

        mpc_weight = mpc_line_data.get(line_number, 0)
        total_mpc += mpc_weight

        # Compound condition weight
        compound_condition_weight += calculate_compound_condition_weight(line)

    # Sum up the complexity metrics for this method
    total_complexity = (
            size + control_structure_complexity + total_nesting + current_inheritance_sum +
            compound_condition_weight + total_try_catch_weight + total_thread_weight + total_cbo + 0
    )

    print("current_inheritance_sum", current_inheritance_sum)
    return {
        "size": size,
        "control_structure_complexity": control_structure_complexity,
        "nesting_level": total_nesting,
        "inheritance_level": current_inheritance_sum,
        "compound_condition_weight": compound_condition_weight,
        "try_catch_weight": total_try_catch_weight,
        "thread_weight": total_thread_weight,
        'cbo_weights': total_cbo,
        # 'mpc_weight': total_mpc,
        "total_complexity": total_complexity
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_csharp_class_dependencies(code):
    """
    Extracts class dependencies from C# code using Pygments-based tokenization.
    """

    cbo_features = {
        "class_dependencies": set(),
        "direct_instantiations": 0,
        "static_method_calls": 0,
        "static_variable_usage": 0,
        "interface_implementations": 0,
        "constructor_injections": 0,
        "setter_injections": 0,
        "global_variable_references": 0
    }

    # Common C# types to exclude
    excluded_classes = {
        'System', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
        'Math', 'Object', 'Thread', 'Runtime', 'Task', 'Action', 'Func', 'Console', 'List', 'Dictionary'
    }

    lexer = CSharpLexer()
    tokens = lex(code, lexer)

    current_class = None
    last_token = None
    last_identifier = None

    for ttype, value in tokens:
        # ✅ Detect class declaration
        if ttype == Token.Keyword and value == "class":
            last_token = "class"
            continue
        if last_token == "class" and ttype == Token.Name.Class:
            current_class = value
            cbo_features["class_dependencies"].add(current_class)
            last_token = None
            continue

        # ✅ Detect object instantiations (new ClassName)
        if last_token == "new" and ttype == Token.Name:
            if value not in excluded_classes:
                cbo_features["class_dependencies"].add(value)
                cbo_features["direct_instantiations"] += 1
            last_token = None
            continue

        # ✅ Detect static method calls (ClassName.Method())
        if last_identifier and ttype == Token.Punctuation and value == ".":
            last_token = "static_method"
            continue
        if last_token == "static_method" and ttype == Token.Name.Function:
            if last_identifier not in excluded_classes:
                cbo_features["class_dependencies"].add(last_identifier)
                cbo_features["static_method_calls"] += 1
            last_token = None
            continue

        # ✅ Detect static variable usage (ClassName.Variable)
        if last_identifier and last_token == "static_method":
            if last_identifier not in excluded_classes:
                cbo_features["class_dependencies"].add(last_identifier)
                cbo_features["static_variable_usage"] += 1
            last_token = None
            continue

        # ✅ Detect constructor injection (ConstructorName(params))
        if last_token == "constructor" and ttype == Token.Punctuation and value == "(":
            cbo_features["constructor_injections"] += 1
            last_token = None
            continue

        # ✅ Detect setter injections (`setClass(Class obj)`)
        if last_token == "setter" and ttype == Token.Punctuation and value == "(":
            cbo_features["setter_injections"] += 1
            last_token = None
            continue

        # ✅ Detect global variable usage (static variable)
        if last_token == "static" and ttype == Token.Name:
            cbo_features["global_variable_references"] += 1
            last_token = None
            continue

        # Store last identifier for reference
        if ttype == Token.Name:
            last_identifier = value
        elif ttype == Token.Keyword and value == "new":
            last_token = "new"
        elif ttype == Token.Keyword and value == "static":
            last_token = "static"
        elif ttype == Token.Keyword and value in {"public", "private", "protected", "internal"}:
            last_token = "constructor"

    # Convert class dependency set to count
    cbo_features["class_dependencies"] = len(cbo_features["class_dependencies"])

    return cbo_features

def calculate_code_complexity_multiple_files_csharp(file_contents):
    results = {}
    results3 = {}
    new_patterns = []

    result1 = calculate_code_complexity_multiple(file_contents)

    mpc_results = calculate_mpc_for_csharp_code(file_contents)

    # Step 1: Track inheritance across all files
    track_inheritance_depth_across_files(file_contents)

    # Iterate through each file content
    for filename, content in file_contents.items():
        class_name = filename.split('.')[0]
        current_nesting = 0
        current_inheritance = 0
        in_control_structure = False
        control_structure_stack = []

        # Split content into lines
        lines = content.splitlines()
        complexity_data = []

        # Extract features from Java code
        new_cbo_features = extract_csharp_class_dependencies(content)
        print("new_cbo_features>>>>>>>>>>>>>>>>>>>>>>>>", new_cbo_features)
        X_new = pd.DataFrame([new_cbo_features])[X.columns]

        # Predict CBO issue
        prediction = rf_model.predict(X_new)[0]

        # Generate Recommendations
        recommendations = []
        if X_new["class_dependencies"][0] > 5:
            recommendations.append(
                "⚠️ Reduce class dependencies by using interfaces instead of direct implementations.")
        if X_new["direct_instantiations"][0] > 3:
            recommendations.append("⚠️ Too many direct object instantiations. Use dependency injection instead.")
        if X_new["static_method_calls"][0] > 3:
            recommendations.append("⚠️ Reduce static method calls to improve testability and flexibility.")
        if X_new["static_variable_usage"][0] > 2:
            recommendations.append("⚠️ Minimize static variable usage to prevent global state issues.")
        if X_new["setter_injections"][0] > 1:
            recommendations.append("⚠️ Too many setter injections detected. Prefer constructor injection.")
        if X_new["interface_implementations"][0] > 2:
            recommendations.append("⚠️ Avoid God Interfaces.(interfaces with too many responsibilities)")
        if X_new["global_variable_references"][0] > 2:
            recommendations.append(
                "⚠️ Avoid using global variables. Use dependency injection or encapsulation instead to prevent hidden dependencies."
            )

        # Store results
        results3[filename] = {
            "prediction": "High CBO (Issue)" if prediction == 1 else "Low CBO (Good)",
            "recommendations": recommendations
        }

        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        # print("Method complex",method_complexities)

        nesting_levels = calculate_nesting_level(content)
        nesting_level_dict = {line[0]: line[2] for line in nesting_levels}

        try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)

        thread_weights = calculate_thread_weight(content)

        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare line_complexities for recommendation engine
        line_complexities = []

        method_inheritance = {}

        # Initialize total WCC value for the file
        total_wcc = 0

        lines = content.splitlines()
        line_weights, total_complexity = calculate_control_structure_complexity(lines)

        for line_number, line in enumerate(lines, start=1):
            # Calculate size (token count)
            size, tokens = calculate_size(line)

            # Skip processing lines with "using", "namespace", or "class"
            if any(keyword in line for keyword in ["using", "namespace", "class"]):
                continue

            # Skip further calculations if size is zero
            if size == 0:
                line_complexities.append({
                    'line_number': line_number,
                    'line_content': line.strip(),
                    'size': size,
                    'tokens': tokens,
                    'control_structure_complexity': 0,
                    'nesting_level': 0,
                    'inheritance_level': 0,
                    'compound_condition_weight': 0,
                    'try_catch_weight': 0,
                    'thread_weight': 0,
                    'cbo_weights': 0,
                    # 'mpc_weight': 0,
                })
                complexity_data.append([
                    line_number,
                    line.strip(),
                    size,
                    ', '.join(tokens),
                    0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue
            cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
            total_cbo_weight = sum(cbo_weights.values())

            mpc_weight = mpc_line_data.get(line_number, 0)

            nesting_level = nesting_level_dict.get(line_number, 0)
            try_catch_weight = try_catch_weight_dict.get(line_number, 0)
            print("try_catch_weight", try_catch_weight)
            thread_weight = thread_weights.get(line_number, 0)

            control_structure_complexity = line_weights.get(line_number, {"weight": 0})["weight"]

            current_inheritance = calculate_inheritance_level2(class_name)
            method_inheritance[class_name] = current_inheritance

            # Calculate weights due to compound conditions
            compound_condition_weight = calculate_compound_condition_weight(line)

            metrics = [control_structure_complexity, nesting_level, compound_condition_weight, try_catch_weight,
                       current_inheritance]
            recommendation = recommend_action(metrics)

            # Check if pattern exists in dataset; if not, add it
            if not ((dataset["control_structure_complexity"] == control_structure_complexity) &
                    (dataset["nesting_level"] == nesting_level) &
                    (dataset["compound_condition_weight"] == compound_condition_weight) &
                    (dataset["try_catch_weight"] == try_catch_weight) &
                    (dataset["current_inheritance"] == current_inheritance)).any():
                new_patterns.append({
                    "control_structure_complexity": control_structure_complexity,
                    "nesting_level": nesting_level,
                    "compound_condition_weight": compound_condition_weight,
                    "try_catch_weight": int(try_catch_weight),
                    "current_inheritance": int(current_inheritance),
                    "label": recommendation.strip('"')
                })

            # Append complexity details to line_complexities for recommendations
            line_complexities.append({
                'line_number': line_number,
                'line_content': line.strip(),
                'size': size,
                'tokens': tokens,
                'control_structure_complexity': control_structure_complexity,
                'nesting_level': nesting_level,
                'inheritance_level': current_inheritance,
                'compound_condition_weight': compound_condition_weight,
                'try_catch_weight': try_catch_weight,
                'thread_weight': thread_weight,
                'cbo_weights': total_cbo_weight,
                # 'mpc_weight': mpc_weight,
            })

            # Calculate the total complexity for this line (this could be the sum of all the metrics)
            total_complexity = (
                    size + control_structure_complexity + nesting_level + current_inheritance +
                    compound_condition_weight + try_catch_weight + thread_weight + total_cbo_weight + 0
            )

            # Update the total WCC for the file
            total_wcc += total_complexity

            print("total_wcc", total_wcc, flush=True)

            # Collect the line's metri
            complexity_data.append([
                line_number,
                line.strip(),
                size,
                ', '.join(tokens),
                control_structure_complexity,
                nesting_level,
                current_inheritance,
                compound_condition_weight,
                try_catch_weight,
                thread_weight,
                total_cbo_weight,
                # mpc_weight,
                total_complexity,
            ])
        # Calculate method complexities
        method_complexities = calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data, line_weights)

        # Update dataset and retrain model if new patterns were identified
        if new_patterns:
            new_data = pd.DataFrame(new_patterns)
            update_dataset_and_model(new_data)

        # Get AI recommendations for each line in the file
        recommendations = ai_recommend_refactoring(line_complexities)

        # Filter out "No action needed" recommendations
        filtered_recommendations = [
            rec for rec in recommendations if rec['recommendation'] != "No action needed"
        ]

        # Calculate contributing factors and plot pie chart
        complexity_factors = calculate_complexity_factors(filename, complexity_data)
        pie_chart_path = plot_complexity_pie_chart(filename, complexity_factors)

        bar_chart_paths = {}
        for method_name, method_data in method_complexities.items():
            relevant_factors = {
                "size": method_data["size"],
                "control_structure_complexity": method_data["control_structure_complexity"],
                "nesting_level": method_data["nesting_level"],
                "inheritance_level": method_data["inheritance_level"],
                "compound_condition_weight": method_data["compound_condition_weight"],
                "try_catch_weight": method_data["try_catch_weight"],
                "thread_weight": method_data["thread_weight"],
                "cbo_weights": method_data["cbo_weights"],
                # "mpc_weight": method_data["mpc_weight"]
            }

            bar_chart_path = plot_complexity_bar_chart(method_name, relevant_factors, filename)
            bar_chart_paths[method_name] = bar_chart_path
            print(f"Bar chart generated for method '{method_name}': {bar_chart_path}")


        # results[filename] = complexity_data
        results[filename] = {
            'complexity_data': complexity_data,
            'cbo': cbo_value,
            'mpc': mpc_value,
            'method_complexities': method_complexities,
            'recommendation': filtered_recommendations,
            'pie_chart_path': pie_chart_path,
            'bar_charts': bar_chart_paths,
            'total_wcc': total_wcc
        }

    return results, results3

# Function to calculate complexity factors for a file
def calculate_complexity_factors(filename, data):
    total_size = 0
    total_control_structure_complexity = 0
    total_nesting_level = 0
    total_inheritance_level = 0
    total_compound_condition_weight = 0
    total_try_catch_weight = 0
    total_thread_weight = 0
    total_cbo_weight = 0
    # total_mpc_weight = 0

    for line in data:
        total_size += line[2]
        total_control_structure_complexity += line[4]
        total_nesting_level += line[5]
        total_inheritance_level += line[6]
        total_compound_condition_weight += line[7]
        total_try_catch_weight += line[8]
        total_thread_weight += line[9]
        total_cbo_weight += line[10]
        # total_mpc_weight += line[11]

    return {
        'Size': total_size,
        'Control Structure Complexity': total_control_structure_complexity,
        'Nesting Level': total_nesting_level,
        'Inheritance Level': total_inheritance_level,
        'Compound Condition Weight': total_compound_condition_weight,
        'Try-Catch Weight': total_try_catch_weight,
        'Thread Weight': total_thread_weight,
        'CBO': total_cbo_weight,
        # 'MPC': total_mpc_weight
    }


# Set the backend to Agg
matplotlib.use("Agg")


# Function to plot complexity factors as a pie chart
def plot_complexity_pie_chart(filename, complexity_factors):
    labels = list(complexity_factors.keys())
    sizes = list(complexity_factors.values())

    # Define the colors for each segment
    colors = plt.cm.tab20.colors[:len(sizes)]  # Using tab20 colormap for variety

    # Calculate percentages for each factor for legend
    total = sum(sizes)
    labels_with_percentages = [f"{label} ({size / total * 100:.1f}%)" for label, size in zip(labels, sizes)]

    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(sizes, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'Code Complexity Contribution for {filename}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add a legend with the color scheme in the top right corner
    plt.legend(wedges, labels_with_percentages, title="Complexity Factors", loc="upper right", bbox_to_anchor=(1.6, 1))

    # Save the pie chart as an image
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{filename}_complexity_pie.png")
    plt.savefig(chart_path, bbox_inches="tight")  # Adjust layout to fit the legend
    plt.close()

    return f"{filename}_complexity_pie.png"


def plot_complexity_bar_chart(method_name, complexity_factors, filename):
    """
    Plots a bar graph to visualize complexity contributions for a specific method.
    """
    labels = list(complexity_factors.keys())
    values = list(complexity_factors.values())

    print("labels---------------------------", labels)
    print("values---------------------------", values)
    # Define colors for the bars
    colors = plt.cm.tab20.colors[:len(values)]  # Using tab20 colormap for variety

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)

    # Add value labels on top of the bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.1f}', ha='center', va='bottom')

    plt.title(f'Complexity Contributions for Method: {method_name}')
    plt.xlabel('Complexity Factors')
    plt.ylabel('Contribution Value')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Save the bar chart as an image
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{filename}_{method_name}_bar_chart.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    return f"{filename}_{method_name}_bar_chart.png"