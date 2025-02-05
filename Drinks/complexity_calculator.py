import logging
import os
import re

import javalang
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib


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

# # Example new Java code feature vector
# new_cbo_features = {
#     "class_dependencies": 9,  # Many dependencies
#     "direct_instantiations": 4,  # Excessive object creation
#     "static_method_calls": 4,  # Overuse of static methods
#     "static_variable_usage": 3,  # Many static variables
#     "interface_implementations": 2,  # Implements many interfaces
#     "constructor_injections": 2,  # Too many constructor dependencies
#     "setter_injections": 2,  # Too many setter dependencies
#     "global_variable_references": 2  # Accesses global state
# }
#
# # Convert to DataFrame
# X_new = pd.DataFrame([new_cbo_features])
#
# # Predict CBO issue
# prediction = rf_model.predict(X_new)[0]
#
# # Generate Recommendations
# recommendations = []
# if X_new["class_dependencies"][0] > 5:
#     recommendations.append("⚠️ Reduce class dependencies by using interfaces instead of direct implementations.")
#
# print("\n🔹 **Prediction:**", "High CBO (Issue)" if prediction == 1 else "Low CBO (Good)")
# print("\n🔹 **Recommended Fixes:**")
# for rec in recommendations:
#     print(f"➡️ {rec}")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Global flag to track whether the class declaration has ended
class_declaration_ended = False

inheritance_depth = {}


# Function to remove comments (both single-line and multi-line)
def remove_comments(line):
    # Remove single-line comments (//)
    line = re.sub(r'//.*', '', line)
    # Remove multi-line comments (/* ... */)
    line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
    return line


# Function to calculate the size of a single line (token count)
def calculate_size(line):
    global class_declaration_ended

    line = remove_comments(line)

    # Check for class declaration and skip until it's completed
    if not class_declaration_ended:
        # Detect the class declaration pattern
        if re.search(r'\bclass\b\s+\w+', line):
            class_declaration_ended = True
            return 0, []  # Do not count tokens until after class declaration
        return 0, []  # Still in class declaration, so ignore

    # Remove annotations like @Override
    line = re.sub(r'@\w+', '', line)
    # Exclude access modifiers and function parameters
    line = re.sub(r'\b(public|private|protected|default|static|else)\b', '', line)  # Ignore access modifiers
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
"""
def calculate_control_structure_complexity(lines):
    
    total_weight = 0
    line_weights = {}

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        # Sequential statement (Weight = 0)
        # if stripped_line.startswith("int") or stripped_line.startswith("double") or "=" in stripped_line:
        #     weight = 0

        # Branching statement (if, else if, else) (Weight = 1)
        if stripped_line.startswith("if") or stripped_line.startswith("}else if"):
            weight = 1

        # Iterative statement (for, while, do-while) (Weight = 2)
        elif stripped_line.startswith("for") or stripped_line.startswith("while") or stripped_line.startswith("do"):
            weight = 2

        elif stripped_line.startswith("switch"):
            case_count = 0

            for subsequent_line in lines[line_number:]:
                subsequent_line = subsequent_line.strip()
                if subsequent_line.startswith("case") or subsequent_line.startswith("default"):
                    case_count += 1
                if subsequent_line == "}":
                    break
            weight = case_count

        else:
            weight = 0  # Default weight for lines not matching any category

        line_weights[line_number] = {
            "line_content": stripped_line,
            "weight": weight
        }
        total_weight += weight

    return line_weights, total_weight
"""

def calculate_control_structure_complexity(java_code):
    """
    Calculate the complexity weight for a given Java code.

    Parameters:
    - java_code (str or list of str): Java code as a string or a list of lines.

    Returns:
    - dict: A dictionary with line numbers as keys and assigned weights as values.
    """
    if isinstance(java_code, list):
        java_code = "\n".join(java_code)  # Convert list of lines to a string

    total_weight = 0
    line_weights = {}

    # Parse Java code into tokens
    tokens = list(javalang.tokenizer.tokenize(java_code))
    lines = java_code.split("\n")

    # Convert tokens into AST for structured parsing
    tree = javalang.parse.parse(java_code)

    for path, node in tree:
        position = getattr(node, "position", None)
        if position is None:
            continue  # Skip nodes without a position

        line_number = position.line  # Extract integer line number
        weight = 0  # Default weight

        # Control Structure Weights
        if isinstance(node, javalang.tree.IfStatement):
            weight = 1  # if / else if
        elif isinstance(node, (javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)):
            weight = 2  # for / while / do-while
        elif isinstance(node, javalang.tree.SwitchStatement):
            # Count the number of case/default statements
            case_count = sum(1 for sub in node.cases if isinstance(sub, javalang.tree.SwitchStatementCase))
            weight = case_count  # Switch complexity based on number of cases

        # Assign weight and update total complexity
        if line_number in line_weights:
            line_weights[line_number]["weight"] += weight  # Accumulate weight for lines with multiple structures
        else:
            line_weights[line_number] = {
                "line_content": lines[line_number - 1].strip(),
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
    control_structure_stack = []
    nesting_levels = []
    line_weights = {}

    # Regular expressions for detecting control structures and braces
    control_regex = re.compile(r'\b(if|else if|else|for|while|do|switch|case|default)\b')
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

# Function to calculate inheritance level for a single line (Wi)
def calculate_inheritance_level(line, current_inheritance):
    # Check if the line defines a class extending another class
    if re.search(r'class\s+\w+\s+extends\s+\w+', line):
        # Increment inheritance level if class extends another class
        current_inheritance += 1
    elif re.search(r'class\s+\w+', line):
        # If a class is defined without extending, assign a default inheritance level of 1
        current_inheritance = 1  # or keep it unchanged if you want to count only extending classes

    return current_inheritance


def calculate_compound_condition_weight(line):
    """
    Calculate the code complexity for compound conditions in a line.
    Formula: Complexity = Base Weight + (Number of Subconditions - 1)
    """
    # Initialize the complexity weight for the line
    complexity = 0

    # Define patterns to identify compound and simple conditions
    compound_condition_pattern = r'\(.*?(?:&&|\|\|).*?\)'
    logical_operator_pattern = r'&&|\|\|'

    # Find all compound conditions in the line
    compound_conditions = re.findall(compound_condition_pattern, line)

    for condition in compound_conditions:
        # Count the number of logical operators (&&, ||) in the condition
        logical_operators = len(re.findall(logical_operator_pattern, condition))
        # Number of subconditions = logical operators + 1
        subconditions = logical_operators + 1
        # Calculate complexity for this condition
        condition_complexity = 1 + (subconditions - 1)
        # Add to total complexity
        complexity += condition_complexity

    # Check for simple conditions if no compound conditions are found
    if not compound_conditions:
        simple_condition_pattern = r'\b(if|while|for|return)\s*\(.*?\)'
        simple_conditions = re.findall(simple_condition_pattern, line)
        if simple_conditions:
            # Simple conditions have a base weight of 1
            complexity += 1

    return complexity

def calculate_try_catch_weight(java_code):
    """
    Calculates the weight of nesting levels specifically for try-catch-finally blocks in Java code.
    - Increment nesting level for try.
    - Assign weights line by line for catch and finally based on nesting level.
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

    # Weights for catch based on nesting levels
    catch_weights = {1: 1, 2: 3, 3: 4, 4: 5}  # Assign weights as in the example
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
                # Increment nesting level for try
                current_nesting += 1

            elif control_type == 'catch':
                # Assign weight for catch based on current nesting level
                weight = catch_weights.get(current_nesting, 1)  # Use nesting level to determine weight
                line_weights[line_no] = weight

            elif control_type == 'finally':
                # Assign fixed weight for finally
                line_weights[line_no] = finally_weight

        # Append the current line and its weight
        nesting_levels.append((line_no, stripped_line, current_nesting, line_weights.get(line_no, 0)))

        # Adjust nesting level for closing braces
        if stripped_line.endswith('}') and 'catch' not in stripped_line:
            current_nesting = max(0, current_nesting - 1)

    return nesting_levels, line_weights

def calculate_thread_weight(java_code):
    complexity = {}

    lines = java_code.splitlines()
    synchronized_stack = []
    block_start_line = None
    last_thread_creation_line = None
    lock_order = []

    for line_no, line in enumerate(lines, start=1):
        score = 0
        recommendations = []
        line = line.strip()

        # Check for thread creation
        if re.search(r'new\s+Thread\b', line):
            if last_thread_creation_line is not None and line_no == last_thread_creation_line + 1:
                score += 2  # Regular thread creation (not nested)
                recommendations.append({
                    "line_number": line_no,
                    "line_content": line,
                    "recommendation": "Avoid creating threads consecutively; consider using a thread pool instead."
                })
            else:
                score += 2
            last_thread_creation_line = line_no  # Update the last thread creation line

        # Check for synchronized block
        lock_match = re.search(r'synchronized\s*\((.*?)\)', line)
        if lock_match:
            lock_variable = lock_match.group(1)  # Extract the lock variable
            if block_start_line is None:
                block_start_line = line_no  # Start of synchronized block
            if synchronized_stack:
                # Detect potential deadlocks due to nested synchronization
                if lock_variable in lock_order:
                    score += 8
                    recommendations.append({
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Avoid re-locking the same lock variable within nested synchronized blocks to prevent deadlocks."
                    })
                elif lock_order and lock_variable != lock_order[-1]:
                    score += 8  # Inconsistent lock order
                    recommendations.append({
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Maintain a consistent locking order to avoid deadlocks."
                    })
                else:
                    score += 6  # Nested synchronization
                    recommendations.append({
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Avoid nested synchronized blocks to reduce the risk of deadlocks."
                    })
            else:
                score += 4  # Basic synchronization block
            synchronized_stack.append(lock_variable)  # Push the lock variable to the stack
            lock_order.append(lock_variable)  # Track lock order

        # Check for end of block
        if line == "}":
            if synchronized_stack:
                start_lock = synchronized_stack.pop()  # Pop the stack
                if block_start_line and (line_no - block_start_line) > 5:  # Broad synchronization block
                    score += 5
                    recommendations.append({
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Refactor to reduce the scope of synchronized blocks for better concurrency."
                    })
                if not synchronized_stack:  # Reset start line if stack is empty
                    block_start_line = None
                if start_lock in lock_order:
                    lock_order.remove(start_lock)  # Maintain lock order consistency

        # Check for method-level synchronization
        if re.search(r'public\s+synchronized\b', line):
            score += 8  # Higher weight for broad synchronization scope
            recommendations.append({
                "line_number": line_no,
                "line_content": line,
                "recommendation": "Avoid method-level synchronization; prefer fine-grained synchronization."
            })

            # Check for synchronized block inside method-level synchronization
            if re.search(r'synchronized\s*\(', line):
                score += 6  # Nested synchronized block inside a synchronized method
                recommendations.append({
                    "line_number": line_no,
                    "line_content": line,
                    "recommendation": "Refactor to avoid nested synchronized blocks within synchronized methods."
                })

        # Store the score and reasons for the current line
        if score > 0:
            complexity[line_no] = {"score": score, "recommendations": recommendations}

    return complexity

def extract_class_references(java_code):
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'
    setter_injection_pattern = r'(public|private)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'
    static_usage_pattern = r'([A-Z][\w]*)\.\w+\s*\('  # Static method call: e.g., Utility.method()
    static_variable_pattern = r'([A-Z][\w]*)\.\w+'
    field_declaration_pattern = r'([A-Z][\w]*)\s+\w+\s*;'

    # Dictionary to store class references per class
    class_references = {}

    # Split code into lines and process each line
    lines = java_code.splitlines()
    recommendations = []
    declared_fields = set()  # Fixed: Use a set instead of a list

    current_class = None
    for line_no, line in enumerate(lines, start=1):
        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            # Ensure the current class is in the dictionary
            if current_class not in class_references:
                class_references[current_class] = {}

        excluded_classes = [
            'System', 'String', 'Integer', 'Float', 'Double', 'Boolean', 'Character',
            'Byte', 'Short', 'Long', 'Math', 'Object', 'Thread', 'Runtime', 'Optional'
        ]

        if current_class:
            # Find class instantiations
            instantiations = re.findall(instantiation_pattern, line)
            for instantiated_class in instantiations:
                if instantiated_class in class_references[current_class]:
                    class_references[current_class][instantiated_class] += 3
                else:
                    class_references[current_class][instantiated_class] = 3
                # recommendations.append({
                #     "line_number": line_no,
                #     "line_content": line,
                #     "recommendation": f"Consider using dependency injection instead of directly instantiating {instantiated_class}."
                # })

            field_declaration_match = re.search(field_declaration_pattern, line)
            if field_declaration_match:
                field_class = field_declaration_match.group(1)
                declared_fields.add(field_class)  # Fixed: Now it works since declared_fields is a set
                class_references[current_class][field_class] = class_references[current_class].get(field_class, 0) + 1
                # recommendations.append({
                #     "line_number": line_no,
                #     "line_content": line.strip(),
                #     "recommendation": f"Field declaration for {field_class} detected. Consider constructor injection."
                # })

            # Detect static method usage
            static_methods = re.findall(static_usage_pattern, line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class, 0) + 2
                    # recommendations.append({
                    #     "line_number": line_no,
                    #     "line_content": line.strip(),
                    #     "recommendation": f"Static method from {static_class} detected. Consider avoiding static methods for better modularity."
                    # })

            # Detect static variable usage
            static_variables = re.findall(static_variable_pattern, line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class, 0) + 1
                    # recommendations.append({
                    #     "line_number": line_no,
                    #     "line_content": line.strip(),
                    #     "recommendation": f"Static variable from {static_class} detected. Avoid over-reliance on static variables."
                    # })

            constructor_matches = re.findall(constructor_pattern, line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    class_references[current_class][param_class] = class_references[current_class].get(param_class, 0) + 1
                    # recommendations.append({
                    #     "line_number": line_no,
                    #     "line_content": line,
                    #     "recommendation": f"Constructor parameter {param_class} detected. Consider constructor injection."
                    # })

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, line)
            for setter_class in setter_matches:
                class_references[current_class][setter_class] = 1
                # recommendations.append({
                #     "line_number": line_no,
                #     "line_content": line,
                #     "recommendation": f"Setter injection for {setter_class} detected. Consider using constructor injection for mandatory dependencies."
                # })

    return class_references, recommendations

# Function to calculate CBO
def calculate_cbo(class_references):
    cbo_results = {}
    # recommendations = []

    # Loop through each class's references and count them
    for class_name, references in class_references.items():
        cbo_results[class_name] = sum(references.values())

    return cbo_results

def extract_class_references_with_lines(java_code):
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    field_declaration_pattern = r'([A-Z][\w]*)\s+\w+\s*;'
    static_usage_pattern = r'([A-Z][\w]*)\.\w+\s*\('  # Static method call
    static_variable_pattern = r'([A-Z][\w]*)\.\w+'
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'  # Constructor parameters (dependency injection)
    setter_injection_pattern = r'(public|private)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'  # Setter injection

    excluded_classes = [
        'System', 'String', 'Integer', 'Float', 'Double', 'Boolean', 'Character',
        'Byte', 'Short', 'Long', 'Math', 'Object', 'Thread', 'Runtime', 'Optional'
    ]

    # Dictionary to store class   references per line
    line_references = []

    # Split code into lines and process each line
    lines = java_code.splitlines()
    current_class = None
    declared_fields = set()

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        line_data = {"line": line_no, "code": stripped_line, "weights": {}}

        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', stripped_line)
        if class_declaration:
            current_class = class_declaration.group(1)

        if current_class:
            # Initialize weights for the current line
            weights = {}

            # Detect field declarations for aggregation coupling
            # Step 1: Detect field declarations (possible aggregation)
            field_declaration_match = re.search(field_declaration_pattern, stripped_line)
            if field_declaration_match:
                field_class = field_declaration_match.group(1)
                declared_fields.add(field_class)  # Track fields for later use
                weights[field_class] = 1  # Default to aggregation (loose coupling)

            # Step 2: Detect composition (field initialization using `new`)
            # Check if the line initializes a previously declared field
            for field_class in declared_fields:
                if f"= new {field_class}" in stripped_line:
                    weights[field_class] = 3  # Overwrite to composition (tight coupling)

            # Find class instantiations
            instantiations = re.findall(instantiation_pattern, stripped_line)
            for instantiated_class in instantiations:
                weights[instantiated_class] = 3

            # Detect constructor injection (dependency injection)
            constructor_matches = re.findall(constructor_pattern, stripped_line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    weights[param_class] = 1  # Loose coupling for dependency injection

            # Detect setter injection (dependency injection)
            setter_matches = re.findall(setter_injection_pattern, stripped_line)
            for setter_class in setter_matches:
                weights[setter_class] = 1  # Loose coupling for setter injection
            # Static method usage
            static_methods = re.findall(static_usage_pattern, stripped_line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    weights[static_class] = weights.get(static_class, 0) + 2  # Medium coupling for static methods

            # Detect static variable usage
            static_variables = re.findall(static_variable_pattern, stripped_line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    weights[static_class] = weights.get(static_class, 0) + 1

            # Add weights to the current line
            line_data["weights"] = weights

        # Add the processed line data to the list
        line_references.append(line_data)

    logging.info("Finished extracting class references line by line.")
    logging.info(line_references)

    return line_references


# Function to calculate CBO line by line
def calculate_cbo_line_by_line(line_references):
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

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Function to extract method calls (message passing) and classify their complexity
def extract_message_passing(java_code):
    # Patterns to match different types of message passing
    simple_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\('  # Simple method call
    moderate_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\(.*new\s+\w+\s*\('  # Moderate with callback or new instance
    async_message_pattern = r'CompletableFuture\s*\.\s*([a-zA-Z_]\w+)'  # Asynchronous message pattern
    callback_pattern = r'CompletableFuture\s*\.\s*(thenApply|thenAccept|thenRun|whenComplete)\s*\(\s*.*\s*->\s*\{'  # Callback methods
    exceptional_pattern = r'CompletableFuture\s*\.\s*exceptionally\s*\(\s*.*\s*->\s*\{'  # Exception handling in async

    # Methods to ignore, such as println
    ignore_methods = {"println", "print", "printf"}

    # Dictionary to store the message passing weights per class
    message_passing = {}

    # Split code into lines and process each line
    lines = java_code.splitlines()

    current_class = None
    for line in lines:
        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            # Ensure the current class is in the dictionary
            if current_class not in message_passing:
                message_passing[current_class] = {}
                logging.info(f'Detected class: {current_class}')  # Log class detection

        if current_class:
            # Find simple message passing
            simple_messages = re.findall(simple_message_pattern, line)
            for method_call in simple_messages:
                method_name = method_call[1]
                # Skip ignored methods
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,
                                                                                                     0) + 1  # Weight = 1
                    logging.info(
                        f'Found simple message passing in {current_class}: {method_name} (Weight = 1)')  # Log simple message passing

            # Find moderate message passing (e.g., with callback)
            moderate_messages = re.findall(moderate_message_pattern, line)
            for method_call in moderate_messages:
                method_name = method_call[1]
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,
                                                                                                     0) + 2  # Weight = 2
                    logging.info(
                        f'Found moderate message passing in {current_class}: {method_name} (Weight = 2)')  # Log moderate message passing

            # Find complex asynchronous message passing
            async_messages = re.findall(async_message_pattern, line)
            for method_call in async_messages:
                method_name = method_call
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,
                                                                                                     0) + 3  # Weight = 3
                    logging.info(
                        f'Found complex message passing in {current_class}: {method_name} (Weight = 3)')  # Log complex message passing
            # Find callback handling in async methods
            callback_messages = re.findall(callback_pattern, line)
            for method_name in callback_messages:
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,
                                                                                                     0) + 3  # Weight = 3
                    logging.info(
                        f'Found callback in async message passing in {current_class}: {method_name} (Weight = 3)')

            # Find exceptional handling in async methods
            exceptional_messages = re.findall(exceptional_pattern, line)
            for method_name in exceptional_messages:
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,
                                                                                                     0) + 3  # Weight = 3
                    logging.info(
                        f'Found exceptional handling in async message passing in {current_class}: {method_name} (Weight = 3)')

    return message_passing


# Function to calculate MPC
def calculate_mpc(message_passing):
    mpc_results = {}

    for class_name, messages in message_passing.items():
        # Sum the weights of message passing interactions
        total_weight = sum(messages.values())
        mpc_results[class_name] = total_weight
        logging.info(f'MPC for {class_name}: {total_weight}')  # Log MPC calculation

    return mpc_results


def extract_message_passing_with_lines(java_code):
    # Patterns to match different types of message passing
    simple_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\('  # Simple method call
    moderate_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\(.*new\s+\w+\s*\('  # Moderate with callback or new instance
    async_message_pattern = r'CompletableFuture\s*\.\s*([a-zA-Z_]\w+)'  # Asynchronous message pattern
    callback_pattern = r'CompletableFuture\s*\.\s*(thenApply|thenAccept|thenRun|whenComplete)\s*\(\s*.*\s*->\s*\{'  # Callback methods
    exceptional_pattern = r'CompletableFuture\s*\.\s*exceptionally\s*\(\s*.*\s*->\s*\{'  # Exception handling in async

    # Methods to ignore, such as println
    ignore_methods = {"println", "print", "printf"}

    # Dictionary to store the message passing weights per line
    message_passing_lines = {}

    # Split code into lines and process each line
    lines = java_code.splitlines()

    current_class = None
    for line_number, line in enumerate(lines, start=1):
        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)

        if current_class:
            if line_number not in message_passing_lines:
                message_passing_lines[line_number] = 0

            # Find simple message passing
            simple_messages = re.findall(simple_message_pattern, line)
            for method_call in simple_messages:
                method_name = method_call[1]
                if method_name not in ignore_methods:
                    message_passing_lines[line_number] += 1  # Weight = 1
                    logging.info(
                        f'Found simple message passing at line {line_number} in {current_class}: {method_name} (Weight = 1)'
                    )

            # Find moderate message passing (e.g., with callback)
            moderate_messages = re.findall(moderate_message_pattern, line)
            for method_call in moderate_messages:
                method_name = method_call[1]
                if method_name not in ignore_methods:
                    message_passing_lines[line_number] += 2  # Weight = 2
                    logging.info(
                        f'Found moderate message passing at line {line_number} in {current_class}: {method_name} (Weight = 2)'
                    )

            # Find complex asynchronous message passing
            async_messages = re.findall(async_message_pattern, line)
            for method_call in async_messages:
                method_name = method_call
                if method_name not in ignore_methods:
                    message_passing_lines[line_number] += 3  # Weight = 3
                    logging.info(
                        f'Found complex message passing at line {line_number} in {current_class}: {method_name} (Weight = 3)'
                    )

            # Find callback handling in async methods
            callback_messages = re.findall(callback_pattern, line)
            for method_name in callback_messages:
                if method_name not in ignore_methods:
                    message_passing_lines[line_number] += 3  # Weight = 3
                    logging.info(
                        f'Found callback in async message passing at line {line_number} in {current_class}: {method_name} (Weight = 3)'
                    )

            # Find exceptional handling in async methods
            exceptional_messages = re.findall(exceptional_pattern, line)
            for method_name in exceptional_messages:
                if method_name not in ignore_methods:
                    message_passing_lines[line_number] += 3  # Weight = 3
                    logging.info(
                        f'Found exceptional handling in async message passing at line {line_number} in {current_class}: {method_name} (Weight = 3)'
                    )

    return message_passing_lines


def calculate_mpc_line_by_line(message_passing_lines):
    mpc_line_results = {}

    # Aggregate MPC values for each line
    for line_number, weight in message_passing_lines.items():
        mpc_line_results[line_number] = weight
        logging.info(f'MPC at line {line_number}: {weight}')  # Log MPC calculation per line

    return mpc_line_results


# Example Integration in File Processing
def calculate_mpc_for_java_code(file_contents):
    # Dictionary to store the results for each file
    mpc_results = {}

    # Process each file
    for filename, java_code in file_contents.items():
        # Extract message passing weights with line numbers for the current file
        message_passing_lines = extract_message_passing_with_lines(java_code)

        # Calculate MPC values line by line
        mpc_results[filename] = calculate_mpc_line_by_line(message_passing_lines)

    return mpc_results

# Function to track inheritance depth across multiple files
def track_inheritance_depth_across_files(file_contents):
    global inheritance_depth  # Reference the global inheritance_depth
    # First pass: record all class declarations (including those with extends)
    for filename, content in file_contents.items():
        lines = content.splitlines()
        for line in lines:
            line = remove_comments(line)
            # Detect class declaration with or without inheritance
            match_inheritance = re.search(r'class\s+(\w+)\s+extends\s+(\w+)', line)
            match_base_class = re.search(r'class\s+(\w+)', line)

            if match_inheritance:
                class_name = match_inheritance.group(1)
                base_class_name = match_inheritance.group(2)

                # Record the inheritance relationship
                inheritance_depth[class_name] = inheritance_depth.get(base_class_name, 1) + 1  # Derived class depth
            elif match_base_class and not re.search(r'extends', line):
                class_name = match_base_class.group(1)
                if class_name not in inheritance_depth:
                    inheritance_depth[class_name] = 1  # Base class gets depth 1

def calculate_inheritance_level2(class_name):
    return inheritance_depth.get(class_name, 1)

# Load or initialize dataset
data_file = "media/synthetic_training_data_1000.csv"
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


def calculate_code_complexity_line_by_line(code):
    lines = code.splitlines()
    current_nesting = 0
    control_structure_stack = []
    current_inheritance = 0
    in_control_structure = False

    method_complexities = calculate_code_complexity_by_method(code)
    line_complexities = []

    for i, line in enumerate(lines, start=1):
        # Calculate complexity for the current line
        size, tokens = calculate_size(line)
        # if size == 0:
        #     continue
        wc = calculate_control_structure_complexity(line)
        current_nesting, in_control_structure, control_structure_stack, wn = calculate_nesting_level(
            line, current_nesting, in_control_structure, control_structure_stack
        )
        current_inheritance = calculate_inheritance_level(line, current_inheritance)
        compound_condition_weight = calculate_compound_condition_weight(line)
        try_catch_weight = calculate_try_catch_weight(line, current_nesting)
        thread_weight = calculate_thread_weight(line)

        line_complexities.append({
            'line_number': i,
            'line_content': line.strip(),
            'size': size,
            'tokens': tokens,
            'control_structure_complexity': wc,
            'nesting_level': current_nesting,
            'inheritance_level': current_inheritance,
            'compound_condition_weight': compound_condition_weight,
            'try_catch_weight': try_catch_weight,
            'thread_weight': thread_weight,
        })

    # Get AI recommendations for each line in the file
    # recommendations = ai_recommend_refactoring(line_complexities)
    # Print the recommendations
    # for recommendation in recommendations:
    #     print(f"Line {recommendation['line_number']}: {recommendation['line_content']}")
    #     print(f"Recommendation: {recommendation['recommendation']}\n")

    # Extract class references and calculate CBO after processing lines
    class_references = extract_class_references(code)
    cbo_results = calculate_cbo(class_references)

    print(cbo_results)

    # Extract message passing data
    message_passing = extract_message_passing(code)

    # Calculate MPC
    mpc_results = calculate_mpc(message_passing)

    # Output the results
    print("Message Passing Coupling (MPC) Results:")
    for class_name, mpc_value in mpc_results.items():
        print(f"{class_name}: {mpc_value}")

    return {
        'line_complexities': line_complexities,
        'cbo': cbo_results
    }

    # return line_complexities

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

# Function to extract CBO features from Java code
def extract_cbo_features(java_code):
    import javalang

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

    try:
        tree = javalang.parse.parse(java_code)

        for path, node in tree:
            if isinstance(node, javalang.tree.Import):
                cbo_features["class_dependencies"].add(node.path)

            if isinstance(node, javalang.tree.ClassCreator):
                cbo_features["direct_instantiations"] += 1

            if isinstance(node, javalang.tree.MethodDeclaration):
                if 'static' in node.modifiers:
                    cbo_features["static_method_calls"] += 1

            if isinstance(node, javalang.tree.FieldDeclaration):
                if "static" in node.modifiers:
                    cbo_features["static_variable_usage"] += 1

            if isinstance(node, javalang.tree.ClassDeclaration):
                if node.implements:
                    cbo_features["interface_implementations"] += len(node.implements)

            if isinstance(node, javalang.tree.ConstructorDeclaration):
                cbo_features["constructor_injections"] += len(node.parameters)

            if isinstance(node, javalang.tree.MethodDeclaration):
                if "set" in node.name.lower():
                    cbo_features["setter_injections"] += 1

            if isinstance(node, javalang.tree.FieldDeclaration):
                if 'static' in node.modifiers and 'final' not in node.modifiers:
                    cbo_features["global_variable_references"] += 1

    except Exception as e:
        print(f"Error parsing Java code: {e}")

    cbo_features["class_dependencies"] = len(cbo_features["class_dependencies"])
    return cbo_features

def calculate_code_complexity_multiple_files(file_contents):
    results = {}
    results3 = {}
    new_patterns = []

    # Step 1: Track inheritance across all files
    track_inheritance_depth_across_files(file_contents)

    result1 = calculate_code_complexity_multiple(file_contents)

    mpc_results = calculate_mpc_for_java_code(file_contents)

    # Iterate through each file content
    for filename, content in file_contents.items():
        class_name = filename.split('.')[0]

        # Split content into lines
        lines = content.splitlines()
        complexity_data = []
        complexity_data2 = []

        # Extract features from Java code
        new_cbo_features = extract_cbo_features(content)
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

        # Extract CBO results for each line
        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Extract class references and message passing for MPC and CBO
        class_references, cbo_recommendations = extract_class_references(content)
        message_passing = extract_message_passing(content)
        nesting_levels = calculate_nesting_level(content)
        nesting_level_dict = {line[0]: line[2] for line in nesting_levels}

        try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)

        thread_weights = calculate_thread_weight(content)

        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare line_complexities for recommendation engine
        line_complexities = []

        method_inheritance = {}

        # Initialize total WCC value for the file
        total_wcc = 0
        # line_weights, total_control_complexity = calculate_control_structure_complexity(content)
        lines = content.splitlines()
        line_weights, total_complexity = calculate_control_structure_complexity(lines)
        print("line_weights, total_complexity<<<<<<<<<<<<<<<<<<<<<<", line_weights, total_complexity)

        for line_number, line in enumerate(lines, start=1):
            # Calculate size (token count)
            size, tokens = calculate_size(line)

            # Skip processing lines with "using", "namespace", or "class"
            if any(keyword in line for keyword in ["class"]):
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
            thread_weight = thread_weights.get(line_number, {"score": 0, "reasons": []})

            complexity_data2.append({
                "line_number": line_number,
                "line_content": line.strip(),
                "thread_weight": thread_weight.get("score", 0),
                "recommendations": thread_weight.get("recommendations", [])
            })

            current_inheritance = calculate_inheritance_level2(class_name)
            method_inheritance[class_name] = current_inheritance

            # Calculate weights due to compound conditions
            compound_condition_weight = calculate_compound_condition_weight(line)

            # control_structure_complexity = line_weights[line_number]["weight"]
            control_structure_complexity = line_weights.get(line_number, {"weight": 0})["weight"]

            metrics = [control_structure_complexity, nesting_level, compound_condition_weight, try_catch_weight, current_inheritance]
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
                    "try_catch_weight":int(try_catch_weight),
                    "current_inheritance":int(current_inheritance),
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
                'thread_weight': thread_weight['score'],
                'cbo_weights': total_cbo_weight,
                # 'mpc_weight': mpc_weight,
            })

            # Calculate the total complexity for this line (this could be the sum of all the metrics)
            total_complexity = (
                    size + control_structure_complexity + nesting_level + current_inheritance +
                    compound_condition_weight + try_catch_weight + thread_weight['score'] + total_cbo_weight + 0
            )

            # Update the total WCC for the file
            total_wcc += total_complexity

            # Collect the line's metrics
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
                thread_weight['score'],
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

        for item3 in complexity_data2:
            if isinstance(item3, dict) and "recommendations" in item3:
                filtered_recommendations.extend(item3["recommendations"])

        filtered_recommendations.extend(cbo_recommendations)

        # Calculate contributing factors and plot pie chart
        complexity_factors = calculate_complexity_factors(filename, complexity_data)
        pie_chart_path = plot_complexity_pie_chart(filename, complexity_factors)

        # Generate bar chart for each method
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


# Refined method pattern for accurate method detection
method_pattern = re.compile(
    r'^\s*(public|private|protected)?\s*(static\s+)?'  # Access modifiers and 'static' keyword
    r'(\w+\s+)?'  # Optional return type (including void)
    r'(\w+)\s*\([^)]*\)\s*\{'  # Method name and parameter list
)

# Keywords to ignore to prevent detecting control structures as methods
control_keywords = {'if', 'for', 'while', 'switch', 'catch'}

def calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data, line_weights):
    """
    Calculate complexity metrics for each method in a given class.

    :param content: The file content (Java code) as a string.
    :param method_inheritance: Dictionary containing inheritance levels.
    :param class_name: The name of the class in the file.
    :param cbo_line_data: Coupling Between Object (CBO) data per line.
    :param mpc_line_data: Message Passing Coupling (MPC) data per line.
    :param line_weights: Control structure weights per line.
    :return: A dictionary with complexity data for each method.
    """
    methods = {}
    method_name = None
    method_lines = []
    brace_counter = 0  # Tracks the number of open braces

    # Precompute additional metrics
    nesting_levels = calculate_nesting_level(content)
    nesting_level_dict = {line[0]: line[2] for line in nesting_levels}
    try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)
    thread_weights = calculate_thread_weight(content)

    # Regular expression for method signature detection
    method_signature_regex = r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(.*\)\s*{'

    # List of keywords to exclude from being treated as methods
    excluded_keywords = {"if", "for", "while", "switch", "Thread", "run"}

    # Iterate through the file content line by line
    for line_number, line in enumerate(content.splitlines(), start=1):
        # Check for method signature
        match = re.search(method_signature_regex, line.strip())
        if match and brace_counter == 0:  # Start of a new method
            candidate_method_name = match.group(1)
            if candidate_method_name in excluded_keywords:
                continue  # Ignore if the match is a control structure

            # If we were processing a previous method, finalize its metrics
            if method_name and method_lines:
                methods[method_name] = calculate_complexity_for_method(
                    method_lines, method_inheritance, class_name, nesting_level_dict,
                    try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights
                )

            # Start tracking the new method
            method_name = candidate_method_name
            method_lines = [(line_number, line)]
            brace_counter = line.count("{") - line.count("}")
        elif method_name:
            # Add the current line to the method
            method_lines.append((line_number, line))

            # Update the brace counter
            brace_counter += line.count("{") - line.count("}")

            # If braces are balanced, the method ends
            if brace_counter == 0:
                methods[method_name] = calculate_complexity_for_method(
                    method_lines, method_inheritance, class_name, nesting_level_dict,
                    try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights
                )
                method_name = None
                method_lines = []

    # Finalize the last detected method
    if method_name and method_lines:
        methods[method_name] = calculate_complexity_for_method(
            method_lines, method_inheritance, class_name, nesting_level_dict,
            try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights
        )

    return methods

# Helper function to calculate complexity for a method based on lines of code
def calculate_complexity_for_method(lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data, line_weights):
    """
    Calculate the complexity of a method based on its lines of code.

    :param lines: List of tuples (line_number, line_content) belonging to a method.
    :param method_inheritance: Dictionary containing inheritance levels.
    :param class_name: The name of the class to which this method belongs.
    :param nesting_level_dict: Dictionary mapping line numbers to nesting levels.
    :param try_catch_weight_dict: Dictionary mapping line numbers to try-catch weights.
    :param thread_weights: Dictionary mapping line numbers to thread weights.
    :param cbo_line_data: Dictionary containing CBO weights for each line.
    :param mpc_line_data: Dictionary containing MPC weights for each line.
    :param line_weights: Dictionary containing control structure weights for each line.
    :return: Dictionary of complexity metrics for the method.
    """
    size = 0
    control_structure_complexity = 0
    nesting_level = 0
    compound_condition_weight = 0
    current_inheritance_sum = 0
    total_nesting = 0
    total_try_catch_weight = 0
    total_thread_weight = 0
    total_cbo = 0
    total_mpc = 0

    # Get the inheritance level for the class
    inheritance_level = method_inheritance.get(class_name, 0)

    # Analyze each line within the method
    for line_number, line_content in lines:  # Unpack tuple into line_number and line_content
        # Remove comments and calculate size (token count) for this line
        line_content = remove_comments(line_content)  # Ensure line_content is used, not tuple
        line_size, tokens = calculate_size(line_content)
        size += line_size

        if line_size == 0:
            continue  # Skip this line if size is 0

        total_inheritance = inheritance_level

        nesting_level = nesting_level_dict.get(line_number, 0)
        total_nesting += nesting_level

        try_catch_weight = try_catch_weight_dict.get(line_number, 0)
        total_try_catch_weight += try_catch_weight

        thread_weight = thread_weights.get(line_number, {"score": 0}).get("score", 0)
        total_thread_weight += thread_weight

        current_inheritance_sum += total_inheritance

        control_structure_complexity += line_weights.get(line_number, {}).get("weight", 0)

        cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
        total_cbo_weight = sum(cbo_weights.values())
        total_cbo += total_cbo_weight

        # mpc_weight = mpc_line_data.get(line_number, 0)
        # total_mpc += mpc_weight

        # Compound condition weight
        compound_condition_weight += calculate_compound_condition_weight(line_content)

    # Sum up the complexity metrics for this method
    total_complexity = (
        size + control_structure_complexity + total_nesting + current_inheritance_sum +
        compound_condition_weight + total_try_catch_weight + total_thread_weight + total_cbo + 0
    )

    print("Size ******************************************************8", size)
    print("current_inheritance_sum ******************************************************8", current_inheritance_sum)
    print("total_complexity&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7", total_complexity)

    return {
        "size": size,
        "control_structure_complexity": control_structure_complexity,
        "nesting_level": total_nesting,
        "inheritance_level": current_inheritance_sum,
        "compound_condition_weight": compound_condition_weight,
        "try_catch_weight": total_try_catch_weight,
        "thread_weight": total_thread_weight,
        "cbo_weights": total_cbo,
        # "mpc_weight": total_mpc,
        "total_complexity": total_complexity
    }

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
