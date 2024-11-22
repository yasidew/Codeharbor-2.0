import logging
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
def calculate_size(line):
    global class_declaration_ended
    # Remove comments from the line
    line = remove_comments(line)

    # Check for class declaration and skip until it's completed
    if not class_declaration_ended:
        # Detect the class declaration pattern
        if re.search(r'\bclass\b\s+\w+', line):
            class_declaration_ended = True
            return 0, []  # Do not count tokens until after class declaration
        return 0, []  # Still in class declaration, so ignore

    # Print to debug line content after removing comments
    print("Line after removing comments:", line)

    # Exclude access modifiers and certain keywords that do not affect complexity
    line = re.sub(r'\b(public|private|protected|static|final|else|return|try|catch)\b', '', line)
    line = re.sub(r'\(\s*\w+\s*\w*\s*\)', '()', line)  # Ignore function parameters

    # Token pattern to capture common constructs in code
    token_pattern = r'''
        "[^"]*"                 # Strings inside double quotes
        | '[^']*'               # Strings inside single quotes
        | \+\+|--               # Increment/decrement operators
        | \b(?:if|for|while|switch|case|default|catch|throw)\b  # Control structures
        | \b(?:int|float|double|char|boolean|long|short|byte|void|string)\b  # Data types
        | [\+\*/%=&|<>!~^]      # Operators (except '-')
        | ==|>=|<=|!=           # Relational operators
        | -?\b\d+\b             # Numbers (positive or negative)
        | \.                    # Dot operator for method/field access
        | ,                     # Comma separator
        | \(|\)|\{|\}|\[|\]     # Brackets and braces
        | [a-zA-Z_]\w*          # Identifiers (variable names, method names, etc.)
    '''

    # Print the line before tokenizing to ensure correct pattern application
    print("Line before tokenizing:", line)

    # Remove ignored tokens like return, try, and ;
    line = re.sub(r'\b(return|try)\b', '', line)
    line = re.sub(r';', '', line)

    # Apply tokenization based on the updated pattern
    tokens = re.findall(token_pattern, line, re.VERBOSE)

    # Print the tokens to confirm what’s being captured
    print("Tokens captured:", tokens)

    # Return the count of tokens and the tokens list
    return len(tokens), tokens
    
"""


# Function to calculate control structure complexity for a single line (Wc)
def calculate_control_structure_complexity(line):
    wc = 0
    wc += len(re.findall(r'\bif\b', line))  # Branch (if-else)
    wc += len(re.findall(r'\bfor\b|\bwhile\b|\bdo\b', line)) * 2  # Iterative
    wc += len(re.findall(r'\bcase\b', line))  # Switch case
    return wc

"""
def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
    # Define control structures that affect nesting
    control_structures = ['if', 'else', 'for', 'while', 'switch', 'try']

    # Initialize Wn (nesting weight)
    wn = 0

    # Check if the line starts a control structure
    if any(cs in line for cs in control_structures):
        # Check for 'else if' (or 'else' and 'if' on the same line)
        if 'else if' in line:
            wn = current_nesting  # Weight for 'else if'
        elif 'else' in line:
            wn = 1  # Weight for 'else'
        else:
            current_nesting += 1  # Increase nesting level for 'if'
            control_structure_stack.append(current_nesting)  # Push current nesting level to the stack
            wn = current_nesting  # Weight for 'if'

        in_control_structure = True  # We're inside a control structure
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Check for closing '}' of control structure
    if '}' in line and control_structure_stack:
        current_nesting = control_structure_stack.pop() - 1  # Pop the last nesting level
        wn = current_nesting + 1  # Wn based on the new current nesting level

        # If stack is empty after popping, we're not in a control structure anymore
        in_control_structure = len(control_structure_stack) > 0
        return current_nesting, in_control_structure, control_structure_stack, wn

    # If we encounter a statement at the same nesting level as the control structure
    if in_control_structure:
        wn = current_nesting  # Assign weight based on the current nesting level

    return current_nesting, in_control_structure, control_structure_stack, wn

"""


def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
    # Define control structures that affect nesting
    control_structures = ['if', 'else', 'for', 'while', 'switch', 'try', 'catch', 'finally']

    # Initialize Wn (nesting weight)
    wn = 0

    # Check if the line starts a control structure
    if any(cs in line for cs in control_structures):
        # Increase nesting level for control structures
        current_nesting += 1
        control_structure_stack.append(current_nesting)  # Push current nesting level to the stack

        # Assign weight based on the new current nesting level
        wn = current_nesting
        in_control_structure = True  # We're inside a control structure
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Check for closing '}' of control structure
    if '}' in line and control_structure_stack:
        current_nesting = control_structure_stack.pop() - 1  # Pop the last nesting level

        # If stack is empty after popping, we're not in a control structure anymore
        in_control_structure = len(control_structure_stack) > 0

        # Return the updated nesting level and weight
        return current_nesting, in_control_structure, control_structure_stack, 0

    # If we're not in any control structure, assign a weight of 0 for sequential statements
    if not in_control_structure:
        wn = 0
    else:
        # If in a control structure, assign weight based on the current nesting level
        wn = current_nesting

    return current_nesting, in_control_structure, control_structure_stack, wn

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

# Function to calculate weight due to compound conditional statements
def calculate_compound_condition_weight(line):
    # Initialize weight for the condition
    weight = 0

    # Identify simple conditions (if statements without logical operators)
    simple_condition_pattern = r'\bif\s*\(.*?\)'

    # Identify compound conditions (if statements with logical operators: && or ||)
    # compound_condition_pattern = r'\bif\s*\(.*?(?:&&|\|\|).*?\)'
    compound_condition_pattern = r'\bif\s*\(.*?(?<!==)(?:&&|\|\|).*?\)'

    # First, check for compound conditions
    compound_conditions = re.findall(compound_condition_pattern, line)
    if compound_conditions:
        # For each compound condition, count the number of logical operators
        for condition in compound_conditions:
            # Count logical operators (&&, ||) in the condition
            logical_operators = len(re.findall(r'&&|\|\|', condition))
            weight += 1 + logical_operators  # 1 for base condition + number of logical operators
    else:
        # If it's a simple condition (no logical operators)
        simple_conditions = re.findall(simple_condition_pattern, line)
        if simple_conditions:
            weight += 1  # Simple condition gets a weight of 1

    return weight


# Function to calculate weight due to try-catch-finally blocks
def calculate_try_catch_weight(line, current_nesting_level):
    weight = 0

    # Detecting try blocks
    if re.search(r'\btry\b', line):
        # Weight is 1 for the outer try and 2 for nested try blocks
        if current_nesting_level > 1:
            weight += 2  # Inner try block weight
        else:
            weight += 1  # Outer try block weight

    # Detecting catch blocks
    elif re.search(r'\bcatch\b', line):
        # Weight is 1 for the outer catch and 2 for nested catch blocks
        if current_nesting_level > 1:
            weight += 2  # Inner catch block weight
        else:
            weight += 1  # Outer catch block weight

    # Detecting finally block
    elif re.search(r'\bfinally\b', line):
        weight += 2  # Weight of 2 for the finally block

    return weight

# Function to calculate weight due to thread operations
def calculate_thread_weight(line):
    weight = 0

    # Detect thread creation
    if re.search(r'new\s+Thread\(', line):
        weight += 2  # Weight of 2 for thread creation

    # Detect thread synchronization
    if re.search(r'synchronized\s*\(', line):
        weight += 3  # Weight of 3 for synchronized blocks/methods

    return weight

# Set up logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Function to extract class references for CBO calculation
def extract_class_references(java_code):
    logging.info("Starting extraction of class references.")
    # Patterns to match class instantiation, inheritance, and interface implementation
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    inheritance_pattern = r'class\s+\w+\s+extends\s+([A-Z][\w]*)'
    interface_pattern = r'class\s+\w+\s+implements\s+([A-Z][\w]*)'
    constructor_pattern = r'\bpublic\s+\w+\s*\(([^)]*)\)'
    method_pattern = r'\b(public|private|protected)?\s+(\w+|\s*<[^>]+>)\s+(\w+)\s*\(([^)]*)\)'

    # Dictionary to store class references per class
    class_references = {}

    # Split code into lines and process each line
    lines = java_code.splitlines()

    current_class = None
    for line in lines:
        # Check for class declaration (detect current class)
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            # Ensure the current class is in the dictionary
            if current_class not in class_references:
                class_references[current_class] = {}
        # Print the current class for debugging
        print(f"Current class: {current_class}")

        if current_class:
            # Find class instantiations
            instantiations = re.findall(instantiation_pattern, line)
            for instantiated_class in instantiations:
                class_references[current_class][instantiated_class] = 2  # Tight coupling: assign 2
                logging.debug(f"Class instantiation found: {instantiated_class} in class {current_class}")

            # Find inheritance (extends)
            inheritance = re.findall(inheritance_pattern, line)
            for inherited_class in inheritance:
                class_references[current_class][inherited_class] = 2  # Tight coupling: assign 2
                logging.debug(f"Inheritance found: {inherited_class} for class {current_class}")

            # Find implemented interfaces (implements)
            interfaces = re.findall(interface_pattern, line)
            for implemented_interface in interfaces:
                class_references[current_class][implemented_interface] = 1  # Loose coupling: assign 1
                logging.debug(f"Interface implementation found: {implemented_interface} for class {current_class}")

            # List of Java primitive data types and their wrapper classes
            excluded_types = [
                'int', 'float', 'double', 'boolean', 'char', 'byte', 'short', 'long',  # Primitives
                'Integer', 'Float', 'Double', 'Boolean', 'Character', 'Byte', 'Short', 'Long',  # Wrapper classes
                'String'  # Adding String as it's often treated like a primitive
            ]

            # Check for any method signature and log method details
            method_match = re.search(method_pattern, line)
            if method_match:
                method_name = method_match.group(3)
                parameters = method_match.group(4)
                logging.debug(f"Found method '{method_name}' in class {current_class} with parameters: {parameters}")

                # Find all class names in method parameters (classes start with an uppercase letter)
                # This also checks inside generics like List<Order>, Set<Customer>
                param_classes = re.findall(r'\b([A-Z][\w]*)\b', parameters)

                # Register each found class as tight coupling, excluding primitives and wrapper classes
                for param_class in param_classes:
                    if param_class not in excluded_types:  # Exclude primitives and wrappers
                        class_references[current_class][param_class] = 2  # Tight coupling to the parameter class
                        logging.debug(
                            f"Tight coupling with {param_class} found in method {method_name} of class {current_class}")

                # Additional check for generics (classes inside angle brackets <>)
                # Example: List<Order>, Set<Customer>, ArrayList<Product>
                generic_matches = re.findall(r'<\s*([A-Z][\w]*)\s*>', parameters)

                for generic_class in generic_matches:
                    if generic_class not in excluded_types:  # Exclude primitives and wrappers
                        class_references[current_class][generic_class] = 2  # Tight coupling to the generic class
                        logging.debug(
                            f"Tight coupling with generic type {generic_class} found in method {method_name} of class {current_class}")

            # Find constructor parameters and add them as references
            constructor_matches = re.findall(constructor_pattern, line)
            for match in constructor_matches:
                # Extract parameter classes, excluding primitive types and wrappers
                param_classes = re.findall(r'([A-Z][\w]*)', match)

                for param_class in param_classes:
                    if param_class not in excluded_types:  # Exclude primitives and wrappers
                        class_references[current_class][param_class] = 1  # Loose coupling: assign 1
                        logging.debug(f"Constructor parameter found: {param_class} in class {current_class}")

                # Handle generics like List<Order>, Set<Customer>, etc.
                generic_matches = re.findall(r'<\s*([A-Z][\w]*)\s*>', match)
                for generic_class in generic_matches:
                    if generic_class not in excluded_types:  # Exclude primitives and wrappers inside generics
                        class_references[current_class][generic_class] = 1  # Loose coupling: assign 1
                        logging.debug(f"Generic type parameter found: {generic_class} in class {current_class}")

    logging.info("Finished extracting class references.")
    logging.info(class_references)

    return class_references


# Function to calculate CBO
def calculate_cbo(class_references):
    cbo_results = {}

    # Create a set to track all class references globally
    #all_references = set()

    # Loop through each class's references and count them
    for class_name, references in class_references.items():
        logging.debug(f"Calculating CBO for class {class_name} with references: {references}")
        # Count unique references to other classes
        #unique_references = references - {class_name}  # Exclude self-references
        # cbo_results[class_name] = len(unique_references)  # CBO is the number of unique class references
        #all_references.update(unique_references)  # Track all references
        cbo_results[class_name] = sum(references.values())
        logging.info(f"CBO for class {class_name}: {cbo_results[class_name]}")

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
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,0) + 1  # Weight = 1
                    logging.info(f'Found simple message passing in {current_class}: {method_name} (Weight = 1)')  # Log simple message passing

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
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name,0) + 3  # Weight = 3
                    logging.info(f'Found complex message passing in {current_class}: {method_name} (Weight = 3)')  # Log complex message passing
            # Find callback handling in async methods
            callback_messages = re.findall(callback_pattern, line)
            for method_name in callback_messages:
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3  # Weight = 3
                    logging.info(f'Found callback in async message passing in {current_class}: {method_name} (Weight = 3)')

            # Find exceptional handling in async methods
            exceptional_messages = re.findall(exceptional_pattern, line)
            for method_name in exceptional_messages:
                if method_name not in ignore_methods:
                    message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 3  # Weight = 3
                    logging.info(f'Found exceptional handling in async message passing in {current_class}: {method_name} (Weight = 3)')

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

# Revised training data with WCC metrics
training_data = np.array([
    [0, 0, 0, 0, 1, 2],  # Simple sequential code, low complexity
    [1, 1, 1, 1, 0, 0],  # Branching with simple condition, low complexity
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 2, 1, 4, 0, 0],
    [0, 4, 1, 0, 2, 0],
    [0, 4, 1, 0, 0, 0],
    [0, 3, 1, 0, 0, 0],
    [0, 2, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 2, 0],
    [2, 2, 0, 2, 1, 0],  # Iterative with a try-catch, moderate complexity
    [1, 3, 2, 1, 2, 1],  # Complex nested conditions with threads
    [1, 3, 1, 1, 2, 3],  # Moderate nesting and thread synchronization
    [0, 0, 0, 0, 0, 0],  # Sequential, no complexity
    [2, 3, 1, 2, 3, 4],  # Nested loops with complex try-catch and threads
    [1, 2, 2, 2, 1, 2],  # Moderate complexity with threads and conditions
    [2, 1, 3, 0, 0, 0],  # Simple branch with deep inheritance
    [0, 0, 3, 0, 0, 0],  # Simple branch with deep inheritance
    [0, 1, 0, 1, 0, 0],   # Sequential in nested try-catch
    [2, 2, 1, 0, 0, 0],
    [0, 2, 1, 0, 0, 0],
    [1, 3, 1, 1, 0, 0],
    [0, 3, 1, 0, 0, 0],
])

# Corresponding labels based on WCC guidelines
labels = [
    "no action needed",
    "no action needed",
    "no action needed",
    "no action needed",
    "reduce no of compund conditional statments",
    "consider reducing nesting in try catch",
    "consider reducing nesting",
    "no action needed",
    "no action needed",
    "no action needed",
    "no action needed",
    "consider simplifying control structures",
    "urgent refactor needed",
    "consider reducing nesting",
    "no action needed",
    "urgent refactor needed",
    "consider simplifying thread management",
    "consider reducing inheritance",
    "consider reducing inheritance",
    "no action needed",
    "no action needed",
    "no action needed",
    "no action needed",
    "no action needed",
]

print("Length of training_data:", len(training_data))
print("Length of labels:", len(labels))
# Step 1: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)

# Step 2: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to calculate inheritance level (Wi)
def calculate_inheritance_level2(class_name):
    # Retrieve the inheritance depth from the dictionary, default to 1 if not found
    return inheritance_depth.get(class_name, 1)


# Step 4: Integrate with existing rule-based recommendations
def ai_recommend_refactoring(line_complexities):
    recommendations = []

    for line_info in line_complexities:
        metrics = [
            line_info['control_structure_complexity'],
            line_info['nesting_level'],
            line_info['inheritance_level'],
            line_info['compound_condition_weight'],
            line_info['try_catch_weight'],
            line_info['thread_weight']
        ]
        # AI-based recommendation
        ai_recommendation = model.predict([metrics])

        # Add to recommendations
        recommendations.append({
            'line_number': line_info['line_number'],
            'line_content': line_info['line_content'],
            'recommendation': ai_recommendation
        })

    return recommendations


# Main function to calculate complexity line by line
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
            line, current_nesting, in_control_structure,control_structure_stack
        )
        current_inheritance = calculate_inheritance_level(line, current_inheritance)
        compound_condition_weight = calculate_compound_condition_weight(line)
        try_catch_weight = calculate_try_catch_weight(line, current_nesting)
        thread_weight = calculate_thread_weight(line)

        line_complexities.append({
            'line_number': i,
            'line_content': line.strip(),
            'size': size,
            'tokens':tokens,
            'control_structure_complexity': wc,
            'nesting_level': current_nesting,
            'inheritance_level': current_inheritance,
            'compound_condition_weight': compound_condition_weight,
            'try_catch_weight': try_catch_weight,
            'thread_weight': thread_weight,
        })

    # Get AI recommendations for each line in the file
    recommendations = ai_recommend_refactoring(line_complexities)
    # Print the recommendations
    for recommendation in recommendations:
        print(f"Line {recommendation['line_number']}: {recommendation['line_content']}")
        print(f"Recommendation: {recommendation['recommendation']}\n")

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


def calculate_code_complexity_multiple_files(file_contents):
    results = {}

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

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        print("Class References: " , class_references)
        message_passing = extract_message_passing(content)
        method_complexities = calculate_code_complexity_by_method(content)
        print("Method complex",method_complexities)


        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare line_complexities for recommendation engine
        line_complexities = []

        for line_number, line in enumerate(lines, start=1):
            # Calculate size (token count)
            size, tokens = calculate_size(line)

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
                })
                complexity_data.append([
                    line_number,
                    line.strip(),
                    size,
                    ', '.join(tokens),
                    0, 0, 0, 0, 0, 0, 0
                ])
                continue

            # Calculate control structure complexity
            control_structure_complexity = calculate_control_structure_complexity(line)

            # Calculate nesting level
            current_nesting, in_control_structure, control_structure_stack, wn = calculate_nesting_level(
                line, current_nesting, in_control_structure, control_structure_stack
            )

            # Calculate inheritance level
            # current_inheritance = calculate_inheritance_level(line, current_inheritance)
            # Calculate inheritance level (using tracked inheritance from other files)
            current_inheritance = calculate_inheritance_level2(class_name)

            # Calculate weights due to compound conditions
            compound_condition_weight = calculate_compound_condition_weight(line)

            # Calculate weights for try-catch-finally blocks
            try_catch_weight = calculate_try_catch_weight(line, current_nesting)

            # Calculate weights for thread operations
            thread_weight = calculate_thread_weight(line)

            # Append complexity details to line_complexities for recommendations
            line_complexities.append({
                'line_number': line_number,
                'line_content': line.strip(),
                'size': size,
                'tokens': tokens,
                'control_structure_complexity': control_structure_complexity,
                'nesting_level': wn,
                'inheritance_level': current_inheritance,
                'compound_condition_weight': compound_condition_weight,
                'try_catch_weight': try_catch_weight,
                'thread_weight': thread_weight,
            })

            # Calculate the total complexity for this line (this could be the sum of all the metrics)
            total_complexity = (
                size + control_structure_complexity + wn + current_inheritance +
                compound_condition_weight + try_catch_weight + thread_weight
            )

            # Collect the line's metrics
            complexity_data.append([
                line_number,
                line.strip(),
                size,
                ', '.join(tokens),
                control_structure_complexity,
                wn,
                current_inheritance,
                compound_condition_weight,
                try_catch_weight,
                thread_weight,
                total_complexity,
            ])
        # Get AI recommendations for each line in the file
        recommendations = ai_recommend_refactoring(line_complexities)

        # Filter out "No action needed" recommendations
        filtered_recommendations = [
            rec for rec in recommendations if rec['recommendation'] != "no action needed"
        ]

        # Calculate contributing factors and plot pie chart
        complexity_factors = calculate_complexity_factors(filename, complexity_data)
        pie_chart_path = plot_complexity_pie_chart(filename, complexity_factors)

        # results[filename] = complexity_data
        results[filename] = {
            'complexity_data': complexity_data,
            'cbo': cbo_value,
            'mpc': mpc_value,
            'method_complexities':method_complexities,
            'recommendation': filtered_recommendations,
            'pie_chart_path': pie_chart_path
        }

    return results


# Refined method pattern for accurate method detection
method_pattern = re.compile(
    r'^\s*(public|private|protected)?\s*(static\s+)?'  # Access modifiers and 'static' keyword
    r'(\w+\s+)?'  # Optional return type (including void)
    r'(\w+)\s*\([^)]*\)\s*\{'  # Method name and parameter list
)

# Keywords to ignore to prevent detecting control structures as methods
control_keywords = {'if', 'for', 'while', 'switch', 'catch'}

"""
# Main function to calculate complexity for each method
def calculate_code_complexity_by_method(code):
    lines = code.splitlines()
    current_nesting = 0
    control_structure_stack = []
    current_inheritance = 0
    in_control_structure = False
    method_complexities = []
    current_method = None
    method_complexity = {
        'size': 0,
        'control_structure_complexity': 0,
        'nesting_level': 0,
        'inheritance_level': 0,
        'compound_condition_weight': 0,
        'try_catch_weight': 0,
        'thread_weight': 0,
        'total_complexity': 0
    }

    for i, line in enumerate(lines, start=1):
        # Remove unnecessary symbols and keywords
        processed_line = re.sub(r'\b(public|private|protected|static|else|return|try)\b', '', line)
        processed_line = processed_line.strip()

        # Detect method start using refined pattern
        method_match = method_pattern.match(line)
        if method_match and not any(keyword in line for keyword in control_keywords):
            # Save the previous method's complexity if we were in a method
            if current_method:
                print(f"Detected method end: {current_method} with complexity: {method_complexity}")
                method_complexities.append({**method_complexity, 'method_name': current_method})

            # Initialize tracking for the new method
            current_method = method_match.group(0).strip()  # Capture method signature
            print(f"Detected method start: {current_method}")
            method_complexity = {k: 0 for k in method_complexity}  # Reset for new method
            continue  # Skip further complexity calculations for the declaration line

        # Accumulate complexity metrics for the current method
        if current_method:
            size, tokens = calculate_size(processed_line)
            print(f"Size: {size}")
            print(f"Tokens: {tokens}")
            method_complexity['size'] += size
            method_complexity['control_structure_complexity'] += calculate_control_structure_complexity(line)
            current_nesting, in_control_structure, control_structure_stack, wn = calculate_nesting_level(
                line, current_nesting, in_control_structure, control_structure_stack
            )
            method_complexity['nesting_level'] += wn
            print(f"WN: {line}")
            method_complexity['inheritance_level'] += calculate_inheritance_level(line, current_inheritance)
            print(f"exity: {calculate_inheritance_level(line, current_inheritance)}")
            method_complexity['compound_condition_weight'] += calculate_compound_condition_weight(line)
            method_complexity['try_catch_weight'] += calculate_try_catch_weight(line, current_nesting)
            method_complexity['thread_weight'] += calculate_thread_weight(line)
            # method_complexity['total_complexity'] = sum(method_complexity.values())

            # Calculate total complexity with weighted metrics
            method_complexity['total_complexity'] = (
                    method_complexity['size']  +
                    method_complexity['control_structure_complexity'] +
                    method_complexity['nesting_level'] +
                    method_complexity['inheritance_level']  +
                    method_complexity['compound_condition_weight']  +
                    method_complexity['try_catch_weight']  +
                    method_complexity['thread_weight']
            )

            # Detect method end by closing brace at correct nesting level
            if '}' in line and current_nesting == 0:
                print(f"Detected method end: {current_method} with complexity: {method_complexity}")
                method_complexities.append({**method_complexity, 'method_name': current_method})
                current_method = None

    # Append the last method if not already saved
    if current_method:
        print(f"Final detected method end: {current_method} with complexity: {method_complexity}")
        method_complexities.append({**method_complexity, 'method_name': current_method})

    print("Method Complexities:", method_complexities)
    return method_complexities
"""

# Function to calculate complexity for each method in a file
def calculate_code_complexity_by_method(content):
    methods = {}
    method_name = None
    method_lines = []

    # Split content by line and analyze each one
    for line in content.splitlines():
        # Detect method declarations (example pattern; adapt as needed)
        match = re.search(r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(.*\)\s*{', line)
        if match:
            # If we're already in a method, calculate its complexity
            if method_name:
                methods[method_name] = calculate_complexity_for_method(method_lines)
            # Start a new method
            method_name = match.group(1)
            method_lines = [line]
        elif method_name:
            # Append lines within the current method
            method_lines.append(line)

    # Final method complexity calculation
    if method_name:
        methods[method_name] = calculate_complexity_for_method(method_lines)

    return methods

# Helper function to calculate complexity for a method based on lines of code
def calculate_complexity_for_method(lines):
    size = 0
    control_structure_complexity = 0
    nesting_level = 0
    inheritance_level = 0
    compound_condition_weight = 0
    try_catch_weight = 0
    thread_weight = 0
    current_inheritance = 0

    current_nesting = 0
    control_structure_stack = []

    # Analyze each line within the method
    for line in lines:
        # Calculate size (token count) for this line
        line_size, tokens = calculate_size(line)
        size += line_size

        # Calculate control structure complexity
        control_structure_complexity += calculate_control_structure_complexity(line)

        # Calculate nesting level and control structures
        current_nesting, _, control_structure_stack, wn = calculate_nesting_level(
            line, current_nesting, False, control_structure_stack
        )
        nesting_level += wn

        inheritance_level += calculate_inheritance_level(line, current_inheritance)

        # Compound condition weight
        compound_condition_weight += calculate_compound_condition_weight(line)

        # Try-catch weight
        try_catch_weight += calculate_try_catch_weight(line, current_nesting)

        # Thread weight
        thread_weight += calculate_thread_weight(line)

    # Sum up the complexity metrics for this method
    total_complexity = (
        size + control_structure_complexity + nesting_level + inheritance_level +
        compound_condition_weight + try_catch_weight + thread_weight
    )

    return {
        "size": size,
        "control_structure_complexity": control_structure_complexity,
        "nesting_level": nesting_level,
        "inheritance_level": inheritance_level,
        "compound_condition_weight": compound_condition_weight,
        "try_catch_weight": try_catch_weight,
        "thread_weight": thread_weight,
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

    for line in data:
        total_size += line[2]
        total_control_structure_complexity += line[4]
        total_nesting_level += line[5]
        total_inheritance_level += line[6]
        total_compound_condition_weight += line[7]
        total_try_catch_weight += line[8]
        total_thread_weight += line[9]

    return {
        'Size': total_size,
        'Control Structure Complexity': total_control_structure_complexity,
        'Nesting Level': total_nesting_level,
        'Inheritance Level': total_inheritance_level,
        'Compound Condition Weight': total_compound_condition_weight,
        'Try-Catch Weight': total_try_catch_weight,
        'Thread Weight': total_thread_weight
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