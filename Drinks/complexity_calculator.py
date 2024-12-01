import logging
import os
import re

import pandas as pd
from pyparsing import (
    Word, alphas, alphanums, Keyword, Suppress, Optional, Group, ZeroOrMore, Forward, OneOrMore, printables, SkipTo,
    ParseException, nestedExpr, Literal, restOfLine
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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


# Function to calculate control structure complexity for a single line (Wc)
def calculate_control_structure_complexity(line):
    wc = 0
    wc += len(re.findall(r'\bif\b', line))  # Branch (if-else)
    wc += len(re.findall(r'\bfor\b|\bwhile\b|\bdo\b', line)) * 2  # Iterative
    wc += len(re.findall(r'\bcase\b', line))  # Switch case
    return wc


# def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
#     # Define control structures that affect nesting
#     control_structures = ['if', 'else', 'for', 'while', 'switch', 'try']
#
#     # Initialize Wn (nesting weight)
#     wn = 0
#
#     # Check if the line starts a control structure
#     if any(cs in line for cs in control_structures):
#         # Check for 'else if' (or 'else' and 'if' on the same line)
#         if 'else if' in line:
#             wn = current_nesting  # Weight for 'else if'
#         elif 'else' in line:
#             wn = 1  # Weight for 'else'
#         else:
#             current_nesting += 1  # Increase nesting level for 'if'
#             control_structure_stack.append(current_nesting)  # Push current nesting level to the stack
#             wn = current_nesting  # Weight for 'if'
#
#         in_control_structure = True  # We're inside a control structure
#         return current_nesting, in_control_structure, control_structure_stack, wn
#
#     # Check for closing '}' of control structure
#     if '}' in line and control_structure_stack:
#         current_nesting = control_structure_stack.pop() - 1  # Pop the last nesting level
#         wn = current_nesting + 1  # Wn based on the new current nesting level
#
#         # If stack is empty after popping, we're not in a control structure anymore
#         in_control_structure = len(control_structure_stack) > 0
#         return current_nesting, in_control_structure, control_structure_stack, wn
#
#     # If we encounter a statement at the same nesting level as the control structure
#     if in_control_structure:
#         wn = current_nesting  # Assign weight based on the current nesting level
#
#     return current_nesting, in_control_structure, control_structure_stack, wn


# def calculate_nesting_level(java_code):
#     """
#     Calculates the nesting level of control structures (if-else, for, while, do-while)
#     in Java code line by line, ignoring class and method declarations.
#     """
#     # Remove comments from the Java code
#     java_code = remove_comments(java_code)
#     lines = java_code.splitlines()
#
#     # Control structure keywords to consider
#     control_keywords = ['if', 'else', 'for', 'while', 'do']
#
#     # State variables to track nesting level
#     current_nesting = 0
#     control_structure_stack = []  # Stack to track nesting levels
#     nesting_levels = []
#
#     # Regular expressions for detecting control structures
#     control_regex = re.compile(r'\b(if|else if|else|for|while|do)\b')
#     open_brace_regex = re.compile(r'\{')  # Opening brace
#     close_brace_regex = re.compile(r'\}')  # Closing brace
#
#     for line_no, line in enumerate(lines, start=1):
#         stripped_line = line.strip()
#
#         # Skip empty lines
#         if not stripped_line:
#             nesting_levels.append((line_no, stripped_line, current_nesting))
#             continue
#
#         # Check if the line contains a control structure
#         control_match = control_regex.search(stripped_line)
#         if control_match:
#             # Increment nesting level for a new control structure
#             current_nesting += 1
#             control_structure_stack.append(current_nesting)
#
#         # Record the current nesting level for the line
#         nesting_levels.append((line_no, stripped_line, current_nesting))
#
#         # Adjust nesting level based on braces
#         # Count the opening and closing braces in the line
#         opening_braces = len(open_brace_regex.findall(stripped_line))
#         closing_braces = len(close_brace_regex.findall(stripped_line))
#
#         # Adjust the nesting level for each closing brace
#         if closing_braces > 0:
#             for _ in range(closing_braces):
#                 if control_structure_stack:
#                     control_structure_stack.pop()
#                     current_nesting = len(control_structure_stack)
#
#     return nesting_levels

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



"""
def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):

    # Define control structures to detect
    control_structures = ['if', 'else', 'for', 'while', 'switch', 'try']
    control_regex = re.compile(r'\b(?:' + '|'.join(control_structures) + r')\b')

    # Initialize Wn (nesting weight)
    wn = 0

    # Check if the line starts a control structure
    if control_regex.search(line):
        if 'else if' in line or 'else if' in line.replace(' ', ''):  # Handle 'else if'
            wn = current_nesting  # Weight for 'else if'
        elif 'else' in line and 'if' not in line:  # Handle 'else' (not 'else if')
            wn = current_nesting  # Weight for 'else'
        else:  # Handle new control structures like 'if', 'for', etc.
            current_nesting += 1
            control_structure_stack.append(current_nesting)
            wn = current_nesting

        in_control_structure = True
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Check for closing '}' of a control structure
    if '}' in line and control_structure_stack:
        control_structure_stack.pop()
        current_nesting = len(control_structure_stack)
        wn = current_nesting

        in_control_structure = len(control_structure_stack) > 0
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Handle sequential statements (non-control-structure lines)
    if in_control_structure:
        # Sequential statements inside control structures take the current nesting weight
        wn = current_nesting
    else:
        wn = 0  # Reset weight for lines outside control structures

    return current_nesting, in_control_structure, control_structure_stack, wn
"""


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
# def calculate_try_catch_weight(line, current_nesting_level):
#     weight = 0
#
#     # Detecting try blocks
#     if re.search(r'\btry\b', line):
#         # Weight is 1 for the outer try and 2 for nested try blocks
#         if current_nesting_level > 1:
#             weight += 2  # Inner try block weight
#         else:
#             weight += 1  # Outer try block weight
#
#     # Detecting catch blocks
#     elif re.search(r'\bcatch\b', line):
#         # Weight is 1 for the outer catch and 2 for nested catch blocks
#         if current_nesting_level > 1:
#             weight += 2  # Inner catch block weight
#         else:
#             weight += 1  # Outer catch block weight
#
#     # Detecting finally block
#     elif re.search(r'\bfinally\b', line):
#         weight += 2  # Weight of 2 for the finally block
#
#     return weight


# def calculate_try_catch_weight(java_code):
#     """
#     Calculates the weight of nesting levels specifically for try-catch-finally blocks in Java code.
#     - Increment nesting level for `try`.
#     - Assign weights line by line for `catch` and `finally`.
#     """
#     java_code = remove_comments(java_code)
#     lines = java_code.splitlines()
#
#     # State variables
#     current_nesting = 0
#     total_weight = 0
#     nesting_levels = []
#     line_weights = {}
#
#     # Regular expressions
#     control_regex = re.compile(r'\b(try|catch|finally)\b')
#
#     # Weights for `catch` based on nesting levels
#     catch_weights = {1: 1, 2: 3, 3: 4, 4: 5}
#
#     for line_no, line in enumerate(lines, start=1):
#         stripped_line = line.strip()
#
#         if not stripped_line:
#             # Empty lines, just carry over nesting and weight
#             nesting_levels.append((line_no, stripped_line, current_nesting, 0))
#             continue
#
#         control_match = control_regex.search(stripped_line)
#         if control_match:
#             control_type = control_match.group()
#
#             if control_type == 'try':
#                 current_nesting += 1  # Increment nesting for `try`
#
#             elif control_type == 'catch':
#                 # Assign weight for `catch` based on the current nesting level
#                 weight = catch_weights.get(current_nesting, 5)
#                 line_weights[line_no] = weight
#
#             elif control_type == 'finally':
#                 # Assign a fixed weight of 2 for `finally`
#                 line_weights[line_no] = 2
#
#         nesting_levels.append((line_no, stripped_line, current_nesting, line_weights.get(line_no, 0)))
#
#         # Adjust nesting levels for closing braces
#         if stripped_line.endswith('}'):
#             current_nesting -= 1
#
#     return nesting_levels, line_weights

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
    catch_weights = {1: 1, 2: 3, 3: 4, 4: 5}
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


"""
# Function to calculate weight due to thread operations
def calculate_thread_weight(line):
    weight = 0

    # Detect thread creation
    if re.search(r'new\s+Thread\(', line):
        weight += 2  # Weight of 2 for thread creation

    # Detect thread synchronization
    if re.search(r'synchronized\s*\(', line):
        weight += 5  # Weight of 3 for synchronized blocks/methods

    return weight
"""

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


# Set up logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Function to extract class references for CBO calculation
def extract_class_references(java_code):
    logging.info("Starting extraction of class references.")
    # Patterns to match class instantiation, inheritance, and interface implementation
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    inheritance_pattern = r'class\s+\w+\s+extends\s+([A-Z][\w]*)'
    interface_pattern = r'class\s+\w+\s+implements\s+([A-Z][\w]*)'
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'
    method_pattern = r'(public|private)?\s*void\s+(?!set[A-Z])\w+\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'
    setter_injection_pattern = r'(public)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'

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
                if instantiated_class in class_references[current_class]:
                    print("param_classes-----------------------------284", instantiated_class)
                    class_references[current_class][instantiated_class] += 3  # Tight coupling: assign 2
                    logging.debug(f"Class instantiation found: {instantiated_class} in class {current_class}")
                else:
                    class_references[current_class][instantiated_class] = 3

            # Find inheritance (extends)
            inheritance = re.findall(inheritance_pattern, line)
            for inherited_class in inheritance:
                if inherited_class in class_references[current_class]:
                    print("param_classes-----------------------------291", inherited_class)
                    class_references[current_class][inherited_class] += 3  # Tight coupling: assign 2
                    logging.debug(f"Inheritance found: {inherited_class} for class {current_class}")
                else:
                    class_references[current_class][inherited_class] = 3

            # Find implemented interfaces (implements)
            interfaces = re.findall(interface_pattern, line)
            for implemented_interface in interfaces:
                print("param_classes-----------------------------298", implemented_interface)
                class_references[current_class][implemented_interface] = 1  # Loose coupling: assign 1
                logging.debug(f"Interface implementation found: {implemented_interface} for class {current_class}")

            constructor_matches = re.findall(constructor_pattern, line)
            for match in constructor_matches:
                # Extract parameter classes, excluding primitive types and wrappers
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                # print("param_classes-----------------------------", param_classes)

                for param_class in param_classes:
                    print("param_classes-----------------------------309", param_class)
                    if param_class in class_references[current_class]:  # Exclude primitives and wrappers
                        class_references[current_class][param_class] += 1  # Loose coupling: assign 1
                        print("class_references[current_class][param_class]-----------------------------------------------------pppppppppppppppp", class_references[current_class][param_class])
                        logging.debug(f"Constructor parameter found: {param_class} in class {current_class}")
                    else:
                        class_references[current_class][param_class] = 1

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, line)
            for setter_class in setter_matches:
                class_references[current_class][setter_class] = 1  # Loose coupling
                logging.debug(f"Setter DI: {setter_class} in class {current_class}")


            # List of Java primitive data types and their wrapper classes
            excluded_types = [
                'int', 'float', 'double', 'boolean', 'char', 'byte', 'short', 'long',  # Primitives
                'Integer', 'Float', 'Double', 'Boolean', 'Character', 'Byte', 'Short', 'Long',  # Wrapper classes
                'String' , 'List', 'Map', 'Set', 'HashMap', 'ArrayList', 'HashSet', 'LinkedList',  # Java standard library classes
        'Optional', 'Stream'
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
                        if param_class in class_references[current_class]:
                            print("AnotherClass..................................336", param_class)

                            class_references[current_class][param_class] += 3  # Tight coupling to the parameter class
                            logging.debug(
                                f"Tight coupling with {param_class} found in method {method_name} of class {current_class}")
                        else:
                            class_references[current_class][param_class] = 3

                # Additional check for generics (classes inside angle brackets <>)
                # Example: List<Order>, Set<Customer>, ArrayList<Product>
                generic_matches = re.findall(r'<\s*([A-Z][\w]*)\s*>', parameters)

                for generic_class in generic_matches:
                    if generic_class not in excluded_types:  # Exclude primitives and wrappers
                        print("param_classes-----------------------------348", generic_class)
                        class_references[current_class][generic_class] = 3  # Tight coupling to the generic class
                        logging.debug(
                            f"Tight coupling with generic type {generic_class} found in method {method_name} of class {current_class}")


    logging.info("Finished extracting class references.")
    logging.info(class_references)

    return class_references


# Function to calculate CBO
def calculate_cbo(class_references):
    cbo_results = {}


    # Create a set to track all class references globally
    # all_references = set()

    # Loop through each class's references and count them
    for class_name, references in class_references.items():
        logging.debug(f"Calculating CBO for class {class_name} with references: {references}")
        print(f"Calculating CBO for class {class_name} with references: {references}")
        # Count unique references to other classes
        # unique_references = references - {class_name}  # Exclude self-references
        # cbo_results[class_name] = len(unique_references)  # CBO is the number of unique class references
        # all_references.update(unique_references)  # Track all references
        cbo_results[class_name] = sum(references.values())
        logging.info(f"CBO for class {class_name}: {cbo_results[class_name]}")

    return cbo_results

def extract_class_references_with_lines(java_code):
    logging.info("Starting extraction of class references.")
    # Patterns to match class instantiation, inheritance, and interface implementation
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    inheritance_pattern = r'class\s+\w+\s+extends\s+([A-Z][\w]*)'
    interface_pattern = r'class\s+\w+\s+implements\s+([A-Z][\w]*)'
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'
    method_pattern = r'(public|private)?\s*void\s+(?!set[A-Z])\w+\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'
    setter_injection_pattern = r'(public)?\s*void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'

    # Dictionary to store class references per line
    line_references = []

    # Split code into lines and process each line
    lines = java_code.splitlines()
    current_class = None

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

            # Find class instantiations
            instantiations = re.findall(instantiation_pattern, stripped_line)
            for instantiated_class in instantiations:
                weights[instantiated_class] = 3  # Tight coupling due to instantiation

            # Find inheritance (extends)
            inheritance = re.findall(inheritance_pattern, stripped_line)
            for inherited_class in inheritance:
                weights[inherited_class] = 3  # Tight coupling due to inheritance

            # Find implemented interfaces (implements)
            interfaces = re.findall(interface_pattern, stripped_line)
            for implemented_interface in interfaces:
                weights[implemented_interface] = 1  # Loose coupling due to interface implementation

            # Detect constructor parameters
            constructor_matches = re.findall(constructor_pattern, stripped_line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    weights[param_class] = 1  # Loose coupling due to constructor injection

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, stripped_line)
            for setter_class in setter_matches:
                weights[setter_class] = 1  # Loose coupling due to setter injection

            # Detect method parameters
            method_match = re.search(method_pattern, stripped_line)
            if method_match:
                param_classes = re.findall(r'([A-Z][\w]*)', method_match.group(2))
                for param_class in param_classes:
                    weights[param_class] = 3  # Tight coupling due to method parameters

            # Add weights to the current line
            line_data["weights"] = weights

        # Add the processed line data to the list
        line_references.append(line_data)

    logging.info("Finished extracting class references line by line.")
    logging.info(line_references)

    return line_references


# Function to calculate CBO line by line
def calculate_cbo_line_by_line(line_references):
    """
    Calculate Coupling Between Objects (CBO) line by line.
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

"""
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
    [1, 3, 2, 1, 2, 2],  # Complex nested conditions with threads
    [1, 3, 1, 1, 2, 5],  # Moderate nesting and thread synchronization
    [0, 0, 0, 0, 0, 0],  # Sequential, no complexity
    [2, 3, 1, 2, 3, 2],  # Nested loops with complex try-catch and threads
    [1, 2, 2, 2, 1, 2],  # Moderate complexity with threads and conditions
    [2, 1, 3, 0, 0, 0],  # Simple branch with deep inheritance
    [0, 0, 3, 0, 0, 0],  # Simple branch with deep inheritance
    [0, 1, 0, 1, 0, 0],  # Sequential in nested try-catch
    [2, 2, 1, 0, 0, 0],
    [0, 2, 1, 0, 0, 0],
    [0, 3, 1, 0, 0, 0],
    [1, 1, 1, 2, 0, 0],
    [1, 3, 1, 1, 0, 0],
    [0, 3, 1, 0, 0, 0]
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
    "consider reducing nesting",
    "consider reducing nesting",
]
"""

# Training data
training_data = np.array([
    # Format: [control_structure_complexity, nesting_level, inheritance_level, compound_condition_weight, try_catch_weight, thread_weight]
    [0, 0, 0, 0, 1, 2],  # Simple sequential code
    [1, 1, 1, 1, 0, 0],  # Simple branching with one condition
    [0, 0, 1, 0, 0, 0],  # Inheritance with no control structure
    [0, 2, 1, 0, 2, 0],  # Nested try-catch with shallow nesting
    [1, 4, 1, 4, 0, 0],  # Deep nesting and high compound conditions
    [0, 5, 1, 0, 3, 0],  # Excessive nesting in try-catch
    [0, 4, 1, 0, 0, 0],  # Deep nesting with no other complexity
    [0, 3, 1, 4, 0, 0],  # High compound condition weight
    [0, 2, 1, 0, 0, 0],  # Moderate nesting
    [1, 1, 1, 0, 0, 0],  # Simple branching with shallow nesting
    [0, 1, 1, 0, 2, 0],  # Try-catch with shallow nesting
    [3, 3, 0, 2, 1, 0],  # Complex iterative with try-catch
    [1, 4, 3, 5, 2, 2],  # Critical refactor: high inheritance, compound conditions, threads
    [1, 3, 1, 1, 2, 5],  # Threads with moderate nesting
    [0, 0, 0, 0, 0, 0],  # Sequential, no complexity
    [3, 4, 2, 4, 3, 2],  # Critical refactor: high across all metrics
    [2, 2, 3, 2, 1, 2],  # Inheritance and threads with conditions
    [3, 1, 3, 0, 0, 0],  # Inheritance with deep branching
    [0, 0, 4, 0, 0, 0],  # Deep inheritance, low other complexity
    [1, 2, 0, 1, 0, 0],  # Nested try-catch with conditions
    [2, 2, 1, 0, 0, 0],  # Iterative with shallow nesting
    [0, 2, 1, 4, 0, 0],  # Nested try-catch with high compound conditions
    [0, 3, 1, 0, 0, 0],  # Moderate nesting
    [2, 4, 1, 4, 0, 0],  # Deep nesting and compound conditions
    [3, 3, 3, 3, 2, 0],  # Critical refactor: inheritance, threads, nesting
    [0, 3, 1, 0, 0, 0],  # Moderate nesting
    [3, 5, 2, 5, 1, 3],  # Critical refactor: threads and conditions
    [2, 4, 3, 4, 2, 3],  # Threads with deep nesting and conditions
    [1, 3, 2, 1, 0, 2],  # Moderate threads and inheritance
    [2, 2, 1, 3, 1, 1],  # Iterative with compound conditions
    [0, 1, 1, 1, 0, 0],  # Simple branching
    [1, 4, 1, 2, 0, 0],  # Reduce deep nesting
    [3, 4, 2, 3, 2, 3],  # Threads with critical nesting
    [0, 3, 1, 1, 0, 0],  # Nested branching
    [1, 3, 2, 4, 1, 0],  # High inheritance and compound conditions
    [2, 1, 3, 0, 0, 0],  # Deep inheritance
    [1, 3, 3, 1, 0, 0],  # High inheritance
    [3, 5, 4, 5, 3, 2],  # Urgent: high across all metrics
    [0, 1, 1, 0, 0, 0],  # Sequential with shallow nesting
    [2, 3, 3, 2, 2, 2],  # Threads with deep nesting and inheritance
    [3, 3, 3, 4, 3, 2],  # Deep inheritance with threads
    [3, 5, 2, 3, 2, 3],  # Critical refactor for threads and conditions
    [2, 3, 1, 3, 0, 0],  # Moderate compound conditions
    [1, 2, 2, 4, 1, 1],  # Threads and compound conditions
    [0, 3, 2, 2, 1, 0],  # Nested loops and conditions
    [1, 2, 1, 0, 0, 0],  # Moderate complexity
    [2, 3, 3, 1, 0, 1],  # High inheritance and compound conditions
    [1, 2, 1, 4, 0, 0],  # Compound conditions in try-catch
    [2, 4, 3, 5, 3, 3],  # High inheritance, conditions, threads
    [1, 4, 2, 4, 2, 3],  # Deep nesting and threads
    [0, 3, 1, 0, 0, 0],  # Moderate nesting
])

# Corresponding labels
labels = [
    "no action needed",
    "no action needed",
    "no action needed",
    "Consider reducing nested try-catch.",
    "Critical refactor: Deep nesting with high compound conditions.",
    "Critical refactor: Excessive nesting in try-catch.",
    "Consider simplifying deep nesting.",
    "Consider reducing compound conditions.",
    "no action needed",
    "no action needed",
    "no action needed",
    "Consider simplifying complex iterative control structure.",
    "Urgent refactor: High inheritance, compound conditions, and threads.",
    "Reduce thread complexity in nested structures.",
    "no action needed",
    "Critical refactor: High across all metrics.",
    "Reduce thread complexity with inheritance.",
    "Consider simplifying deep branching.",
    "Consider simplifying deep inheritance.",
    "Consider reducing nested try-catch with conditions.",
    "no action needed",
    "Consider reducing try-catch with high compound conditions.",
    "Consider reducing moderate nesting.",
    "Critical refactor: Deep nesting and compound conditions.",
    "Urgent refactor: Inheritance, threads, and nesting.",
    "Reduce thread complexity in critical nesting.",
    "Urgent refactor: Threads and compound conditions.",
    "Urgent refactor: Threads with deep nesting.",
    "Reduce thread complexity with moderate inheritance.",
    "Consider reducing iterative compound conditions.",
    "no action needed",
    "Consider reducing deep nesting.",
    "Critical refactor: Threads with nesting.",
    "Consider simplifying nested branching.",
    "Reduce compound conditions with high inheritance.",
    "Consider simplifying deep inheritance.",
    "Urgent refactor: High across all metrics.",
    "no action needed",
    "no action needed",
    "Urgent refactor: Deep inheritance and threads.",
    "Urgent refactor: Threads and high conditions.",
    "Consider simplifying compound conditions.",
    "Consider reducing thread complexity in compound conditions.",
    "Reduce nested loops and conditions.",
    "no action needed",
    "Reduce compound conditions with high inheritance.",
    "Consider simplifying try-catch with compound conditions.",
    "Urgent refactor: High across all metrics.",
    "Reduce thread complexity in deep nesting.",
    "Consider simplifying moderate nesting.",
"no action needed",
]


# CBO and MPC dataset (per class, not line-by-line)
coupling_data = np.array([
    [2, 3],  # Low MPC, Moderate CBO
    [3, 5],  # High MPC, High CBO
    [4, 6],  # High MPC, High CBO
    [5, 7],  # Very High MPC and CBO
    [1, 2],  # Low MPC and CBO
    [2, 2],  # Low MPC and Moderate CBO
    [7, 9],  # Extremely High MPC and CBO
    [6, 8],  # High MPC and CBO
])

# Corresponding labels for CBO and MPC
coupling_labels = [
    "no action needed",
    "reduce coupling",
    "urgent refactor for high MPC and CBO",
    "urgent refactor for very high MPC and CBO",
    "no action needed",
    "reduce coupling",
    "urgent refactor for extreme MPC and CBO",
    "reduce coupling",
]

print("Length of training_data:", len(training_data))
print("Length of labels:", len(labels))
# Step 1: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)

X_train_coupling, X_test_coupling, y_train_coupling, y_test_coupling = train_test_split(
    coupling_data, coupling_labels, test_size=0.2, random_state=42
)

# Step 2: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_coupling = RandomForestClassifier(n_estimators=100, random_state=42)
model_coupling.fit(X_train_coupling, y_train_coupling)

# Step 3: Evaluate the model
print("=== Complexity Metrics Report ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== Coupling Metrics Report ===")
y_pred_coupling = model_coupling.predict(X_test_coupling)
print(classification_report(y_test_coupling, y_pred_coupling))


# Function to calculate inheritance level (Wi)
def calculate_inheritance_level2(class_name):
    # Retrieve the inheritance depth from the dictionary, default to 1 if not found
    return inheritance_depth.get(class_name, 1)


# Step 4: Integrate with existing rule-based recommendations
def ai_recommend_refactoring(line_complexities, cbo_value, mpc_value):
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

        metrics1 = [
            cbo_value,
            mpc_value
        ]
        # AI-based recommendation
        ai_recommendation = model.predict([metrics])
        ai_recommendation1 = model_coupling.predict([metrics1])

        # Add to recommendations
        recommendations.append({
            'line_number': line_info['line_number'],
            'line_content': line_info['line_content'],
            'recommendation': ai_recommendation,
            'recommendation1': ai_recommendation1,
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

def calculate_code_complexity_multiple_files(file_contents):
    results = {}

    # Step 1: Track inheritance across all files
    track_inheritance_depth_across_files(file_contents)

    result1 = calculate_code_complexity_multiple(file_contents)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@result1@@@@@@@@@@@@@@@@@@@@@@@@", result1)

    mpc_results = calculate_mpc_for_java_code(file_contents)

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

        # Extract CBO results for each line
        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        # print("Method complex",method_complexities)

        print("class_references................................", class_references)

        # nesting_levels  = calculate_nesting_level1(content)
        # display_nesting_levels(nesting_levels)
        nesting_levels = calculate_nesting_level(content)
        nesting_level_dict = {line[0]: line[2] for line in nesting_levels}

        print("nesting_levels----------------------", nesting_levels)

        # try_catch_weights = calculate_try_catch_weight(content)
        # try_catch_weight_dict = {line[0]: line[3] for line in try_catch_weights}
        # print("try_catch_weight_dict###############################################", try_catch_weight_dict)
        #
        # print("try_catch_weights---------------------------------", try_catch_weights)

        try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)

        print("try_catch_weights---------------------------------", try_catch_weight_dict)
        print("try_catch_weights Nesting Levels-++++++++++++++++++++++++++---------------------------------", try_catch_weights)

        thread_weights = calculate_thread_weight(content)

        print("thread_weights---------------------------------", thread_weights)

        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        print("cbo_value$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", cbo_value)

        # Prepare line_complexities for recommendation engine
        line_complexities = []

        method_inheritance = {}

        # Initialize total WCC value for the file
        total_wcc = 0

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
                    'cbo_weights': 0,
                    'mpc_weight': 0,
                })
                complexity_data.append([
                    line_number,
                    line.strip(),
                    size,
                    ', '.join(tokens),
                    0, 0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue

            cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
            total_cbo_weight = sum(cbo_weights.values())

            mpc_weight = mpc_line_data.get(line_number, 0)

            nesting_level = nesting_level_dict.get(line_number, 0)
            try_catch_weight = try_catch_weight_dict.get(line_number, 0)
            print("try_catch_weight", try_catch_weight)
            thread_weight = thread_weights.get(line_number, 0)

            # Calculate control structure complexity
            control_structure_complexity = calculate_control_structure_complexity(line)

            # Calculate nesting level
            # current_nesting, in_control_structure, control_structure_stack, wn = calculate_nesting_level(
            #     line, current_nesting, in_control_structure, control_structure_stack
            # )

            # Calculate inheritance level
            # current_inheritance = calculate_inheritance_level(line, current_inheritance)
            # Calculate inheritance level (using tracked inheritance from other files)
            current_inheritance = calculate_inheritance_level2(class_name)
            method_inheritance[class_name] = current_inheritance

            # Calculate weights due to compound conditions
            compound_condition_weight = calculate_compound_condition_weight(line)

            # Calculate weights for try-catch-finally blocks
            # try_catch_weight = calculate_try_catch_weight(line, current_nesting)

            # Calculate weights for thread operations
            # thread_weight = calculate_thread_weight(line)

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
                'mpc_weight': mpc_weight,
            })

            # Calculate the total complexity for this line (this could be the sum of all the metrics)
            total_complexity = (
                    size + control_structure_complexity + nesting_level + current_inheritance +
                    compound_condition_weight + try_catch_weight + thread_weight + total_cbo_weight + mpc_weight
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
                thread_weight,
                total_cbo_weight,
                mpc_weight,
                total_complexity,
            ])
        # Calculate method complexities
        method_complexities = calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data)
        # Get AI recommendations for each line in the file
        recommendations = ai_recommend_refactoring(line_complexities, cbo_value, mpc_value)

        # Filter out "No action needed" recommendations
        filtered_recommendations = [
            rec for rec in recommendations if rec['recommendation'] != "no action needed"
        ]

        print("method_complexities>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>????????????????????",method_complexities)

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
                "mpc_weight": method_data["mpc_weight"]
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

    return results


# Refined method pattern for accurate method detection
method_pattern = re.compile(
    r'^\s*(public|private|protected)?\s*(static\s+)?'  # Access modifiers and 'static' keyword
    r'(\w+\s+)?'  # Optional return type (including void)
    r'(\w+)\s*\([^)]*\)\s*\{'  # Method name and parameter list
)

# Keywords to ignore to prevent detecting control structures as methods
control_keywords = {'if', 'for', 'while', 'switch', 'catch'}


# Function to calculate complexity for each method in a file
def calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data):
    methods = {}
    method_name = None
    method_lines = []

    nesting_levels = calculate_nesting_level(content)
    nesting_level_dict = {line[0]: line[2] for line in nesting_levels}

    try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)

    print("nesting_levels----------------------", nesting_levels)

    thread_weights = calculate_thread_weight(content)

    # Split content by line and analyze each one
    for line in content.splitlines():
        # Detect method declarations (example pattern; adapt as needed)
        match = re.search(r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(.*\)\s*{', line)
        if match:
            method_name = match.group(1)
            if method_name in ["if", "for", "while", "switch", "Thread", "run"]:  # Ignore control structures
                continue
            # If we're already in a method, calculate its complexity
            if method_name:
                methods[method_name] = calculate_complexity_for_method(method_lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data)
            # Start a new method
            method_name = match.group(1)
            method_lines = [line]
        elif method_name:
            # Append lines within the current method
            method_lines.append(line)

    # Final method complexity calculation
    if method_name:
        methods[method_name] = calculate_complexity_for_method(method_lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data)

    return methods


# Helper function to calculate complexity for a method based on lines of code
def calculate_complexity_for_method(lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data):
    size = 0
    control_structure_complexity = 0
    nesting_level = 0
    compound_condition_weight = 0
    try_catch_weight = 0
    # thread_weight = 0
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

        total_inheritance = method_inheritance[current_class]

        nesting_level = nesting_level_dict.get(line_number, 0)
        total_nesting += nesting_level

        print("total_nesting))))))))))))))))))))))))))))))))))", total_nesting)

        try_catch_weight = try_catch_weight_dict.get(line_number, 0)
        total_try_catch_weight += try_catch_weight

        thread_weight = thread_weights.get(line_number, 0)
        total_thread_weight += thread_weight

        size += line_size

        current_inheritance_sum += total_inheritance

        # Calculate control structure complexity
        control_structure_complexity += calculate_control_structure_complexity(line)

        cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
        total_cbo_weight = sum(cbo_weights.values())
        total_cbo += total_cbo_weight

        mpc_weight = mpc_line_data.get(line_number, 0)
        total_mpc += mpc_weight

        # Calculate nesting level and control structures
        # current_nesting, _, control_structure_stack, wn = calculate_nesting_level(
        #     line, current_nesting, False, control_structure_stack
        # )
        # wn =calculate_nesting_level(line)
        # nesting_level += wn

        # Compound condition weight
        compound_condition_weight += calculate_compound_condition_weight(line)

        # Try-catch weight
        # try_catch_weight += calculate_try_catch_weight(line, current_nesting)

        # Thread weight
        # thread_weight += calculate_thread_weight(line)

    # Sum up the complexity metrics for this method
    total_complexity = (
            size + control_structure_complexity + total_nesting + current_inheritance_sum +
            compound_condition_weight + total_try_catch_weight + total_thread_weight + total_cbo + total_mpc
    )

    print("current_inheritance_sum", current_inheritance_sum)
    return {
        "size": size,
        "control_structure_complexity": control_structure_complexity,
        "nesting_level": nesting_level,
        "inheritance_level": current_inheritance_sum,
        "compound_condition_weight": compound_condition_weight,
        "try_catch_weight": total_try_catch_weight,
        "thread_weight": total_thread_weight,
        'cbo_weights': total_cbo,
        'mpc_weight': total_mpc,
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
    total_mpc_weight = 0

    for line in data:
        total_size += line[2]
        total_control_structure_complexity += line[4]
        total_nesting_level += line[5]
        total_inheritance_level += line[6]
        total_compound_condition_weight += line[7]
        total_try_catch_weight += line[8]
        total_thread_weight += line[9]
        total_cbo_weight += line[10]
        total_mpc_weight += line[11]

    return {
        'Size': total_size,
        'Control Structure Complexity': total_control_structure_complexity,
        'Nesting Level': total_nesting_level,
        'Inheritance Level': total_inheritance_level,
        'Compound Condition Weight': total_compound_condition_weight,
        'Try-Catch Weight': total_try_catch_weight,
        'Thread Weight': total_thread_weight,
        'CBO': total_cbo_weight,
        'MPC': total_mpc_weight
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
