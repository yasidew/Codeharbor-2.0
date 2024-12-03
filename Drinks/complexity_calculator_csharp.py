import logging
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

# Global dictionary for tracking inheritance depth
inheritance_depth = {}

# Global flag to detect when the class declaration ends
class_declaration_ended = False


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
    wc += len(re.findall(r'\bfor\b|\bwhile\b|\bdo\b', line)) * 2  # Iterative structures
    wc += len(re.findall(r'\bcase\b', line))  # Switch cases
    wc += len(re.findall(r'\btry\b', line))  # Try block
    return wc


# Function to calculate nesting level complexity (Wn)
# def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
#     control_structures = ['if', 'else', 'for', 'while', 'switch', 'try', 'catch']
#     control_regex = re.compile(r'\b(?:' + '|'.join(control_structures) + r')\b')
#
#     wn = 0
#
#     # Detect the start of a control structure
#     if control_regex.search(line):
#         if 'else if' in line or 'else if' in line.replace(' ', ''):  # Handle 'else if'
#             wn = current_nesting
#         elif 'else' in line and 'if' not in line:  # Handle 'else' (not 'else if')
#             wn = current_nesting
#         else:
#             current_nesting += 1
#             control_structure_stack.append(current_nesting)
#             wn = current_nesting
#
#         in_control_structure = True
#         return current_nesting, in_control_structure, control_structure_stack, wn
#
#     # Detect the end of a control structure
#     if '}' in line and control_structure_stack:
#         control_structure_stack.pop()
#         current_nesting = len(control_structure_stack)
#         wn = current_nesting
#         in_control_structure = len(control_structure_stack) > 0
#         return current_nesting, in_control_structure, control_structure_stack, wn
#
#     # Handle sequential statements
#     if in_control_structure:
#         wn = current_nesting
#     else:
#         wn = 0  # Reset weight outside control structures
#
#     return current_nesting, in_control_structure, control_structure_stack, wn

def calculate_nesting_level(java_code):
    """
    Calculates the nesting level of control structures (if-else, for, while, do-while)
    in Java code line by line, ignoring class and method declarations.
    """
    # Remove comments from the Java code
    java_code = remove_comments(java_code)
    lines = java_code.splitlines()

    # Control structure keywords to consider
    control_keywords = ['if', 'else', 'for', 'while', 'do']

    # State variables to track nesting level
    current_nesting = 0
    control_structure_stack = []  # Stack to track nesting levels
    nesting_levels = []

    # Regular expressions for detecting control structures
    control_regex = re.compile(r'\b(if|else if|else|for|while|do)\b')
    open_brace_regex = re.compile(r'\{')  # Opening brace
    close_brace_regex = re.compile(r'\}')  # Closing brace

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        # Skip empty lines
        if not stripped_line:
            nesting_levels.append((line_no, stripped_line, current_nesting))
            continue

        # Check if the line contains a control structure
        control_match = control_regex.search(stripped_line)
        if control_match:
            # Increment nesting level for a new control structure
            current_nesting += 1
            control_structure_stack.append(current_nesting)

        # Record the current nesting level for the line
        nesting_levels.append((line_no, stripped_line, current_nesting))

        # Adjust nesting level based on braces
        # Count the opening and closing braces in the line
        opening_braces = len(open_brace_regex.findall(stripped_line))
        closing_braces = len(close_brace_regex.findall(stripped_line))

        # Adjust the nesting level for each closing brace
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
    weight = 0

    # Detect simple conditions
    simple_condition_pattern = r'\bif\s*\(.*?\)'

    # Detect compound conditions with logical operators (&&, ||)
    compound_condition_pattern = r'\bif\s*\(.*?(?<!==)(?:&&|\|\|).*?\)'

    compound_conditions = re.findall(compound_condition_pattern, line)
    if compound_conditions:
        for condition in compound_conditions:
            logical_operators = len(re.findall(r'&&|\|\|', condition))
            weight += 1 + logical_operators  # Base condition + number of logical operators
    else:
        simple_conditions = re.findall(simple_condition_pattern, line)
        if simple_conditions:
            weight += 1  # Simple condition weight

    return weight


# Function to calculate weight for try-catch-finally blocks
# def calculate_try_catch_weight(line, current_nesting_level):
#     weight = 0
#
#     # Detect try blocks
#     if re.search(r'\btry\b', line):
#         if current_nesting_level > 1:
#             weight += 2  # Inner try block weight
#         else:
#             weight += 1  # Outer try block weight
#
#     # Detect catch blocks
#     elif re.search(r'\bcatch\b', line):
#         if current_nesting_level > 1:
#             weight += 2  # Inner catch block weight
#         else:
#             weight += 1  # Outer catch block weight
#
#     # Detect finally block
#     elif re.search(r'\bfinally\b', line):
#         weight += 2  # Weight of 2 for the finally block
#
#     return weight

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


# Function to calculate weight for thread-related operations
# def calculate_thread_weight(line):
#     weight = 0
#
#     # Detect thread creation
#     if re.search(r'new\s+Thread\(', line):
#         weight += 2  # Weight of 2 for thread creation
#
#     # Detect thread synchronization
#     if re.search(r'lock\s*\(', line):  # C# equivalent of synchronized
#         weight += 5  # Weight of 5 for lock blocks
#
#     return weight

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
    Extracts class references for Coupling Between Objects (CBO) calculation in C# code.
    """
    # Patterns to match C# constructs
    instantiation_pattern = r'new\s+([A-Z][\w]*)'  # Object instantiation
    inheritance_pattern = r'class\s+\w+\s*:\s*([A-Z][\w]*)'  # Inheritance (base class after ':')
    interface_pattern = r'class\s+\w+\s*:\s*.*\s*,\s*([A-Z][\w]*)'  # Interface implementation
    constructor_pattern = r'\bpublic\s+\w+\s*\(([^)]*)\)'  # Constructor parameter classes
    method_pattern = r'\b(public|private|protected|internal)?\s+(\w+|<[^>]+>)\s+(\w+)\s*\(([^)]*)\)'  # Method signature

    # Dictionary to store class references per class
    class_references = {}
    current_class = None

    # Split code into lines and process each
    lines = csharp_code.splitlines()
    for line in lines:
        line = remove_comments(line)

        # Detect current class declaration
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            if current_class not in class_references:
                class_references[current_class] = {}
            logging.info(f"Detected class: {current_class}")

        if current_class:
            # Detect object instantiations
            instantiations = re.findall(instantiation_pattern, line)
            for instantiated_class in instantiations:
                class_references[current_class][instantiated_class] = 2  # Tight coupling

            # Detect inheritance
            inheritance = re.findall(inheritance_pattern, line)
            for inherited_class in inheritance:
                class_references[current_class][inherited_class] = 2  # Tight coupling

            # Detect interface implementations
            interfaces = re.findall(interface_pattern, line)
            for implemented_interface in interfaces:
                class_references[current_class][implemented_interface] = 1  # Loose coupling

            # Detect method signatures and parameters
            method_match = re.search(method_pattern, line)
            if method_match:
                parameters = method_match.group(4)  # Extract parameters
                param_classes = re.findall(r'\b([A-Z][\w]*)\b', parameters)
                for param_class in param_classes:
                    class_references[current_class][param_class] = 2  # Tight coupling

            # Detect constructor parameters
            constructor_matches = re.findall(constructor_pattern, line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)  # Extract parameter classes
                for param_class in param_classes:
                    class_references[current_class][param_class] = 1  # Loose coupling

    logging.info("Finished extracting class references.")
    return class_references


def calculate_cbo(class_references):
    """
    Calculates Coupling Between Objects (CBO) from class references.
    """
    cbo_results = {}
    for class_name, references in class_references.items():
        cbo_results[class_name] = sum(references.values())
        logging.info(f"CBO for {class_name}: {cbo_results[class_name]}")
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


# Refined method pattern for accurate method detection
method_pattern = re.compile(
    r'^\s*(public|private|protected)?\s*(static\s+)?'  # Access modifiers and 'static' keyword
    r'(\w+\s+)?'  # Optional return type (including void)
    r'(\w+)\s*\([^)]*\)\s*\{'  # Method name and parameter list
)

# Keywords to ignore to prevent detecting control structures as methods
control_keywords = {'if', 'for', 'while', 'switch', 'catch'}

# Function to calculate complexity for each method in a C# file
def calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data):
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


# Helper function to calculate complexity for a C# method based on lines of code
def calculate_complexity_for_method(lines, method_inheritance, class_name, nesting_level_dict, try_catch_weight_dict, thread_weights, cbo_line_data, mpc_line_data):
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

        print("total_nesting))))))))))))))))))))))))))))))))))", total_nesting)

        try_catch_weight = try_catch_weight_dict.get(line_number, 0)
        total_try_catch_weight += try_catch_weight

        thread_weight = thread_weights.get(line_number, 0)
        total_thread_weight += thread_weight

        size += line_size
        current_inheritance_sum += total_inheritance

        # Calculate control structure complexity
        control_structure_complexity += calculate_control_structure_complexity(line)

        # Calculate nesting level and control structures
        # current_nesting, _, control_structure_stack, wn = calculate_nesting_level(
        #     line, current_nesting, False, control_structure_stack
        # )
        # nesting_level += wn

        cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
        total_cbo_weight = sum(cbo_weights.values())
        total_cbo += total_cbo_weight

        mpc_weight = mpc_line_data.get(line_number, 0)
        total_mpc += mpc_weight

        # Compound condition weight
        compound_condition_weight += calculate_compound_condition_weight(line)

        # Try-catch weight
        # try_catch_weight += calculate_try_catch_weight(line, current_nesting)

        # Thread weight (specific to threading constructs in C#)
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
        "try_catch_weight": try_catch_weight,
        "thread_weight": thread_weight,
        'cbo_weights': total_cbo,
        'mpc_weight': total_mpc,
        "total_complexity": total_complexity
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_code_complexity_multiple_files_csharp(file_contents):
    results = {}

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

        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        # print("Method complex",method_complexities)

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
        print("try_catch_weights Nesting Levels-++++++++++++++++++++++++++---------------------------------",
              try_catch_weights)

        thread_weights = calculate_thread_weight(content)

        print("thread_weights---------------------------------", thread_weights)

        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare line_complexities for recommendation engine
        line_complexities = []

        method_inheritance = {}

        # Initialize total WCC value for the file
        total_wcc = 0

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
                mpc_weight,
                total_complexity,
            ])
        # Calculate method complexities
        method_complexities = calculate_code_complexity_by_method(content, method_inheritance, class_name, cbo_line_data, mpc_line_data)

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
            # 'recommendation': filtered_recommendations,
            'pie_chart_path': pie_chart_path,
            'bar_charts': bar_chart_paths,
            'total_wcc': total_wcc
        }

    return results

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