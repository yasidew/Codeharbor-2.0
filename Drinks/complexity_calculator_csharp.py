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
def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
    control_structures = ['if', 'else', 'for', 'while', 'switch', 'try', 'catch']
    control_regex = re.compile(r'\b(?:' + '|'.join(control_structures) + r')\b')

    wn = 0

    # Detect the start of a control structure
    if control_regex.search(line):
        if 'else if' in line or 'else if' in line.replace(' ', ''):  # Handle 'else if'
            wn = current_nesting
        elif 'else' in line and 'if' not in line:  # Handle 'else' (not 'else if')
            wn = current_nesting
        else:
            current_nesting += 1
            control_structure_stack.append(current_nesting)
            wn = current_nesting

        in_control_structure = True
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Detect the end of a control structure
    if '}' in line and control_structure_stack:
        control_structure_stack.pop()
        current_nesting = len(control_structure_stack)
        wn = current_nesting
        in_control_structure = len(control_structure_stack) > 0
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Handle sequential statements
    if in_control_structure:
        wn = current_nesting
    else:
        wn = 0  # Reset weight outside control structures

    return current_nesting, in_control_structure, control_structure_stack, wn


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
def calculate_try_catch_weight(line, current_nesting_level):
    weight = 0

    # Detect try blocks
    if re.search(r'\btry\b', line):
        if current_nesting_level > 1:
            weight += 2  # Inner try block weight
        else:
            weight += 1  # Outer try block weight

    # Detect catch blocks
    elif re.search(r'\bcatch\b', line):
        if current_nesting_level > 1:
            weight += 2  # Inner catch block weight
        else:
            weight += 1  # Outer catch block weight

    # Detect finally block
    elif re.search(r'\bfinally\b', line):
        weight += 2  # Weight of 2 for the finally block

    return weight


# Function to calculate weight for thread-related operations
def calculate_thread_weight(line):
    weight = 0

    # Detect thread creation
    if re.search(r'new\s+Thread\(', line):
        weight += 2  # Weight of 2 for thread creation

    # Detect thread synchronization
    if re.search(r'lock\s*\(', line):  # C# equivalent of synchronized
        weight += 5  # Weight of 5 for lock blocks

    return weight


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



def calculate_code_complexity_multiple_files_csharp(file_contents):
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

        print("content...............................", content)

        # Split content into lines
        lines = content.splitlines()
        complexity_data = []

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        # print("Method complex",method_complexities)

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
            sys.stdout.write(f"Your message\n ${size}")

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
            method_inheritance[class_name] = current_inheritance

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
                wn,
                current_inheritance,
                compound_condition_weight,
                try_catch_weight,
                thread_weight,
                total_complexity,
            ])
        # method_complexities = calculate_code_complexity_by_method(content, method_inheritance, class_name)

        # Calculate contributing factors and plot pie chart
        complexity_factors = calculate_complexity_factors(filename, complexity_data)
        pie_chart_path = plot_complexity_pie_chart(filename, complexity_factors)

        # results[filename] = complexity_data
        results[filename] = {
            'complexity_data': complexity_data,
            'cbo': cbo_value,
            'mpc': mpc_value,
            # 'method_complexities': method_complexities,
            # 'recommendation': filtered_recommendations,
            'pie_chart_path': pie_chart_path,
            'total_wcc': total_wcc
        }

    return results


# Function to calculate overall complexity factors
# def calculate_complexity_factors(data):
#     totals = {key: 0 for key in ['Size', 'Control Structure Complexity', 'Nesting Level', 'Inheritance Level',
#                                  'Compound Condition Weight', 'Try-Catch Weight', 'Thread Weight']}
#
#     for line in data:
#         totals['Size'] += line['size']
#         totals['Control Structure Complexity'] += line['control_structure_complexity']
#         totals['Nesting Level'] += line['nesting_level']
#         totals['Inheritance Level'] += line['inheritance_level']
#         totals['Compound Condition Weight'] += line['compound_condition_weight']
#         totals['Try-Catch Weight'] += line['try_catch_weight']
#         totals['Thread Weight'] += line['thread_weight']
#
#     return totals

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
