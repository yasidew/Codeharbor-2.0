import logging
import re
import matplotlib.pyplot as plt

# Global dictionary for tracking inheritance depth
inheritance_depth = {}


# Function to remove comments (both single-line and multi-line)
def remove_comments(line):
    line = re.sub(r'//.*', '', line)  # Remove single-line comments
    line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)  # Remove multi-line comments
    return line


# Function to calculate the size of a line (token count)
def calculate_size(line):
    line = remove_comments(line)

    # Exclude access modifiers and keywords that don't contribute to complexity
    line = re.sub(r'\b(public|private|protected|internal|static|else|const|readonly)\b', '', line)

    # Token patterns based on C# syntax
    token_pattern = r'''
        "[^"]*"                 # Strings inside double quotes
        | '[^']*'               # Strings inside single quotes
        | \+\+|--                # Pre/post increment/decrement (++i, --i, i++, i--)
        | \b(?:if|for|while|switch|case|default|catch)\b\s*\(  # Control structures
        | \b(?:int|float|double|char|bool|long|short|byte|string|void|decimal)\b  # Data types
        | &&|\|\|                # Logical operators
        | [\+\*/%=&|<>!~^]       # Operators
        | ==|>=|<=|!=            # Relational operators
        | -?\d+                  # Numbers
        | \.                     # Dot operator
        | ,                      # Comma
        | [a-zA-Z_]\w*           # Identifiers
    '''

    # Tokenize based on the pattern
    tokens = re.findall(token_pattern, line, re.VERBOSE)
    return len(tokens), tokens


# Function to calculate control structure complexity (Wc)
def calculate_control_structure_complexity(line):
    wc = len(re.findall(r'\bif\b|\bfor\b|\bwhile\b|\bswitch\b|\bcase\b', line)) * 1
    wc += len(re.findall(r'\bfor\b|\bwhile\b', line)) * 2  # Iterative constructs
    return wc


# Function to calculate nesting level for control structures
def calculate_nesting_level(line, current_nesting, in_control_structure, control_structure_stack):
    control_structures = ['if', 'else', 'for', 'while', 'switch', 'try', 'catch', 'finally']
    wn = 0

    # Detect opening of a control structure
    if any(cs in line for cs in control_structures):
        current_nesting += 1
        control_structure_stack.append(current_nesting)
        wn = current_nesting
        in_control_structure = True
        return current_nesting, in_control_structure, control_structure_stack, wn

    # Detect closing of a control structure
    if '}' in line and control_structure_stack:
        current_nesting = control_structure_stack.pop() - 1
        in_control_structure = len(control_structure_stack) > 0
        return current_nesting, in_control_structure, control_structure_stack, 0

    # No nesting change for regular statements
    wn = current_nesting if in_control_structure else 0
    return current_nesting, in_control_structure, control_structure_stack, wn


# Function to calculate inheritance level (Wi)
def calculate_inheritance_level(line, current_inheritance):
    if re.search(r'class\s+\w+\s*:\s*\w+', line):
        current_inheritance += 1
        print("current_inheritancec",current_inheritance)
    return current_inheritance


# Function to calculate weight for compound conditions
def calculate_compound_condition_weight(line):
    weight = 0
    simple_condition_pattern = r'\bif\s*\(.*?\)'
    compound_condition_pattern = r'\bif\s*\(.*?(?:&&|\|\|).*?\)'

    compound_conditions = re.findall(compound_condition_pattern, line)
    if compound_conditions:
        for condition in compound_conditions:
            logical_operators = len(re.findall(r'&&|\|\|', condition))
            weight += 1 + logical_operators  # Base weight + operators count
    else:
        simple_conditions = re.findall(simple_condition_pattern, line)
        weight += len(simple_conditions)  # Simple condition weight

    return weight


# Function to calculate try-catch-finally weights
def calculate_try_catch_weight(line, current_nesting_level):
    weight = 0
    if re.search(r'\btry\b', line):
        weight += 1 if current_nesting_level <= 1 else 2  # Outer or nested try weight
    elif re.search(r'\bcatch\b', line):
        weight += 1 if current_nesting_level <= 1 else 2
    elif re.search(r'\bfinally\b', line):
        weight += 2
    return weight


# Function to calculate thread operation weight
def calculate_thread_weight(line):
    weight = 0
    if re.search(r'new\s+Thread\(', line):
        weight += 2
    if re.search(r'synchronized\s*\(', line):
        weight += 3
    return weight


# Function to extract class references for CBO calculation
def extract_class_references(code):
    instantiation_pattern = r'new\s+([A-Z][\w]*)'
    inheritance_pattern = r'class\s+\w+\s*:\s*([\w\s,]+)'
    class_references = {}
    current_class = None

    lines = code.splitlines()
    for line in lines:
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            class_references[current_class] = {}

        if current_class:
            instantiations = re.findall(instantiation_pattern, line)
            for instantiated_class in instantiations:
                class_references[current_class][instantiated_class] = 2

            inheritance = re.findall(inheritance_pattern, line)
            for inherited_class in inheritance:
                class_references[current_class][inherited_class.strip()] = 2

    return class_references


# Function to calculate Coupling Between Objects (CBO)
def calculate_cbo(class_references):
    cbo_results = {}
    for class_name, references in class_references.items():
        cbo_results[class_name] = sum(references.values())
    return cbo_results


# Function to extract message passing interactions (MPC)
def extract_message_passing(code):
    simple_message_pattern = r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\('
    message_passing = {}
    current_class = None

    lines = code.splitlines()
    for line in lines:
        class_declaration = re.search(r'class\s+([A-Z][\w]*)', line)
        if class_declaration:
            current_class = class_declaration.group(1)
            message_passing[current_class] = {}

        if current_class:
            simple_messages = re.findall(simple_message_pattern, line)
            for _, method_name in simple_messages:
                message_passing[current_class][method_name] = message_passing[current_class].get(method_name, 0) + 1

    return message_passing


# Function to calculate Message Passing Coupling (MPC)
def calculate_mpc(message_passing):
    mpc_results = {}
    for class_name, messages in message_passing.items():
        mpc_results[class_name] = sum(messages.values())
    return mpc_results


# Function to track inheritance depth across files
def track_inheritance_depth_across_files(file_contents):
    global inheritance_depth
    for filename, content in file_contents.items():
        lines = content.splitlines()
        for line in lines:
            line = remove_comments(line)
            match_inheritance = re.search(r'class\s+(\w+)\s*:\s*([\w\s,]+)', line)
            match_base_class = re.search(r'class\s+(\w+)', line)

            if match_inheritance:
                class_name = match_inheritance.group(1)
                base_class_name = match_inheritance.group(2).split(',')[0].strip()
                inheritance_depth[class_name] = inheritance_depth.get(base_class_name, 1) + 1
            elif match_base_class and not re.search(r':', line):
                class_name = match_base_class.group(1)
                inheritance_depth[class_name] = 1


# Function to retrieve the inheritance level
def calculate_inheritance_level2(class_name):
    return inheritance_depth.get(class_name, 1)


# Function to calculate complexity line by line
def calculate_code_complexity_line_by_line_csharp(code, filename):
    lines = code.splitlines()
    current_nesting = 0
    control_structure_stack = []
    in_control_structure = False
    current_inheritance = 0

    track_inheritance_depth_across_files({filename: code})
    line_complexities = []

    for i, line in enumerate(lines, start=1):
        size, tokens = calculate_size(line)
        wc = calculate_control_structure_complexity(line)
        current_inheritance = calculate_inheritance_level(line, current_inheritance)
        current_nesting, in_control_structure, control_structure_stack, wn = calculate_nesting_level(
            line, current_nesting, in_control_structure, control_structure_stack
        )

        # class_name_match = re.search(r'class\s+(\w+)', line)

        line_complexities.append({
            'line_number': i,
            'line_content': line.strip(),
            'size': size,
            'tokens': tokens,
            'control_structure_complexity': wc,
            'nesting_level': current_nesting,
            'inheritance_level': current_inheritance,
            'compound_condition_weight': calculate_compound_condition_weight(line),
            'try_catch_weight': calculate_try_catch_weight(line, current_nesting),
            'thread_weight': calculate_thread_weight(line),
        })

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
        'cbo': cbo_results,
        'mpc':mpc_results
    }

    # return line_complexities


# Function to calculate complexity across multiple files
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

        # Split content into lines
        lines = content.splitlines()
        complexity_data = []

        # Extract class references and message passing for MPC and CBO
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)

        # Calculate MPC and CBO for this file
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        for line_number, line in enumerate(lines, start=1):
            # Calculate size (token count)
            size, tokens = calculate_size(line)

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

        # results[filename] = complexity_data
        results[filename] = {
            'complexity_data': complexity_data,
            'cbo': cbo_value,
            'mpc': mpc_value
        }

        # Calculate contributing factors and plot pie chart
        # complexity_factors = calculate_complexity_factors(filename, complexity_data)
        # plot_complexity_pie_chart(filename, complexity_factors)

    return results


# Function to calculate overall complexity factors
def calculate_complexity_factors(data):
    totals = {key: 0 for key in ['Size', 'Control Structure Complexity', 'Nesting Level', 'Inheritance Level',
                                 'Compound Condition Weight', 'Try-Catch Weight', 'Thread Weight']}

    for line in data:
        totals['Size'] += line['size']
        totals['Control Structure Complexity'] += line['control_structure_complexity']
        totals['Nesting Level'] += line['nesting_level']
        totals['Inheritance Level'] += line['inheritance_level']
        totals['Compound Condition Weight'] += line['compound_condition_weight']
        totals['Try-Catch Weight'] += line['try_catch_weight']
        totals['Thread Weight'] += line['thread_weight']

    return totals


# Function to plot complexity factors as a pie chart
def plot_complexity_pie_chart(filename, complexity_factors):
    labels = list(complexity_factors.keys())
    sizes = list(complexity_factors.values())

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Code Complexity Contribution for {filename}')
    plt.axis('equal')
    plt.show()
