import json
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import javalang
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBClassifier

from Drinks.process_java_file_cbo import process_java_files, extract_cbo_features1


class CBOMetrics:
    def __init__(self, java_code):
        """
        Parses Java code and extracts:
        - Constructor & Setter Injection (Weight = 1)
        - Direct Object Instantiation (new Keyword) (Weight = 3)
        - Static Method Calls (Weight = 1)
        - Static Variable Usages (Weight = 1)
        """
        self.tree = javalang.parse.parse(java_code)
        self.dependencies = set()
        self.primitive_types = {"byte", "short", "int", "long", "float", "double", "boolean", "char"}
        # Updated built-in classes set (all these are considered built-in and will be skipped)
        self.built_in_classes = {
            "Thread", "String", "HashMap", "BufferedReader", "FileReader", "StopWatch", "DuplicateRecordException",
            "Runnable", "System.out", "System.err", "Logger", "Math",
            "InputStreamReader", "URL", "StringBuilder", "LinkedList", "HashSet",
            "Stack", "Properties", "FileInputStream", "Random", "Exception",
            "RuntimeException", "ArithmeticException", "System", "Integer", "Double", "Float", "Boolean", "Character",
            "Long", "Short", "Byte", "BigInteger", "BigDecimal", "Object", "Collections",
            "Arrays", "List", "ArrayList", "HashMap", "HashSet", "LinkedList", "Map", "Set", "Sort", "PageRequest",
            "UUID", "Files", "Paths", "Runtime", "Optional", "Objects", "LocalDate", "Year", "IntStream", "Scanner",
            "NumberFormatException", "RecordNotFoundException", "BadRequestException", "InputStreamResource",
            "InvalidFileTypeException", "IllegalArgumentException", 'ArrayList', 'Hashtable', 'Queue', 'Stack',
            'SortedList', 'List', 'Dictionary', 'SortedDictionary', 'SortedList', 'Queue', 'Stack',
            'HashSet', 'SortedSet', 'ConcurrentBag', 'ConcurrentQueue', 'ConcurrentStack', 'ConcurrentDictionary', "Timer", "Random","ExecutorService", "URL"
        }
        self.constructor_injections = {}
        self.setter_injections = {}
        self.direct_instantiations = []
        self.assignment_weights = []
        self.static_method_calls = []
        self.static_variable_usages = []

    def extract_dependencies(self):
        """
        Extracts constructor injections, setter injections, direct instantiations,
        static method calls, and static variable usages.
        """
        # --- Constructor Injection (DI, Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.ConstructorDeclaration):
            constructor_line = node.position.line if node.position else "Unknown"
            for index, param in enumerate(node.parameters):
                param_name = param.name
                param_type = param.type.name
                if param_type not in self.primitive_types and param_type not in self.built_in_classes:
                    self.dependencies.add(param_type)
                    assignment_line = constructor_line + index + 1
                    self.constructor_injections[param_type] = {
                        "type": param_type,
                        "assignment": f"this.{param_name} = {param_name};",
                        "weight": 1,
                        "constructor_line": constructor_line
                    }
                    self.assignment_weights.append({
                        "line": assignment_line,
                        "statement": f"this.{param_name} = {param_name};",
                        "weight": 1
                    })

        # --- Setter Injection (DI, Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.MethodDeclaration):
            if node.name.startswith("set") and node.parameters:
                setter_line = node.position.line if node.position else "Unknown"
                param_name = node.parameters[0].name
                param_type = node.parameters[0].type.name
                if param_type not in self.primitive_types and param_type not in self.built_in_classes:
                    self.dependencies.add(param_type)
                    self.setter_injections[param_type] = {
                        "type": param_type,
                        "setter_method": f"public void {node.name}({param_type} param) {{\n    this.{param_name} = param;\n}}",
                        "weight": 1,
                        "setter_line": setter_line
                    }
                    self.assignment_weights.append({
                        "line": setter_line + 1,
                        "statement": f"this.{param_name} = {param_name};",
                        "weight": 1
                    })

        # --- Direct Object Instantiations (new Keyword, Weight = 3) ---
        for path, node in self.tree.filter(javalang.tree.ClassCreator):
            class_name = node.type.name
            instantiation_line = "Unknown"
            for ancestor in reversed(path):
                if hasattr(ancestor, 'position') and ancestor.position:
                    instantiation_line = ancestor.position.line
                    break
            # Only add if it's not a primitive or built-in class
            if class_name not in self.primitive_types and class_name not in self.built_in_classes:
                self.dependencies.add(class_name)
                self.direct_instantiations.append({
                    "line": instantiation_line,
                    "instantiation": f"new {class_name}();",
                    "weight": 3
                })

        # --- Static Method Calls (Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.MethodInvocation):
            if hasattr(node, "qualifier") and node.qualifier:
                qualifier = str(node.qualifier)
                # Only consider if the qualifier starts with uppercase and is NOT a built-in class
                if qualifier and qualifier[0].isupper() and qualifier not in self.built_in_classes:
                    self.dependencies.add(qualifier)
                    method_call_line = node.position.line if node.position else "Unknown"
                    self.static_method_calls.append({
                        "line": method_call_line,
                        "method_call": f"{qualifier}.{node.member}()",
                        "weight": 1
                    })

        # --- Static Variable Usages (Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.MemberReference):
            if hasattr(node, "qualifier") and node.qualifier:
                qualifier = str(node.qualifier)
                if qualifier and qualifier[0].isupper() and qualifier not in self.built_in_classes:
                    self.dependencies.add(qualifier)
                    variable_usage_line = node.position.line if node.position else "Unknown"
                    self.static_variable_usages.append({
                        "line": variable_usage_line,
                        "variable_usage": f"{qualifier}.{node.member}",
                        "weight": 1
                    })

    def calculate_cbo(self):
        """
        Calculates the Coupling Between Objects (CBO) value based on unique dependencies.
        """
        cbo_value = len(self.dependencies)
        return (cbo_value,
                self.constructor_injections,
                self.setter_injections,
                self.direct_instantiations,
                self.assignment_weights,
                self.static_method_calls,
                self.static_variable_usages)

    def get_cbo_report(self):
        """
        Extracts dependencies, calculates CBO, and returns a structured report.
        """
        self.extract_dependencies()
        (cbo_value,
         constructor_injections,
         setter_injections,
         direct_instantiations,
         assignment_weights,
         static_method_calls,
         static_variable_usages) = self.calculate_cbo()

        report = {
            "CBO Score": cbo_value,
            "Constructor Injections": [
                f"Line {info['constructor_line']}: {info['type']} DI (Weight: {info['weight']})"
                for info in constructor_injections.values()
            ],
            "Setter Injections": [
                f"Line {info['setter_line']}: {info['type']} DI (Weight: {info['weight']})"
                for info in setter_injections.values()
            ],
            "Direct Object Instantiations": [
                f"Line {info['line']}: {info['instantiation']} (Weight: {info['weight']})"
                for info in direct_instantiations
            ],
            "Dependency Assignment Weights": [
                f"Line {info['line']}: {info['statement']} (Weight: {info['weight']})"
                for info in assignment_weights
            ],
            "Static Method Calls": [
                f"Line {info['line']}: {info['method_call']} (Weight: {info['weight']})"
                for info in static_method_calls
            ],
            "Static Variable Usages": [
                f"Line {info['line']}: {info['variable_usage']} (Weight: {info['weight']})"
                for info in static_variable_usages
            ]
        }
        return report


output_csv = "media/cbo_features_output.csv"
model_output = "xgboost_java_model.pkl"

if os.path.exists(model_output):
    model = joblib.load(model_output)
    # print(f"✅ Loaded existing model java cbo from {model_output}")
else:
    # Load dataset
    df = pd.read_csv(output_csv)

    # Drop 'file_name' column as it's not a feature
    df.drop(columns=["file_name"], inplace=True)

    # Define features (X) and labels (y)
    X = df.drop(columns=["cbo_label"])
    y = df["cbo_label"]

    # Split data into training and testing sets
    if len(X) < 2:
        print("Warning: Not enough data for train-test split. Training on the entire dataset.")
        X_train, y_train = X, y  # Use all data for training
        X_test, y_test = X, y  # Empty test set (or keep a single instance)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the XGBoost model
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained model
    joblib.dump(model, model_output)
    print(f"Model saved as {model_output}")

JAVA_STANDARD_CLASSES = {
    "Thread", "String", "HashMap", "BufferedReader", "FileReader", "Runnable", "System.out", "Logger", "Math"
                                                                                                       "InputStreamReader",
    "URL", "StringBuilder", "LinkedList", "HashSet",
    "Stack", "Properties", "FileInputStream", "Random", "Exception",
    "RuntimeException", "ArithmeticException", "String", "System", "Math", "Integer", "Double", "Float", "Boolean",
    "Character",
    "Long", "Short", "Byte", "BigInteger", "BigDecimal", "Object", "Collections",
    "Arrays", "List", "ArrayList", "HashMap", "HashSet", "LinkedList", "Map", "Set", "Sort", "PageRequest", "UUID",
    "TreeMap", "Files", "Paths", "Runtime", "Optional", "Objects", "LocalDate", "Year", "IntStream", "Scanner"
}


def extract_cbo_features(java_code):
    """
    Extracts software quality metrics related to CBO (Coupling Between Objects) for Java.
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

    try:
        tree = javalang.parse.parse(java_code)

        for path, node in tree:
            # ✅ Detect Direct Instantiations (new ClassName())
            if isinstance(node, javalang.tree.ClassCreator):
                if hasattr(node, "type") and node.type.name:
                    class_name = node.type.name

                    if class_name in JAVA_STANDARD_CLASSES:
                        continue

                    cbo_features["class_dependencies"].add(class_name)
                    cbo_features["direct_instantiations"] += 1
                    print(f"Detected direct instantiation of>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {class_name}")

            # ✅ Detect Constructor Injections
            if isinstance(node, javalang.tree.ConstructorDeclaration):
                for param in node.parameters:
                    if hasattr(param, "type") and hasattr(param.type, "name"):
                        class_name = param.type.name
                        cbo_features["class_dependencies"].add(class_name)
                        cbo_features["constructor_injections"] += 1

            # ✅ Detect Setter Injections (Methods starting with "set")
            if isinstance(node, javalang.tree.MethodDeclaration):
                if "set" in node.name.lower():
                    for param in node.parameters:
                        if hasattr(param, "type") and hasattr(param.type, "name"):
                            class_name = param.type.name
                            cbo_features["class_dependencies"].add(class_name)
                            cbo_features["setter_injections"] += 1

            # ✅ Detect Static Method Calls (ClassName.methodName())
            if isinstance(node, javalang.tree.MethodInvocation):
                if hasattr(node, "qualifier") and node.qualifier:
                    class_name = node.qualifier
                    if class_name[
                        0].isupper() and class_name not in JAVA_STANDARD_CLASSES:  # Ignore Java built-in classes
                        cbo_features["class_dependencies"].add(class_name)
                        cbo_features["static_method_calls"] += 1
                        print(f"Detected static method call >>> {class_name}")

            # ✅ Detect Static Variable Usages (ClassName.variableName)
            if isinstance(node, javalang.tree.MemberReference):
                if hasattr(node, "qualifier") and node.qualifier:
                    class_name = node.qualifier
                    if class_name[0].isupper():  # Ensure it's a class reference
                        cbo_features["class_dependencies"].add(class_name)
                        cbo_features["static_variable_usage"] += 1

            # ✅ Detect Implemented Interfaces
            if isinstance(node, javalang.tree.ClassDeclaration):
                if node.implements:
                    for implemented_interface in node.implements:
                        if hasattr(implemented_interface, "name"):
                            cbo_features["class_dependencies"].add(implemented_interface.name)
                            cbo_features["interface_implementations"] += 1

            # ✅ Detect Method Parameters as Dependencies (NEW FIX)
            if isinstance(node, javalang.tree.MethodDeclaration):
                for param in node.parameters:
                    if hasattr(param, "type") and hasattr(param.type, "name"):
                        class_name = param.type.name
                        cbo_features["class_dependencies"].add(class_name)

    except Exception as e:
        print(f"Error parsing Java code: {e}")

    # Convert class_dependencies from a set to a count
    cbo_features["class_dependencies"] = len(cbo_features["class_dependencies"])
    return cbo_features


json_file = "media/java_code_dataset.json"
output_csv1 = "media/cbo_features_output.csv"
process_java_files(json_file, output_csv1)


def load_existing_java_dataset():
    """
    Loads existing Java dataset from JSON file. If file does not exist, creates an empty list.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Return empty list if JSON file is corrupted
    return []


def save_java_dataset(java_data):
    """
    Saves the updated Java dataset back to the JSON file.
    """
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(java_data, file, indent=4)


def add_java_code(file_name, java_code):
    """
    Adds new Java code to the dataset and retrains the model.
    """
    java_data = load_existing_java_dataset()

    # **Check for duplicates**
    for entry in java_data:
        if entry["java_code"].strip() == java_code.strip():
            print(f"⚠️ Duplicate Java code detected for {file_name}. Skipping...")
            return

    # **Append new Java code to dataset*
    java_data.append({"file_name": file_name, "java_code": java_code})
    save_java_dataset(java_data)

    print(f"✅ New Java code added: {file_name}")
    process_java_files(json_file, output_csv1)


def get_code_recommendations(java_code, model_path):
    """
    Predicts code  and provides recommendations.
    """
    # Load trained model
    model = joblib.load(model_path)

    # Extract CBO features
    features = extract_cbo_features1(java_code)

    # Convert features into DataFrame
    feature_df = pd.DataFrame([features])

    # Predict complexity
    prediction = model.predict(feature_df)[0]

    # Generate recommendations
    recommendations = []

    if features["direct_instantiations"] >= 5:
        recommendations.append("⚠️ Too many direct object instantiations. Use dependency injection instead.")
    if features["static_method_calls"] >= 5:
        recommendations.append("⚠️ Reduce static method calls to improve testability and flexibility.")
    if features["static_variable_usage"] >= 3:
        recommendations.append("⚠️ Minimize static variable usage to prevent global state issues.")
    # if features["setter_injections"] > 1:
    #     recommendations.append("⚠️ Too many setter injections detected. Prefer constructor injection.")
    if features["interface_implementations"] > 2:
        recommendations.append("⚠️ Avoid God Interfaces (interfaces with too many responsibilities).")
    # if features["global_variable_references"] > 2:
    #     recommendations.append("⚠️ Avoid global variables. Use dependency injection or encapsulation instead.")

    # Output results
    result = {
        "prediction": "High CBO (Issue)" if prediction == 1 else "Low CBO (Good)",
        "recommendations": recommendations
    }

    return result


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


def remove_comments(line):
    return re.sub(r'//.*|/\*.*?\*/', '', line)  # Remove single-line and multi-line comments


def calculate_size(line):
    global class_declaration_ended

    line = remove_comments(line)

    # ✅ Skip import and package statements
    if re.match(r'^\s*(import|package)\s+[a-zA-Z0-9_.\*]+;', line):
        return 0, []  # Ignore imports and package declarations

    # ✅ Check for class declaration and skip until it's completed
    if re.search(r'\bclass\b\s+\w+', line):
        class_declaration_ended = True
        return 0, []  # Ignore class declaration and set size to 0 # Still inside class declaration

    # ✅ Remove annotations like @Override
    line = re.sub(r'@\w+', '', line)

    # ✅ Remove access modifiers (public, private, protected, static)
    line = re.sub(r'\b(public|private|protected|default|static)\b', '', line)

    # ✅ Handle Variable Declaration vs. Definition
    variable_declaration_pattern = r'\b(?:int|float|double|char|boolean|long|short|byte|String)\b\s+([a-zA-Z_]\w*)(\s*,\s*[a-zA-Z_]\w*)*;'
    variable_definition_pattern = r'\b(?:int|float|double|char|boolean|long|short|byte|String)\b\s+([a-zA-Z_]\w*)\s*=\s*[^;]+;'

    if re.search(variable_declaration_pattern, line):
        line = re.sub(r'(\b(?:int|float|double|char|boolean|long|short|byte|String)\b)\s+[a-zA-Z_]\w*', r'\1', line)

    elif re.search(variable_definition_pattern, line):
        pass  # Keep variable names in definitions

    # ✅ Handle user-defined class variables (DatabaseService dbService; → Only "DatabaseService")
    user_defined_class_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\s+[a-zA-Z_]\w*;'
    line = re.sub(user_defined_class_pattern, r'\1', line)

    # ✅ Tokenize method signatures correctly
    method_signature_regex = (
        r'\b(?:public|private|protected)?\s*'
        r'(?:static\s+)?'
        r'(?:synchronized\s+)?'
        r'([\w<>\[\]]+)\s+'
        r'(?!set)(\w+)\s*\([^)]*\)\s*'
        r'(?:throws\s+[\w<>\[\]]+(?:\s*,\s*[\w<>\[\]]+)*)?\s*'
        r'\{'
    )
    line = re.sub(method_signature_regex, r'\1 \2()', line)

    constructor_regex = (
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{'
    )

    # Replace constructors with class names only (ignoring parameters)
    line = re.sub(constructor_regex, r'\1()', line)

    # ✅ Handle system calls
    system_call_pattern = r'\b(System)\s*\.\s*(out|err)\s*\.\s*([a-zA-Z_]\w*)\s*\('
    line = re.sub(system_call_pattern, r'\1 , . , \2 , . , \3 (', line)

    # ✅ Handle user-defined method calls
    user_defined_call_pattern = r'\b([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\(.*?\)'
    line = re.sub(user_defined_call_pattern, r'\1 . \2()', line)

    # ✅ Handle logger calls
    logger_call_pattern = r'\b(logger|Logger)\s*\.\s*([a-zA-Z_]\w*)\s*\("(.*?)"\)'
    line = re.sub(logger_call_pattern, r'\1 . \2 "\3"', line)

    # ✅ Handle Generics (List<String>, HashMap<String, Integer>)
    generic_type_pattern = r'\b(?:List|Set|Map|HashMap|TreeMap|LinkedList|ArrayList|Queue|Deque)\s*<\s*[\w\s,<>]+\s*>'
    generics = re.findall(generic_type_pattern, line)

    for generic in generics:
        line = line.replace(generic, generic.replace(" ",
                                                     ""))  # Remove spaces in generics (e.g., "Map<String, Integer>" -> "Map<String,Integer>")

    # ✅ Token Patterns (Updated for WCC Compliance)
    token_pattern = r'''
        "[^"]*"                 # Strings inside double quotes
        | '[^']*'               # Strings inside single quotes
        | \+\+|--               # Pre and post increment/decrement
        | \belse\s+if\b         # "else if" as one token
        | \bif\b                # "if" should be counted
        | \b(?:for|while|switch|case|default|catch)\b  # Control structures
        | \b(?:int|float|double|char|boolean|long|short|byte|void|String)\b  # Data types
        | &&|\|\||===|==|!==|>=|<=|!=   # Logical and comparison operators as one token
        | [\+\*/%=&|<>!~^]      # Operators
        | -?\d+                 # Numbers
        | \.                    # Dot operator
        | \b[a-zA-Z_]\w*\(\)    # Method calls
        | \b[a-zA-Z_]\w*\b      # Identifiers
        | \b(?:List|Set|Map|HashMap|TreeMap|LinkedList|ArrayList|Queue|Deque)<[a-zA-Z_,<>]+>\b  # Generic collections
    '''

    # ✅ Remove ignored tokens: return, try, ;
    line = re.sub(r'\b(return|try)\b', '', line)
    line = re.sub(r';', '', line)
    line = re.sub(r'\belse\b(?!\s+if)', '', line)  # Remove "else" but not "else if"

    # ✅ Tokenize the line
    tokens = re.findall(token_pattern, line, re.VERBOSE)

    return len(tokens), tokens


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
    control_regex = re.compile(r'\b(if|}else if|else|for|while|do|switch|case|default)\b')
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
    Calculate the complexity of C# compound conditions in a line of code.
    Formula: Complexity = 1 + (Number of Logical Operators)
    Only applies to if, while, do-while, and switch conditions.
    """

    complexity = 0

    # Ensure the line contains only if, while, do-while, or switch conditions
    valid_condition_pattern = r'\b(if|while|do\s*while|switch)\s*\(.*?\)'
    if not re.search(valid_condition_pattern, line):
        return complexity  # Return 0 if the line is not a valid condition type

    # Pattern to detect logical operators in conditions
    logical_operator_pattern = r'&&|\|\|'

    # Count logical operators in the condition
    logical_operators = len(re.findall(logical_operator_pattern, line))

    # Apply formula: Complexity = 1 + (number of logical operators)
    complexity = 1 + logical_operators

    return complexity


def calculate_try_catch_weight(java_code):
    """
    Calculates the weight of nesting levels specifically for try-catch-finally blocks in Java code.
    - Increments nesting level for try.
    - Assigns weights for catch and finally based on nesting level.
    - Properly handles nested try-catch-finally blocks.
    """
    java_code = remove_comments(java_code)
    lines = java_code.splitlines()

    current_nesting = 0
    nesting_levels = []
    line_weights = {}

    control_regex = re.compile(r'\b(try|catch|finally)\b')
    catch_weights = {1: 1, 2: 2, 3: 3, 4: 4}
    finally_weight = 2

    try_stack = []  # Stack to track try nesting levels

    for line_no, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        if not stripped_line:
            nesting_levels.append((line_no, stripped_line, current_nesting, 0))
            continue

        control_match = control_regex.search(stripped_line)
        if control_match:
            control_type = control_match.group()

            if control_type == 'try':
                # Increase nesting level and push onto stack
                current_nesting += 1
                try_stack.append(current_nesting)

            elif control_type == 'catch':
                # Assign weight based on the last try nesting level
                weight = catch_weights.get(try_stack[-1] if try_stack else 1, 1)
                line_weights[line_no] = weight

            elif control_type == 'finally':
                # Assign a fixed weight for finally
                line_weights[line_no] = finally_weight

        # Append line with its weight
        nesting_levels.append((line_no, stripped_line, current_nesting, line_weights.get(line_no, 0)))

        # Adjust nesting level when closing a try block
        if stripped_line.endswith('}'):
            if try_stack:
                try_stack.pop()
            current_nesting = max(0, current_nesting - 1)

    return nesting_levels, line_weights


def extract_thread_classes(java_files):
    """
    Extracts all class names that extend Thread from a list of Java files.
    """
    thread_classes = set()

    for java_code in java_files.values():  # Process all files
        lines = java_code.splitlines()

        for line in lines:
            class_match = re.search(r'class\s+(\w+)\s+extends\s+Thread', line)
            if class_match:
                thread_classes.add(class_match.group(1))  # Store thread class name

    return thread_classes


def calculate_thread_weight(java_files):
    """
    Calculates thread complexity from multiple Java files.
    """
    complexity = {}
    thread_classes = extract_thread_classes(java_files)  # Step 1: Extract thread classes

    for file_name, java_code in java_files.items():
        lines = java_code.splitlines()

        synchronized_stack = []
        block_start_line = None
        last_thread_creation_line = None
        inside_synchronized_method = False

        for line_no, line in enumerate(lines, start=1):
            score = 0
            recommendations = []
            line = line.strip()

            # Check for thread creation (direct or via subclass)
            if re.search(r'new\s+Thread\b', line) or any(
                    re.search(r'new\s+' + cls + r'\b', line) for cls in thread_classes):
                if last_thread_creation_line is not None and line_no == last_thread_creation_line + 1:
                    score += 2  # Consecutive thread creation
                    recommendations.append({
                        "file": file_name,
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Avoid creating threads consecutively; consider using a thread pool instead."
                    })
                else:
                    score += 2  # Regular thread creation
                last_thread_creation_line = line_no

            # Check for synchronized block
            lock_match = re.search(r'synchronized\s*\((.*?)\)', line)
            if lock_match:
                lock_variable = lock_match.group(1)
                if block_start_line is None:
                    block_start_line = line_no
                if synchronized_stack:
                    score += 4  # Nested synchronization
                    recommendations.append({
                        "file": file_name,
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Avoid nested synchronized blocks for better concurrency."
                    })
                else:
                    if not inside_synchronized_method:
                        score += 3  # Basic synchronization block
                        recommendations.append({
                            "file": file_name,
                            "line_number": line_no,
                            "line_content": line,
                            "recommendation": "Review synchronized block scope for optimal concurrency."
                        })
                if inside_synchronized_method:
                    score += 4  # Extra weight for nested synchronization inside method
                    recommendations.append({
                        "file": file_name,
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Avoid using synchronized blocks inside a synchronized method."
                    })
                synchronized_stack.append(lock_variable)

            # Check for method-level synchronization
            if re.search(r'public\s+synchronized\b', line):
                score += 5  # Adjusted weight (was 5)
                inside_synchronized_method = True
                recommendations.append({
                    "file": file_name,
                    "line_number": line_no,
                    "line_content": line,
                    "recommendation": "Consider using synchronized blocks instead of method-level synchronization."
                })

                # Check for nested synchronized blocks inside synchronized method
                if re.search(r'synchronized\s*\(', line):
                    score += 4
                    recommendations.append({
                        "file": file_name,
                        "line_number": line_no,
                        "line_content": line,
                        "recommendation": "Refactor to avoid nested synchronized blocks within synchronized methods."
                    })

            # Store the score and recommendations for the current line if any
            if score > 0:
                complexity[(file_name, line_no)] = {"score": score, "recommendations": recommendations}

    return complexity

def detect_deadlocks(java_code):
    lines = java_code.splitlines()
    lock_graph = defaultdict(set)  # Graph to detect cycles (Lock Dependency Graph)
    lock_stack = defaultdict(list)  # Track nested locks per method
    method_locks = defaultdict(set)  # Store locks used in each method
    deadlock_warnings = []
    method_name = None

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()

        # Detect method declarations (Simple heuristic)
        method_match = re.match(r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\(', line)
        if method_match:
            method_name = method_match.group(3)

        # Detect synchronized blocks
        sync_match = re.search(r'synchronized\s*\((.*?)\)', line)
        if sync_match and method_name:
            lock_var = sync_match.group(1).strip()

            if lock_stack[method_name]:
                prev_lock = lock_stack[method_name][-1]  # Get the last acquired lock
                lock_graph[prev_lock].add(lock_var)  # Register dependency (prev_lock → new_lock)

            lock_stack[method_name].append(lock_var)  # Push new lock onto the stack
            method_locks[method_name].add(lock_var)

        # Detect end of synchronized block (heuristic: closing brace)
        if line == "}":
            if method_name and lock_stack[method_name]:
                lock_stack[method_name].pop()  # Remove the last acquired lock

    # **Detect cycles using Depth-First Search (DFS)**
    def has_cycle():
        visited = set()
        rec_stack = set()

        def dfs(lock):
            if lock in rec_stack:
                return True  # Cycle detected
            if lock in visited:
                return False

            visited.add(lock)
            rec_stack.add(lock)

            for neighbor in lock_graph[lock]:
                if dfs(neighbor):
                    return True

            rec_stack.remove(lock)
            return False

        return any(dfs(lock) for lock in lock_graph)

    # **Detect cycles using Kahn's Algorithm (Topological Sorting)**
    def detect_deadlock_kahn():
        in_degree = defaultdict(int)
        for key in lock_graph:
            for neighbor in lock_graph[key]:
                in_degree[neighbor] += 1

        queue = deque([node for node in lock_graph if in_degree[node] == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in lock_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited_count != len(lock_graph)  # If not all nodes were visited, cycle exists

    # Check if there's a deadlock
    if has_cycle() or detect_deadlock_kahn():
        deadlock_warnings.append({
            "warning": "🔴 Potential deadlock detected: Circular locking dependency found!"
        })

    return deadlock_warnings

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

            field_declaration_match = re.search(field_declaration_pattern, line)
            if field_declaration_match:
                field_class = field_declaration_match.group(1)
                declared_fields.add(field_class)  # Fixed: Now it works since declared_fields is a set
                class_references[current_class][field_class] = class_references[current_class].get(field_class, 0) + 1

            # Detect static method usage
            static_methods = re.findall(static_usage_pattern, line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class,
                                                                                                        0) + 2

            # Detect static variable usage
            static_variables = re.findall(static_variable_pattern, line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class,
                                                                                                        0) + 1

            constructor_matches = re.findall(constructor_pattern, line)
            for match in constructor_matches:
                param_classes = re.findall(r'([A-Z][\w]*)', match)
                for param_class in param_classes:
                    class_references[current_class][param_class] = class_references[current_class].get(param_class,
                                                                                                       0) + 1

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, line)
            for setter_class in setter_matches:
                class_references[current_class][setter_class] = 1

    return class_references

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
MODEL_PATH = "random_forest_reco_model.pkl"
if os.path.exists(data_file):
    dataset = pd.read_csv(data_file)
else:
    dataset = pd.DataFrame(
        columns=["control_structure_complexity", "nesting_level", "compound_condition_weight", "try_catch_weight",
                 "current_inheritance", "label"])


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
    X = data[["control_structure_complexity", "nesting_level", "compound_condition_weight", "try_catch_weight",
              "current_inheritance"]]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate model performanc
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print("=" * 50)
    print("📊 Model Evaluation Metrics")
    print("=" * 50)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print(f"🔹 Precision: {precision:.2f}")
    print(f"🔹 Recall: {recall:.2f}")
    print(f"🔹 F1 Score: {f1:.2f}")
    print("=" * 50)

    return model

# model = train_model(dataset)

# Use existing model if available
if Path(MODEL_PATH).exists():
    model = joblib.load(MODEL_PATH)
    # print("✅ Pre-trained model loaded from disk.")
elif not dataset.empty:
    model = train_model(dataset)
    joblib.dump(model, MODEL_PATH)
    print("✅ New model trained and saved.")
else:
    model = None
    print("⚠️ Dataset is empty. No model available.")


def update_dataset_and_model(new_data):
    global dataset, model

    new_data = clean_and_convert_dataset(new_data)

    dataset = pd.concat([dataset, new_data], ignore_index=True).drop_duplicates()
    dataset.to_csv(data_file, index=False)
    model = train_model(dataset)


def recommend_action(metrics):
    control_structure_complexity, nesting_level, compound_condition_weight, try_catch_weight, current_inheritance = metrics

    if control_structure_complexity == 1 and nesting_level >= 5:
        if compound_condition_weight > 5:
            return "Refactor: Reduce nested if-statements and simplify complex conditions using helper functions or early returns."
        return "Refactor: Reduce deep if-nesting by restructuring logic or using guard clauses."

    if control_structure_complexity == 2 and nesting_level >= 5:
        if compound_condition_weight > 5:
            return "Refactor: Reduce deeply nested for-loops and simplify conditions using helper functions."
        return "Refactor: Optimize loop nesting by breaking logic into smaller functions or using iterators."

    if control_structure_complexity >= 3 and nesting_level >= 5:
        return "Refactor: Minimize switch-case complexity by using polymorphism or strategy pattern."

    if try_catch_weight > 5:
        return "Refactor: Reduce excessive try-catch nesting by isolating error-prone logic into separate functions."

    if compound_condition_weight > 5:
        return "Refactor: Simplify complex conditions using boolean variables or encapsulating logic in functions."

    if try_catch_weight > 3 and try_catch_weight <= 5:
        return "Refactor: Flatten try-catch blocks by handling specific exceptions separately."

    if current_inheritance == 5:
        return "Refactor: Reduce deep inheritance by favoring composition over inheritance."

    if current_inheritance > 3:
        return "Refactor: Simplify class hierarchy by breaking down large inheritance chains into composition-based structures."

    return "No refactoring needed."


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
        # Calculate complexity for the current lin
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
        'cbo': 0
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

def extract_line_number(entry):
    """ Extracts the numeric line number from a string like 'Line 6: some code' """
    match = re.search(r'Line (\d+):', entry)
    return int(match.group(1)) if match else None


def calculate_code_complexity_multiple_files(file_contents):
    results = {}
    results3 = {}
    new_patterns = []

    # Step 1: Track inheritance across all files
    track_inheritance_depth_across_files(file_contents)
    complexity_results = calculate_thread_weight(file_contents)

    # result1 typically contains line-based data from a previous function
    result1 = calculate_code_complexity_multiple(file_contents)
    mpc_results = calculate_mpc_for_java_code(file_contents)

    # Iterate through each file content
    for filename, content in file_contents.items():
        # add_java_code(filename, content)
        class_name = filename.split('.')[0]

        # Split content into lines
        lines = content.splitlines()
        complexity_data = []
        complexity_data2 = []

        # Run CBO Analysis using CBOMetrics
        cbo_analyzer = CBOMetrics(content)
        cbo_report = cbo_analyzer.get_cbo_report()

        # ----------------------------------------------------
        # Define these dictionaries to avoid "unresolved reference"
        # ----------------------------------------------------
        cbo_constructor_lines = {
            extract_line_number(info): info
            for info in cbo_report["Constructor Injections"]
            if extract_line_number(info)
        }
        cbo_setter_lines = {
            extract_line_number(info): info
            for info in cbo_report["Setter Injections"]
            if extract_line_number(info)
        }
        cbo_instantiation_lines = {
            extract_line_number(info): info
            for info in cbo_report["Direct Object Instantiations"]
            if extract_line_number(info)
        }
        cbo_dependency_assignment_lines = {
            extract_line_number(info): info
            for info in cbo_report["Dependency Assignment Weights"]
            if extract_line_number(info)
        }
        cbo_static_method_lines = {
            extract_line_number(info): info
            for info in cbo_report["Static Method Calls"]
            if extract_line_number(info)
        }
        cbo_static_variable_lines = {
            extract_line_number(info): info
            for info in cbo_report["Static Variable Usages"]
            if extract_line_number(info)
        }

        # (If you have a model output, use it for code recommendations)
        model_based_recommendations = get_code_recommendations(content, model_output)

        results3[filename] = {
            "prediction": model_based_recommendations.get("prediction", "Unknown"),
            "recommendations": model_based_recommendations.get("recommendations", [])
        }

        # Extract line-based CBO data (from result1) & MPC data
        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Additional code analysis
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        nesting_levels = calculate_nesting_level(content)
        nesting_level_dict = {line[0]: line[2] for line in nesting_levels}
        try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)
        if filename in {key[0] for key in complexity_results.keys()}:
            thread_weights = {key[1]: value for key, value in complexity_results.items() if key[0] == filename}
        else:
            thread_weights = {}
            print(f"Warning: No thread weight data found for {filename}!")


        # Calculate file-level CBO & MPC if needed
        cbo_value = calculate_cbo(class_references)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare for AI recommendations
        line_complexities = []
        method_inheritance = {}

        # Control structure complexities (line_weights) for each line
        line_weights, total_control_complexity = calculate_control_structure_complexity(lines)

        # Initialize total WCC for the file
        total_wcc = 0

        for line_number, line in enumerate(lines, start=1):
            # 1. Calculate size (token count)
            size, tokens = calculate_size(line)

            # 2. Skip lines that we do not measure
            # if any(keyword in line for keyword in ["class"]):
            #     continue
            if size == 0:
                # If size=0, record minimal data and skip
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
                })
                complexity_data.append([
                    line_number,
                    line.strip(),
                    size,
                    ', '.join(tokens),
                    0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue

            # 3. Determine the line-based CBO weight
            cbo_weight = 0
            if line_number in cbo_instantiation_lines:
                cbo_weight += 3
            if line_number in cbo_dependency_assignment_lines:
                cbo_weight += 1
            if line_number in cbo_static_method_lines:
                cbo_weight += 3
            if line_number in cbo_static_variable_lines:
                cbo_weight += 3

            # (Alternatively, you may want to incorporate data from cbo_line_data)
            cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
            total_cbo_weight = sum(cbo_weights.values())

            # 4. MPC weight (if used)
            mpc_weight = mpc_line_data.get(line_number, 0)

            # 5. Additional metrics
            nesting_level = nesting_level_dict.get(line_number, 0)
            try_catch_weight = try_catch_weight_dict.get(line_number, 0)
            thread_weight_info = thread_weights.get(line_number, {"score": 0, "reasons": []})
            current_inheritance = calculate_inheritance_level2(class_name)
            method_inheritance[class_name] = current_inheritance
            compound_condition_weight = calculate_compound_condition_weight(line)
            control_structure_complexity = line_weights.get(line_number, {"weight": 0})["weight"]

            # 6. Identify if there's "loose coupling" => L_i = 1
            #    Example: if cbo_weight == 1, we consider it "loose coupling"
            if cbo_weight == 1:
                loose_coupling_offset = 1
            else:
                loose_coupling_offset = 0

            # 7. Compute line-by-line WCC using Option A:
            #
            #    WCC_i = S_i * (
            #        CS_i + N_i + I_i + CC_i + TC_i + TH_i + (CBO_{h,i} - L_i)
            #    )
            #
            #    If you want to ensure no negative coupling, clamp at zero:
            adjusted_cbo = max(0, cbo_weight - loose_coupling_offset)

            line_wcc = size * (
                    control_structure_complexity
                    + nesting_level
                    + current_inheritance
                    + compound_condition_weight
                    + try_catch_weight
                    + thread_weight_info['score']
                    + adjusted_cbo
            )

            # 8. Accumulate WCC into total
            total_wcc += line_wcc

            # 9. Store line-level data
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
                thread_weight_info['score'],
                cbo_weight,
                line_wcc  # store the computed WCC for this line
            ])

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
                'thread_weight': thread_weight_info['score'],
                'cbo_weights': cbo_weight,
                'line_wcc': line_wcc
            })

            # (Optional) If you have an internal dataset to track patterns:
            # Check if pattern is new, etc.
            metrics = [
                control_structure_complexity,
                nesting_level,
                compound_condition_weight,
                try_catch_weight,
                current_inheritance
            ]
            recommendation = recommend_action(metrics)
            if not (
                    (dataset["control_structure_complexity"] == control_structure_complexity)
                    & (dataset["nesting_level"] == nesting_level)
                    & (dataset["compound_condition_weight"] == compound_condition_weight)
                    & (dataset["try_catch_weight"] == try_catch_weight)
                    & (dataset["current_inheritance"] == current_inheritance)
            ).any():
                new_patterns.append({
                    "control_structure_complexity": control_structure_complexity,
                    "nesting_level": nesting_level,
                    "compound_condition_weight": compound_condition_weight,
                    "try_catch_weight": int(try_catch_weight),
                    "current_inheritance": int(current_inheritance),
                    "label": recommendation.strip('"')
                })

            # Collect some extra data if needed for thread recommendations
            complexity_data2.append({
                "line_number": line_number,
                "line_content": line.strip(),
                "thread_weight": thread_weight_info.get("score", 0),
                "recommendations": thread_weight_info.get("recommendations", [])
            })

        # Calculate complexities at the method level
        method_complexities = calculate_code_complexity_by_method(
            line_complexities
        )

        # If new patterns appear, update dataset & model
        # if new_patterns:
        #     new_data = pd.DataFrame(new_patterns)
        #     update_dataset_and_model(new_data)

        # Get AI recommendations for each line
        recommendations = ai_recommend_refactoring(line_complexities)

        # Filter out "No action needed"
        filtered_recommendations = [
            rec for rec in recommendations if rec['recommendation'] != "No action needed"
        ]
        # Merge in thread-based or other extra recommendations
        for item3 in complexity_data2:
            if isinstance(item3, dict) and "recommendations" in item3:
                filtered_recommendations.extend(item3["recommendations"])

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
            }
            bar_chart_path = plot_complexity_bar_chart(method_name, relevant_factors, filename)
            bar_chart_paths[method_name] = bar_chart_path

        # Assemble results for this file
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
method_signature_regex = (
    r'\b(?:public|private|protected)?\s*'  # Optional access modifier
    r'(?:static\s+)?'  # Optional 'static'
    r'(?:synchronized\s+)?'  # Optional 'synchronized'
    r'(?:[\w<>\[\]]+\s+)+'  # Return type (ensures this is not a constructor)
    r'(?!set)(\w+)\s*\([^)]*\)\s*'  # Method name (that does not start with "set") and parameter list
    r'(?:throws\s+[\w<>\[\]]+(?:\s*,\s*[\w<>\[\]]+)*)?\s*'  # Optional 'throws' clause
    r'\{'  # Opening brace
)
# Exclude common control keywords.
excluded_keywords = {"if", "for", "while", "switch", "catch", "Thread", "run"}


def calculate_code_complexity_by_method(line_complexities):
    """
    Group precomputed per-line complexity data (line_complexities) into methods and
    aggregate metrics for each method using Option A.

    Each entry in line_complexities is a dictionary containing keys:
      - 'line_number'
      - 'line_content'
      - 'size'
      - 'control_structure_complexity'
      - 'nesting_level'
      - 'inheritance_level'
      - 'compound_condition_weight'
      - 'try_catch_weight'
      - 'thread_weight'
      - 'cbo_weights'
      - 'line_wcc'   (precomputed per-line, Option A)

    :param line_complexities: List of per-line metric dictionaries.
    :return: Dictionary mapping method names to aggregated complexity metrics.
    """
    methods = {}
    method_name = None
    method_lines = []
    brace_counter = 0

    for entry in line_complexities:
        ln = entry.get('line_number')
        line_content = entry.get('line_content', '')

        # Look for a method signature in the current line.
        match = re.search(method_signature_regex, line_content)
        if match and brace_counter == 0:
            candidate_method_name = match.group(1)
            if candidate_method_name in excluded_keywords:
                continue  # Skip control structure lines.
            # Finalize the previous method if any.
            if method_name and method_lines:
                methods[method_name] = aggregate_method_metrics(method_lines)
            # Start a new method.
            method_name = candidate_method_name
            method_lines = [entry]
            brace_counter = line_content.count("{") - line_content.count("}")
        elif method_name:
            # Add current line to the method.
            method_lines.append(entry)
            brace_counter += line_content.count("{") - line_content.count("}")
            # If the braces balance, the method is complete.
            if brace_counter == 0:
                methods[method_name] = aggregate_method_metrics(method_lines)
                method_name = None
                method_lines = []
    # Finalize any method still in progress.
    if method_name and method_lines:
        methods[method_name] = aggregate_method_metrics(method_lines)
    return methods


def aggregate_method_metrics(method_lines):
    """
    Aggregate the per-line metrics of a method and re-calculate the overall method complexity
    using Option A.

    For Option A on a per-line basis:
      line_wcc_i = S_i * (CS_i + N_i + I_i + CC_i + TC_i + TH_i + (CBO_i - L_i))
    where L_i is the loose coupling offset (already applied when computing each line's cbo_weights).

    We calculate:
      - total_size = sum(S_i)
      - weighted_sum = sum( S_i * (CS_i + N_i + I_i + CC_i + TC_i + TH_i + cbo_weights) )
      - weighted_average = weighted_sum / total_size
      - total_complexity_option_a = total_size * weighted_average

    Additionally, we also sum the precomputed per-line 'line_wcc' values for reference.

    :param method_lines: List of per-line metric dictionaries for the method.
    :return: Dictionary with aggregated metrics including total complexity computed via Option A.
    """
    total_size = sum(entry.get("size", 0) for entry in method_lines)

    # Sum over each line: size * (CS + N + I + CC + TC + TH + cbo_weights)
    weighted_sum = sum(
        entry.get("size", 0) * (
                entry.get("control_structure_complexity", 0) +
                entry.get("nesting_level", 0) +
                entry.get("inheritance_level", 0) +
                entry.get("compound_condition_weight", 0) +
                entry.get("try_catch_weight", 0) +
                entry.get("thread_weight", 0) +
                entry.get("cbo_weights", 0)
        )
        for entry in method_lines
    )
    if total_size > 0:
        weighted_average = weighted_sum / total_size
    else:
        weighted_average = 0

    # Total complexity calculated using Option A on aggregated data.
    total_complexity_option_a = total_size * weighted_average

    # Also sum the per-line precomputed complexity values.
    total_line_wcc = sum(entry.get("line_wcc", 0) for entry in method_lines)

    aggregated = {
        "size": total_size,
        "control_structure_complexity": sum(entry.get("control_structure_complexity", 0) for entry in method_lines),
        "nesting_level": sum(entry.get("nesting_level", 0) for entry in method_lines),
        "inheritance_level": sum(entry.get("inheritance_level", 0) for entry in method_lines),
        "compound_condition_weight": sum(entry.get("compound_condition_weight", 0) for entry in method_lines),
        "try_catch_weight": sum(entry.get("try_catch_weight", 0) for entry in method_lines),
        "thread_weight": sum(entry.get("thread_weight", 0) for entry in method_lines),
        "cbo_weights": sum(entry.get("cbo_weights", 0) for entry in method_lines),
        "total_complexity": total_line_wcc,
        # "total_line_wcc": total_line_wcc
    }
    return aggregated


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