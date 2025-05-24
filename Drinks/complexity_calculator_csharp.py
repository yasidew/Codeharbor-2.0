import json
import logging
import re
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pygments.lexers import CSharpLexer
from pygments.token import Token
from pygments import lex
from xgboost import XGBClassifier

from Drinks.complexity_calculator import model_output
from Drinks.process_java_file_cbo import process_csharp_files, process_csharp_files1, calculate_cbo_csharp1


def calculate_cbo_csharp(lines):
    """
    Calculates Coupling Between Objects (CBO) in C# code using Pygments tokenization.

    Parameters:
        - lines (list of str): List of lines in C# code.

    Returns:
        - dict: A structured report with CBO details and assigned complexity weights.
    """

    dependencies = set()
    constructor_injections = {}
    setter_injections = {}
    direct_instantiations = []
    static_method_calls = []
    static_variable_usages = []
    injection_initiations = []  # Track assignment inside constructor

    primitive_types = {"byte", "short", "int", "long", "float", "double", "bool", "char"}
    built_in_classes = {'Console', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
                        'Math', 'Object', 'Thread', 'Runtime', 'Optional', 'Task', 'Action', 'Func', 'Exception',
                        'SystemException', 'InvalidOperationException', 'ArgumentException',
                        'NullReferenceException', 'IndexOutOfRangeException', 'DivideByZeroException',
                        'FormatException', 'OverflowException', 'StackOverflowException', 'OutOfMemoryException',
                        'IOException', 'FileNotFoundException', 'UnauthorizedAccessException', 'TimeoutException'}

    direct_instantiation_pattern = re.compile(r"\bnew\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    static_variable_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
    static_method_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\(")

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        tokens = list(lex(stripped_line, CSharpLexer()))

        if not stripped_line:
            continue

        # ✅ Constructor Injection (Weight = 1)
        if any(tok[0] == Token.Keyword and tok[1] == "public" for tok in
               tokens) and "(" in stripped_line and ")" in stripped_line:
            class_name = stripped_line.split("(")[0].split()[-1]
            if class_name not in primitive_types and class_name not in built_in_classes:
                dependencies.add(class_name)
                constructor_injections[class_name] = {
                    "line": line_number,
                    "weight": 1
                }

        # ✅ Detect Injection Initiation inside constructor (e.g., `_paymentProcessor = paymentProcessor;`)
        elif "=" in stripped_line and any(tok[0] == Token.Name for tok in tokens):
            parts = stripped_line.split("=")
            left_side = parts[0].strip()
            right_side = parts[1].strip()
            if left_side.startswith("_") and right_side:
                injection_initiations.append({
                    "line": line_number,
                    "assignment": stripped_line,
                    "weight": 1
                })

        # ✅ Setter Injection (Weight = 1)
        elif stripped_line.startswith("public void set"):
            parts = stripped_line.split(" ")
            if len(parts) > 2:
                param_type = parts[-1].replace("(", "").replace(")", "")
                if param_type not in primitive_types and param_type not in built_in_classes:
                    dependencies.add(param_type)
                    setter_injections[param_type] = {
                        "line": line_number,
                        "weight": 1
                    }

        match = direct_instantiation_pattern.search(stripped_line)
        if match:
            class_name = match.group(1)
            if class_name not in primitive_types and class_name not in built_in_classes:
                dependencies.add(class_name)
                direct_instantiations.append({
                    "line": line_number,
                    "instantiation": f"new {class_name}()",
                    "weight": 3
                })

        match = static_method_pattern.search(stripped_line)

        if match:

            class_name = match.group(1)

            if class_name not in primitive_types and class_name not in built_in_classes:
                dependencies.add(class_name)

                static_method_calls.append({

                    "line": line_number,

                    "method_call": stripped_line,

                    "weight": 1

                })

        # ✅ Static Variable Usages (Weight = 1)
        match = static_variable_pattern.search(stripped_line)
        if match:
            class_name = match.group(1)
            if class_name not in primitive_types and class_name not in built_in_classes:
                dependencies.add(class_name)
                static_variable_usages.append({
                    "line": line_number,
                    "variable_usage": stripped_line,
                    "weight": 1
                })

    cbo_value = len(dependencies)

    return {
        "CBO Score": cbo_value,
        "Constructor Injections": constructor_injections,
        "Setter Injections": setter_injections,
        "Direct Object Instantiations": direct_instantiations,
        "Static Method Calls": static_method_calls,
        "Static Variable Usages": static_variable_usages,
        "Injection Initiations": injection_initiations
    }


ccccc = """

public class Order
{
    private readonly IPaymentProcessor _paymentProcessor;
    private readonly Cat _cat;
    public int OrderId { get; private set; }
    public string CustomerName { get; private set; }
    public double OrderAmount { get; private set; }
    public bool IsPaid { get; private set; }
    public string OrderStatus { get; private set; }

    // Constructor Injection for Dependency
    public Order(int orderId, string customerName, double amount, IPaymentProcessor paymentProcessor, Cat cat){
        OrderId = orderId;
        CustomerName = customerName;
        OrderAmount = amount;
        _paymentProcessor = paymentProcessor;
        IsPaid = false;
        OrderStatus = "Pending";
        _cat = cat;
    }

    public setOrder(Cat cat){
        _cat = cat;
    }

    // Method to Process Payment
    public void ProcessOrder(){
        Console.WriteLine($"Processing order {CustomerName}...");
        Cat cat = new Cat()

        // Compound condition - Ensures amount is valid and payment processor is set
        if (OrderAmount > 0 && _paymentProcessor != null){
            IsPaid = _paymentProcessor.ProcessPayment(OrderAmount);
            OrderStatus = IsPaid ? "Paid" : "Payment Failed";
        }
        else{
            OrderStatus = "Invalid Order";
        }

        Console.WriteLine($"Order Status: {OrderStatus}");
    }

    // Method to Apply Discount
    public void ApplyDiscount(){
        // Conditional Statement with Compound Condition
        if (OrderAmount >= 100 && OrderAmount < 200){
            OrderAmount *= 0.90; // 10% Discount
            Console.WriteLine("Applied 10% discount.");
        }
        else if (OrderAmount >= 200){
            OrderAmount *= 0.80; // 20% Discount
            Console.WriteLine("Applied 20% discount.");
        }
        else{
            Console.WriteLine("No discount applied.");
        }
    }

    // Method to Change Order Status
    public void UpdateOrderStatus(string newStatus){
        // Nested Conditional Statement
        switch (newStatus.ToLower()){
            case "shipped":
                if (IsPaid){
                    OrderStatus = "Shipped";
                    Console.WriteLine("Order has been shipped.");
                } else{
                    Console.WriteLine("Order cannot be shipped. Payment pending.");
                }
                break;

            case "delivered":
                if (OrderStatus == "Shipped"){
                    OrderStatus = "Delivered";
                    Console.WriteLine("Order has been delivered.");
                }else {
                    Console.WriteLine("Order cannot be delivered before shipping.");
                }
                break;

            default:
                Console.WriteLine("Invalid status update.");
                break;
        }
    }
}
"""


def extract_cbo_features_csharp(csharp_code):
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

    # ✅ Exclude built-in classes like Console, Math, and System
    excluded_classes = {"Console", "Math", "System", "Thread", "Task", "Runtime"}

    # ✅ Regex Patterns for Feature Extraction
    instantiation_pattern = r'new\s+([A-Z][\w]*)\s*\('  # Detects `new ClassName()`
    static_method_pattern = r'\b([A-Z][\w]*)\s*\.\s*([A-Z]\w*)\s*\('  # Detects `ClassName.Method()`
    static_variable_pattern = r'([A-Z][\w]*)\.\w+\s*='  # Detects `ClassName.Variable = Value`
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'  # Constructor Injection
    setter_pattern = r'public\s+void\s+set[A-Z]\w*\s*\(([^)]*)\)'  # Setter Injection
    class_pattern = r'class\s+([A-Z][\w]*)'  # Class Declaration
    interface_pattern = r'class\s+\w+\s*:\s*([\w\s,]+)'

    # ✅ Extract Class Dependencies
    class_dependencies = re.findall(class_pattern, csharp_code)
    for cls in class_dependencies:
        cbo_features["class_dependencies"].add(cls)

    # ✅ Extract Constructor Injections
    constructors = re.findall(constructor_pattern, csharp_code)
    for params in constructors:
        param_classes = re.findall(r'([A-Z][\w]*)\s+\w+', params)
        for param_class in param_classes:
            cbo_features["class_dependencies"].add(param_class)
            cbo_features["constructor_injections"] += 1  # Each param counts separately

    # ✅ Extract Setter Injections
    setters = re.findall(setter_pattern, csharp_code)
    for params in setters:
        param_classes = re.findall(r'([A-Z][\w]*)\s+\w+', params)
        for param_class in param_classes:
            cbo_features["class_dependencies"].add(param_class)
            cbo_features["setter_injections"] += 1

    # ✅ Extract Direct Object Instantiations
    instantiations = re.findall(instantiation_pattern, csharp_code)
    for instantiated_class in instantiations:
        cbo_features["class_dependencies"].add(instantiated_class)
        cbo_features["direct_instantiations"] += 1

    # ✅ Extract Static Method Calls (excluding built-in classes)
    static_methods = re.findall(static_method_pattern, csharp_code)
    for static_class, method_name in static_methods:
        if static_class not in excluded_classes:
            cbo_features["class_dependencies"].add(static_class)
            cbo_features["static_method_calls"] += 1

    # ✅ Extract Static Variable Usage (excluding built-in classes)
    static_variables = re.findall(static_variable_pattern, csharp_code)
    for static_class in static_variables:
        if static_class not in excluded_classes:
            cbo_features["class_dependencies"].add(static_class)
            cbo_features["static_variable_usage"] += 1

    # ✅ Extract Interface Implementations
    interfaces = re.findall(interface_pattern, csharp_code)
    for interface_list in interfaces:
        implemented_interfaces = interface_list.split(',')
        cbo_features["interface_implementations"] += len(implemented_interfaces)

    cbo_features["class_dependencies"] = len(cbo_features["class_dependencies"])

    return cbo_features


class CBOMetricsCSharp:
    def __init__(self, csharp_code):
        """
        Parses C# code and extracts:
        - Constructor & Setter Method Injection
        - Direct Object Instantiation (new Keyword)
        - Assigns weights to dependencies
        - Calculates CBO value
        """
        self.code = csharp_code
        self.dependencies = set()
        self.constructor_injections = {}
        self.setter_injections = {}
        self.direct_instantiations = []

    def extract_dependencies(self):
        """
        Uses Pygments to tokenize C# code and identify dependencies.
        """

        tokens = list(lex(self.code, CSharpLexer()))
        prev_token = None
        class_name = None

        for i, (tok_type, tok_value) in enumerate(tokens):
            # **Detect Constructor Injections (Constructor with parameters)**
            if tok_type in Token.Name and i + 2 < len(tokens):
                next_tok_type, next_tok_value = tokens[i + 1]

                # Class name followed by '(' is likely a constructor
                if next_tok_value == "(":
                    constructor_line = i
                    param_types = []
                    j = i + 2
                    while j < len(tokens) and tokens[j][1] != ")":
                        if tokens[j][0] in Token.Name and tokens[j - 1][1] != ",":
                            param_types.append(tokens[j][1])  # Extract **only** parameter types
                        j += 1
                    if param_types:
                        self.constructor_injections[tok_value] = {
                            "parameters": param_types,
                            "constructor_line": constructor_line,
                            "weight": 1,
                        }
                        self.dependencies.update(param_types)

            # **Detect Setter Injections (public void Method(Type obj))**
            if prev_token and prev_token[1] == "public" and tok_type in Token.Keyword and tok_value == "void":
                method_name = tokens[i + 1][1] if i + 1 < len(tokens) else None
                if method_name and i + 3 < len(tokens) and tokens[i + 2][1] == "(":
                    param_type = tokens[i + 3][1]  # Extract **only** parameter type
                    if param_type:
                        self.setter_injections[method_name] = {
                            "param_type": param_type,
                            "setter_line": i,
                            "weight": 1,
                        }
                        self.dependencies.add(param_type)

            # **Detect Direct Instantiations (`new ClassName()`)**
            if prev_token and prev_token[1] == "new" and tok_type in Token.Name:
                instantiation_line = i
                class_name = tok_value
                self.direct_instantiations.append({
                    "line": instantiation_line,
                    "instantiation": f"new {class_name}();",
                    "weight": 3,
                })
                self.dependencies.add(class_name)

            prev_token = (tok_type, tok_value)

    def calculate_cbo(self):
        """
        Calculates the Coupling Between Objects (CBO) value based on dependencies.
        """
        cbo_value = len(self.dependencies)  # Count unique dependencies
        return cbo_value, self.constructor_injections, self.setter_injections, self.direct_instantiations

    def get_cbo_report(self):
        """
        Extracts dependencies, calculates CBO, and returns a structured report.
        """
        self.extract_dependencies()
        cbo_value, constructor_injections, setter_injections, direct_instantiations = self.calculate_cbo()

        report = {
            "CBO Score": cbo_value,
            "Constructor Injections": [
                f"Constructor: {class_name} (Params: {info['parameters']}) (Weight: {info['weight']})"
                for class_name, info in constructor_injections.items()
            ],
            "Setter Injections": [
                f"Setter: {method_name} (Param: {info['param_type']}) (Weight: {info['weight']})"
                for method_name, info in setter_injections.items()
            ],
            "Direct Object Instantiations": [
                f"Line {info['line']}: {info['instantiation']} (Weight: {info['weight']})"
                for info in direct_instantiations
            ],
        }
        return report


# Load dataset
df = pd.read_csv("media/Updated_Dataset_with_CBO_Labeling.csv")
model_filename = "xgboost_c#_cbo_model.pkl"

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    # print(f"✅ Loaded existing model c# cbo from {model_filename}")
else:
    # Drop 'file_name' column as it's not a feature
    df.drop(columns=["file_name"], inplace=True)

    # Define features (X) and labels (y)
    X = df.drop(columns=["cbo_label"])
    y = df["cbo_label"]

    # Split data into training and testing sets
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

    # Save trained model for use in C#
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    # Load the trained model
    # rf_model = joblib.load("xgboost_c#_cbo_model.pkl")

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


def calculate_size(line):
    global class_declaration_ended

    line = remove_comments(line)

    # ✅ Skip using directives (equivalent to Java's imports)
    if re.match(r'^\s*using\s+[a-zA-Z0-9_.\*]+;', line):
        return 0, []  # Ignore using statements

    if re.match(r'^\s*namespace\s+[a-zA-Z0-9_.]+', line):
        return 0, []  # Ignore namespace declarations

    # ✅ Check for class/struct declaration and skip until it's completed
    if re.search(r'\b(class|struct)\s+\w+', line):
        class_declaration_ended = True
        return 0, []  # Ignore class/struct declaration

    # ✅ Remove attributes like [Obsolete]
    line = re.sub(r'\[\w+\]', '', line)

    # ✅ Remove access modifiers and keywords
    line = re.sub(
        r'\b(public|private|protected|internal|abstract|static|sealed|readonly|virtual|override|unsafe|async|extern|else)\b',
        '', line)

    # ✅ Handle method and constructor signatures (ignoring parameters)
    method_signature_regex = (
        r'\b(?:public|private|protected|internal)?\s*'
        r'(?:static\s+)?'
        r'(?:async\s+)?'
        r'([\w<>\[\]]+)\s+'  # Return type
        r'(?!set)(\w+)\s*\([^)]*\)\s*'  # Method name
        r'\{'
    )
    line = re.sub(method_signature_regex, r'\1 \2()', line)
    line = re.sub(r'\[\w+\]', '', line)

    constructor_regex = (
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{'
    )
    line = re.sub(constructor_regex, r'\1()', line)

    # ✅ Handle system calls (e.g., Console.WriteLine(), Debug.Log())
    system_call_pattern = r'\b(Console|Debug)\s*\.\s*([a-zA-Z_]\w*)\s*\('
    line = re.sub(system_call_pattern, r'\1 , . , \2 (', line)

    # ✅ Handle user-defined method calls
    user_defined_call_pattern = r'\b([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\(.*?\)'
    line = re.sub(user_defined_call_pattern, r'\1 , . , \2()', line)

    # ✅ Handle logger calls (e.g., Logger.Info("Message"))
    logger_call_pattern = r'\b(logger|Logger)\s*\.\s*([a-zA-Z_]\w*)\s*\("(.*?)"\)'
    line = re.sub(logger_call_pattern, r'\1 , . , \2 "\3"', line)

    # ✅ Handle Generics (List<int>, Dictionary<string, object>)
    generic_type_pattern = r'\b(?:List|Queue|Stack|Dictionary|HashSet|SortedSet|LinkedList)\s*<\s*[\w\s,<>]+\s*>'
    generics = re.findall(generic_type_pattern, line)

    for generic in generics:
        line = line.replace(generic, generic.replace(" ", ""))  # Remove spaces in generics

    # ✅ Token Patterns (Updated for WCC Compliance)
    token_pattern = r'''
        "[^"]*"                 # Strings inside double quotes
        | '[^']*'               # Strings inside single quotes
        | \+\+|--               # Pre and post increment/decrement as one token
        | \belse\s+if\b         # "else if" as one token
        | \bif\b                # "if" should be counted
        | \b(?:for|while|switch|case|default|catch)\b  # Control structures
        | \b(?:int|float|double|char|bool|long|short|byte|string|void)\b  # Data types
        | &&|\|\||===|==|!==|>=|<=|!=   # Logical and comparison operators as one token
        | [-+*/%=&|<>!~^]         # Operators
        | -?\d+                 # Numbers
        | \.                    # Dot operator
        | \b[a-zA-Z_]\w*\(\)    # Method calls
        | \b[a-zA-Z_]\w*\b      # Identifiers
        | \b(?:List|Queue|Stack|Dictionary|HashSet|SortedSet|LinkedList)<[a-zA-Z_,<>]+>\b  # Generic collections
    '''

    # ✅ Remove ignored tokens: return, try, ;
    line = re.sub(r'\b(return|try)\b', '', line)
    line = re.sub(r';', '', line)
    line = re.sub(r'\belse\b(?!\s+if)', '', line)  # Remove "else" but not "else if"

    # ✅ Tokenize the line
    tokens = re.findall(token_pattern, line, re.VERBOSE)

    return len(tokens), tokens


def get_code_lines(csharp_code):
    """ Convert C# code into a list of lines for reference. """
    return {i + 1: line.strip() for i, line in enumerate(csharp_code.split("\n"))}


def calculate_control_structure_complexity(lines):
    """
    Calculates control structure complexity for a given C# code.

    Parameters:
        - lines (list): List of C# code lines.

    Returns:
        - dict: A dictionary with line numbers as keys and assigned complexity weights as values.
        - int: Total complexity of the code.
    """
    total_weight = 0
    line_weights = {}

    # Stack to keep track of switch-case blocks
    switch_stack = []

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

        # ✅ Loops (for, while, do-while, foreach) → Weight = 2
        elif re.match(r"^\s*(for|while|do|foreach)\b", stripped_line):
            weight = 2

        # ✅ Switch-Case Complexity → Weight = Number of cases + 1 (if default is present)
        elif re.match(r"^\s*switch\b", stripped_line):
            case_count = 0
            default_found = False
            switch_stack.append(line_number)  # Track the switch block

            # Scan forward to count cases & check for a default statement
            for subsequent_line in lines[line_number:]:
                subsequent_line = subsequent_line.strip()
                if re.match(r"^\s*case\b", subsequent_line):
                    case_count += 1
                if re.match(r"^\s*default\b", subsequent_line):
                    default_found = True
                if subsequent_line == "}":  # End of switch block
                    break

            # Calculate switch weight
            weight = case_count + (1 if default_found else 0)

            # ✅ Ensure the minimum weight for a switch with exactly 2 cases and 1 default is 3
            if case_count == 2 and default_found:
                weight = 3

        # ✅ Handle nested switch cases correctly
        elif re.match(r"^\s*case\b", stripped_line) or re.match(r"^\s*default\b", stripped_line):
            weight = 0  # Cases/defaults themselves do not add weight

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
    catch_weights = {1: 1, 2: 2, 3: 3, 4: 4}
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
                weight = catch_weights.get(current_nesting, 1)
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


def calculate_thread_weight(csharp_code):
    complexity = {}

    lines = csharp_code.splitlines()
    lock_stack = []
    block_start_line = None
    last_thread_creation_line = None

    for line_no, line in enumerate(lines, start=1):
        score = 0
        recommendations = []
        stripped_line = line.strip()

        # Check for thread creation (e.g., new Thread(...))
        if re.search(r'new\s+Thread\b', stripped_line):
            if last_thread_creation_line is not None and line_no == last_thread_creation_line + 1:
                score += 2  # Consecutive thread creation
                recommendations.append({
                    "line_number": line_no,
                    "line_content": stripped_line,
                    "recommendation": "Avoid creating threads consecutively; consider using a thread pool instead."
                })
            else:
                score += 2
            last_thread_creation_line = line_no

        # Check for lock block (C# uses the "lock" keyword)
        lock_match = re.search(r'lock\s*\((.*?)\)', stripped_line)
        if lock_match:
            lock_variable = lock_match.group(1)  # Extract the lock variable
            if block_start_line is None:
                block_start_line = line_no  # Start of lock block
            if lock_stack:
                # Nested lock block: assign higher weight for nested locking
                score += 4
                recommendations.append({
                    "line_number": line_no,
                    "line_content": stripped_line,
                    "recommendation": "Avoid nested lock blocks for better concurrency."
                })
            else:
                score += 3  # Basic lock block weight
                recommendations.append({
                    "line_number": line_no,
                    "line_content": stripped_line,
                    "recommendation": "Review lock block scope for optimal concurrency."
                })
            lock_stack.append(lock_variable)

        # Check for end of block (assuming a closing brace "}" ends the lock block)
        if stripped_line == "}":
            if lock_stack:
                start_lock = lock_stack.pop()
                # If the lock block spans many lines, add extra weight.
                if block_start_line and (line_no - block_start_line) > 5:
                    score += 5
                    recommendations.append({
                        "line_number": line_no,
                        "line_content": stripped_line,
                        "recommendation": "Refactor to reduce the scope of lock blocks for better concurrency."
                    })
                if not lock_stack:  # Reset block start when no nested blocks remain
                    block_start_line = None

        # Check for method-level synchronization attribute in C#
        # For example: [MethodImpl(MethodImplOptions.Synchronized)]
        if re.search(r'\[MethodImpl\s*\(\s*MethodImplOptions\.Synchronized\s*\)\]', stripped_line):
            score += 5
            recommendations.append({
                "line_number": line_no,
                "line_content": stripped_line,
                "recommendation": "Avoid method-level synchronization; prefer fine-grained locking."
            })

        if score > 0:
            complexity[line_no] = {"score": score, "recommendations": recommendations}

    return complexity


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
                    class_references[current_class][param_class] = class_references[current_class].get(param_class,
                                                                                                       0) + 1

            # Detect setter injection
            setter_matches = re.findall(setter_injection_pattern, stripped_line)
            for setter_class in setter_matches:
                class_references[current_class][setter_class] = 1

            # Detect static method usage
            static_methods = re.findall(static_usage_pattern, stripped_line)
            for static_class in static_methods:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class,
                                                                                                        0) + 2

            # Detect static variable usage
            static_variables = re.findall(static_variable_pattern, stripped_line)
            for static_class in static_variables:
                if static_class not in excluded_classes:
                    class_references[current_class][static_class] = class_references[current_class].get(static_class,
                                                                                                        0) + 1

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


class CBOMetricsCSharp:
    def __init__(self, csharp_code):
        """
        Parses C# code and extracts:
        - Constructor & Setter Method Injection (Weight = 1)
        - Direct Object Instantiation (Weight = 3) [user-defined classes only]
        - Static Method Calls (Weight = 3)
        - Static Variable Usage (Weight = 1)
        """
        self.csharp_code = csharp_code
        self.dependencies = set()
        self.constructor_injections = []
        self.setter_injections = []
        self.direct_instantiations = []
        self.static_method_calls = []
        self.static_variable_usages = []
        self.assignment_weights = []

    def extract_class_references_with_lines(self):
        """
        Extracts class references from C# code line by line for detailed CBO calculations.
        """
        instantiation_pattern = r'new\s+([A-Z][\w]*)\s*(?:\(|\{)'
        field_declaration_pattern = r'([A-Z][\w]*)\s+\w+\s*;'
        static_usage_pattern = r'([A-Z][\w]*)\.\w+\s*\('
        static_variable_pattern = r'([A-Z][\w]*)\.\w+'
        constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)\s*\{'
        setter_injection_pattern = r'public\s+void\s+set[A-Z]\w*\s*\(\s*([A-Z][\w]*)\s+\w+\s*\)'

        excluded_classes = [
            'Console', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
            'Math', 'Object', 'Thread', 'Runtime', 'Optional', 'Task', 'Action', 'Func', 'Exception', 'SystemException',
            'InvalidOperationException', 'ArgumentException',
            'NullReferenceException', 'IndexOutOfRangeException', 'DivideByZeroException',
            'FormatException', 'OverflowException', 'StackOverflowException', 'OutOfMemoryException',
            'IOException', 'FileNotFoundException', 'UnauthorizedAccessException', 'TimeoutException'
        ]

        line_references = []
        lines = self.csharp_code.splitlines()
        declared_fields = set()
        current_class = None
        constructor_start = False
        setter_start = False

        for line_no, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            line_data = {"line": line_no, "code": stripped_line, "weights": {}}

            class_declaration = re.search(r'class\s+([A-Z][\w]*)', stripped_line)
            if class_declaration:
                current_class = class_declaration.group(1)

            if current_class:
                weights = {}

                # **Detect Constructor Start**
                if re.search(constructor_pattern, stripped_line):
                    constructor_start = True
                    constructor_line = line_no

                # **Detect Setter Method Start**
                if re.search(setter_injection_pattern, stripped_line):
                    setter_start = True
                    setter_line = line_no

                # **Detect Constructor and Setter Assignments**
                if constructor_start and "=" in stripped_line and ";" in stripped_line:
                    match = re.search(r'this\.(\w+)\s*=\s*(\w+);', stripped_line)
                    if match:
                        param_name = match.group(1)
                        weights[param_name] = 1  # Constructor Assignment Weight
                        self.assignment_weights.append(
                            f"Line {line_no}: this.{param_name} = {param_name}; (Weight: 1)"
                        )

                if setter_start and "=" in stripped_line and ";" in stripped_line:
                    match = re.search(r'this\.(\w+)\s*=\s*(\w+);', stripped_line)
                    if match:
                        param_name = match.group(1)
                        weights[param_name] = 1  # Setter Assignment Weight
                        self.assignment_weights.append(
                            f"Line {line_no}: this.{param_name} = {param_name}; (Weight: 1)"
                        )
                        setter_start = False  # Reset setter detection

                # **Field Declaration (Dependency Injection)**
                field_declaration_match = re.search(field_declaration_pattern, stripped_line)
                if field_declaration_match:
                    field_class = field_declaration_match.group(1)
                    declared_fields.add(field_class)
                    weights[field_class] = 1  # DI weight

                # **Check for assignments inside constructor**
                for field_class in declared_fields:
                    if f"= new {field_class}" in stripped_line and field_class not in excluded_classes:
                        weights[field_class] = 3  # Direct Instantiation weight
                        self.direct_instantiations.append(
                            f"Line {line_no}: new {field_class}(); (Weight: 3)"
                        )

                # **Constructor Injections**
                constructor_matches = re.findall(constructor_pattern, stripped_line)
                for match in constructor_matches:
                    param_classes = re.findall(r'([A-Z][\w]*)', match)
                    for param_class in param_classes:
                        if param_class not in excluded_classes:
                            weights[param_class] = 1
                            self.constructor_injections.append(
                                f"Line {line_no}: {param_class} DI (Weight: 1)"
                            )

                # **Setter Injection**
                setter_matches = re.findall(setter_injection_pattern, stripped_line)
                for setter_class in setter_matches:
                    weights[setter_class] = 1
                    self.setter_injections.append(
                        f"Line {line_no}: {setter_class} DI (Weight: 1)"
                    )

                # **Direct Object Instantiations**
                instantiations = re.findall(instantiation_pattern, stripped_line)
                for instantiated_class in instantiations:
                    if instantiated_class not in excluded_classes:
                        weights[instantiated_class] = 3
                        self.direct_instantiations.append(
                            f"Line {line_no}: new {instantiated_class}(); (Weight: 3)"
                        )

                # **Static Method Calls**
                static_methods = re.findall(static_usage_pattern, stripped_line)
                for static_class in static_methods:
                    if static_class not in excluded_classes:
                        # Assign weight 3 for static method call only.
                        weights[static_class] = 3
                        self.static_method_calls.append(
                            f"Line {line_no}: {static_class}.method() (Weight: 3)"
                        )

                # **Static Variable Usage**
                # Remove matches that are part of a static method call to avoid double-counting.
                static_variables = re.findall(static_variable_pattern, stripped_line)
                for static_class in static_variables:
                    if static_class not in excluded_classes and static_class not in static_methods:
                        weights[static_class] = 1
                        self.static_variable_usages.append(
                            f"Line {line_no}: {static_class}.variable (Weight: 1)"
                        )

                line_data["weights"] = weights
            line_references.append(line_data)

        return line_references

    def calculate_cbo(self):
        """
        Calculates the Coupling Between Objects (CBO) value based on dependencies.
        """
        all_dependencies = set()
        for line in self.extract_class_references_with_lines():
            all_dependencies.update(line["weights"].keys())

        cbo_value = len(all_dependencies)  # Count unique dependencies

        return {
            "CBO Score": cbo_value,
            "Constructor Injections": self.constructor_injections,
            "Setter Injections": self.setter_injections,
            "Direct Object Instantiations": self.direct_instantiations,
            "Static Method Calls": self.static_method_calls,
            "Static Variable Usages": self.static_variable_usages,
            "Dependency Assignments": self.assignment_weights
        }


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
MODEL_PATH = "random_forest_reco_c#_model.pkl"
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
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score Report:")
    print(classification_report(y_test, y_pred))
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

    # Clean the new data
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

method_signature_regex_csharp = (
    r'\b(?:public|private|protected|internal)?\s*'   # Access modifier (optional)
    r'(?:static\s+|sealed\s+|virtual\s+|override\s+|async\s+)*'  # Optional modifiers
    r'(?:void|[\w<>\[\]]+)\s+'  # Return type (void, int, etc.)
    r'(?!set|get)(\w+)\s*\([^)]*\)\s*'  # Exclude property setters/getters
    r'(\s*\{)?'  # Allow optional space before opening `{`
)

# Keywords to exclude so that control structures, etc., aren't misidentified as methods.
excluded_keywords = {"if", "for", "while", "switch", "catch", "Thread", "run"}


def calculate_code_complexity_by_method_csharp(line_complexities):
    """
    Group precomputed per-line complexity data (line_complexities) into methods and
    aggregate metrics for each method using Option A.

    Each entry in line_complexities is a dictionary containing keys like:
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
        line_number = entry.get('line_number')
        line_content = entry.get('line_content', '')

        # Look for a C# method signature in the current line using our updated regex.
        match = re.search(method_signature_regex_csharp, line_content)
        if match and brace_counter == 0:
            candidate_method_name = match.group(1)
            if candidate_method_name in excluded_keywords:
                continue  # Skip lines that match control structures.
            # Finalize the previous method if any.
            if method_name and method_lines:
                methods[method_name] = aggregate_method_metrics_csharp(method_lines)
            # Start a new method.
            method_name = candidate_method_name
            method_lines = [entry]
            brace_counter = line_content.count("{") - line_content.count("}")
        elif method_name:
            # Add the current line to the method.
            method_lines.append(entry)
            brace_counter += line_content.count("{") - line_content.count("}")
            # If the braces balance, the method is complete.
            if brace_counter == 0:
                methods[method_name] = aggregate_method_metrics_csharp(method_lines)
                method_name = None
                method_lines = []

    # Finalize any method still in progress.
    if method_name and method_lines:
        methods[method_name] = aggregate_method_metrics_csharp(method_lines)

    return methods


def aggregate_method_metrics_csharp(method_lines):
    """
    Aggregate the per-line metrics of a C# method and re-calculate the overall method complexity
    using Option A.

    For Option A on a per-line basis:
      line_wcc_i = S_i * (CS_i + N_i + I_i + CC_i + TC_i + TH_i + (CBO_i - L_i))
    (We assume the 'cbo_weights' field already includes any offset for loose coupling if needed.)

    We calculate:
      - total_size = sum(S_i)
      - weighted_sum = sum( S_i * (CS_i + N_i + I_i + CC_i + TC_i + TH_i + cbo_weights) )
      - weighted_average = weighted_sum / total_size
      - total_complexity_option_a = total_size * weighted_average

    Additionally, we sum the precomputed per-line 'line_wcc' values for reference.

    :param method_lines: List of per-line complexity dictionaries for the method.
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

    # Weighted average of the inner sum.
    if total_size > 0:
        weighted_average = weighted_sum / total_size
    else:
        weighted_average = 0

    # Total complexity (Option A) for the method.
    total_complexity_option_a = total_size * weighted_average

    # Also sum the per-line precomputed complexity values (line_wcc).
    total_line_wcc = sum(entry.get("line_wcc", 0) for entry in method_lines)

    # Return aggregated metrics.
    return {
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


def extract_line_number(entry):
    """ Extracts the numeric line number from a string like 'Line 6: some code' """
    match = re.search(r'Line (\d+):', entry)
    return int(match.group(1)) if match else None


json_file = "media/c#_code_dataset.json"
output_csv1 = "media/cshar_cbo_features_output.csv"

def load_existing_csharp_dataset():
    """
    Loads existing C# dataset from JSON file. If file does not exist, creates an empty list.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Return empty list if JSON file is corrupted
    return []


def save_csharp_dataset(csharp_data):
    """
    Saves the updated C# dataset back to the JSON file.
    """
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(csharp_data, file, indent=4)


def add_csharp_code(file_name, csharp_code):
    """
    Adds new C# code to the dataset and retrains the model.
    """
    csharp_data = load_existing_csharp_dataset()

    # **Check for duplicates**
    for entry in csharp_data:
        if entry["csharp_code"].strip() == csharp_code.strip():
            print(f"⚠️ Duplicate C# code detected for {file_name}. Skipping...")
            return

    # **Append new C# code to dataset**
    csharp_data.append({"file_name": file_name, "csharp_code": csharp_code})
    save_csharp_dataset(csharp_data)

    print(f"✅ New C# code added: {file_name}")
    process_csharp_files1(json_file, output_csv1)

# 🚀 **Predict CBO & Provide Recommendations**
def get_csharp_code_recommendations(csharp_code, model_path):
    """
    Predicts CBO complexity and provides recommendations for C# code.
    Ensures feature names match the trained model.
    """

    # Load trained model
    model = joblib.load(model_path)

    # Extract CBO features
    features = calculate_cbo_csharp1(csharp_code)

    # **Expected features based on training data**
    expected_features = [
        "direct_instantiations",
        "static_method_calls",
        "static_variable_usage",
        "interface_implementations",
        "injection_initiations",
    ]

    # **Correct Mapping of Extracted Features**
    cleaned_features = {
        "direct_instantiations": len(features.get("Direct Object Instantiations", [])),
        "static_method_calls": len(features.get("Static Method Calls", [])),
        "static_variable_usage": len(features.get("Static Variable Usages", [])),
        "interface_implementations": len(features.get("Interface Implementations", [])) * 0.5,
        "injection_initiations": len(features.get("Injection Initiations", [])) * 0.5,
    }

    # Convert to DataFrame
    feature_df = pd.DataFrame([cleaned_features])

    # **Check if feature names match before prediction**
    if list(feature_df.columns) != expected_features:
        raise ValueError(f"Feature mismatch: Expected {expected_features}, but got {list(feature_df.columns)}")

    # **Predict Complexity**
    prediction = model.predict(feature_df)[0]

    # **Generate Recommendations**
    recommendations = []
    if cleaned_features["direct_instantiations"] > 5:
        recommendations.append("⚠️ Too many direct object instantiations. Use dependency injection instead.")
    if cleaned_features["static_method_calls"] > 5:
        recommendations.append("⚠️ Reduce static method calls to improve testability and flexibility.")
    if cleaned_features["static_variable_usage"] > 3:
        recommendations.append("⚠️ Minimize static variable usage to prevent global state issues.")
    if cleaned_features["injection_initiations"] > 2:
        recommendations.append("⚠️ Too many dependency injection assignments. Consider simplifying your design.")
    if cleaned_features["interface_implementations"] > 3:
        recommendations.append("⚠️ Avoid God Interfaces (interfaces with too many responsibilities).")

    return {
        "prediction": "High CBO (Issue)" if prediction == 1 else "Low CBO (Good)",
        "recommendations": recommendations
    }


def calculate_code_complexity_multiple_files_csharp(file_contents):
    results = {}
    results3 = {}
    new_patterns = []

    # 1) Possibly reuse existing code for preliminary calculations
    result1 = calculate_code_complexity_multiple(file_contents)
    mpc_results = calculate_mpc_for_csharp_code(file_contents)

    # Step 1: Track inheritance across all files
    track_inheritance_depth_across_files(file_contents)

    # 2) Iterate through each file
    for filename, content in file_contents.items():
        # add_csharp_code(filename, content)
        class_name = filename.split('.')[0]

        # Preprocessing
        lines = content.splitlines()
        complexity_data = []

        # CBO feature extraction
        cbo_report = calculate_cbo_csharp(lines)

        model_based_recommendations = get_csharp_code_recommendations(lines, model_filename)

        results3[filename] = {
            "prediction": model_based_recommendations.get("prediction", "Unknown"),
            "recommendations": model_based_recommendations.get("recommendations", [])
        }

        # 3) Retrieve any existing line-based data
        cbo_line_data = result1.get(filename, [])
        mpc_line_data = mpc_results.get(filename, [])

        # Extract references, nesting, etc.
        class_references = extract_class_references(content)
        message_passing = extract_message_passing(content)
        nesting_levels = calculate_nesting_level(content)
        nesting_level_dict = {line[0]: line[2] for line in nesting_levels}
        try_catch_weights, try_catch_weight_dict = calculate_try_catch_weight(content)
        thread_weights = calculate_thread_weight(content)

        # Compute file-level CBO & MPC
        cbo_value = calculate_cbo(class_references).get(class_name, 0)
        mpc_value = calculate_mpc(message_passing).get(class_name, 0)

        # Prepare line_complexities for AI recommendations
        line_complexities = []
        method_inheritance = {}

        # Initialize total WCC for the file
        total_wcc = 0

        # Control structure complexities
        line_weights, total_control_complexity = calculate_control_structure_complexity(lines)

        cbo_instantiation_lines = {info["line"]: info for info in cbo_report.get("Direct Object Instantiations", []) if
                                   isinstance(info, dict)}
        cbo_static_calls = {info["line"]: info for info in cbo_report.get("Static Method Calls", []) if
                            isinstance(info, dict)}
        cbo_static_vars = {info["line"]: info for info in cbo_report.get("Static Variable Usages", []) if
                           isinstance(info, dict)}
        cbo_dependency_assignment_lines = {info["line"]: info for info in cbo_report.get("Injection Initiations", []) if
                                           isinstance(info, dict)}

        # 4) Process each line (Option A)
        for line_number, line in enumerate(lines, start=1):
            # Calculate line size (token count)
            size, tokens = calculate_size(line)

            # Skip lines we don't measure
            # if any(keyword in line for keyword in ["using", "namespace", "class"]):
            #     continue
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
                })
                complexity_data.append([
                    line_number,
                    line.strip(),
                    size,
                    ', '.join(tokens),
                    0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue

            # Determine line-based CBO weight
            cbo_weight = 0
            if line_number in cbo_instantiation_lines:
                cbo_weight += 3
            if line_number in cbo_dependency_assignment_lines:
                cbo_weight += 1
            if line_number in cbo_static_calls:
                cbo_weight += 3
            if line_number in cbo_static_vars:
                cbo_weight += 3

            # Also incorporate any prior line-based data (if relevant)
            cbo_weights = cbo_line_data[line_number - 1]["weights"] if line_number - 1 < len(cbo_line_data) else {}
            total_cbo_weight = sum(cbo_weights.values())

            # MPC weight if needed
            mpc_weight = mpc_line_data.get(line_number, 0)

            # Additional metrics
            nesting_level = nesting_level_dict.get(line_number, 0)
            try_catch_weight = try_catch_weight_dict.get(line_number, 0)
            thread_weight_info = thread_weights.get(line_number, {"score": 0, "reasons": []})

            # Control structure complexity
            control_structure_complexity = line_weights.get(line_number, {"weight": 0})["weight"]

            # Inheritance
            current_inheritance = calculate_inheritance_level2(class_name)
            method_inheritance[class_name] = current_inheritance

            # Compound condition weight
            compound_condition_weight = calculate_compound_condition_weight(line)

            # Optionally detect “loose coupling” (L_i=1 if cbo_weight==1, else 0)
            if cbo_weight == 1:
                loose_coupling_offset = 1
            else:
                loose_coupling_offset = 0

            # Adjust CBO so it doesn't go negative
            adjusted_cbo = max(0, cbo_weight - loose_coupling_offset)

            # =========== Option A Calculation ===========
            # line_wcc = S_i * (CS_i + N_i + I_i + CC_i + TC_i + TH_i + adjusted_cbo)
            line_wcc = size * (
                    control_structure_complexity
                    + nesting_level
                    + current_inheritance
                    + compound_condition_weight
                    + try_catch_weight
                    + thread_weight_info['score']
                    + adjusted_cbo
            )
            total_wcc += line_wcc
            # ============================================

            # For pattern tracking (optional)
            metrics = [
                control_structure_complexity,
                nesting_level,
                compound_condition_weight,
                try_catch_weight,
                current_inheritance
            ]
            recommendation = recommend_action(metrics)

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

            # Collect line-based data for recommendations
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
                'line_wcc': line_wcc  # store the computed WCC
            })

            # Also store line data in complexity_data
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
                line_wcc  # store the computed line WCC
            ])

        # 5) Now compute method-level complexities
        method_complexities = calculate_code_complexity_by_method_csharp(
            line_complexities
        )

        # 6) If new patterns appear, update dataset & model
        # if new_patterns:
        #     new_data = pd.DataFrame(new_patterns)
        #     update_dataset_and_model(new_data)

        # 7) AI recommendations
        recommendations = ai_recommend_refactoring(line_complexities)
        filtered_recommendations = [
            rec for rec in recommendations if rec['recommendation'] != "No action needed"
        ]

        # 8) Plotting or analysis
        complexity_factors = calculate_complexity_factors(filename, complexity_data)
        pie_chart_path = plot_complexity_pie_chart(filename, complexity_factors)

        # Generate bar charts per method
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
            print(f"Bar chart generated for method '{method_name}': {bar_chart_path}")

        # 9) Final assembly of results
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
