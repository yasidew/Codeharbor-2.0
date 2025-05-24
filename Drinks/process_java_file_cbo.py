import javalang
import pandas as pd
import json
import re
from pygments.token import Token
from pygments.lexers import CSharpLexer
from pygments import lex


JAVA_STANDARD_CLASSES = {
    "Thread", "String", "HashMap", "BufferedReader", "FileReader", "Runnable", "System.out", "Logger", "Math",
    "InputStreamReader", "URL", "StringBuilder", "LinkedList", "HashSet",
    "Stack", "Properties", "FileInputStream", "Random", "Exception",
    "RuntimeException", "ArithmeticException", "String", "System", "Math", "Integer", "Double", "Float", "Boolean",
    "Character",
    "Long", "Short", "Byte", "BigInteger", "BigDecimal", "Object", "Collections",
    "Arrays", "List", "ArrayList", "HashMap", "HashSet", "LinkedList", "Map", "Set", "Sort", "PageRequest", "UUID",
    "Files", "Paths", "Runtime", "Optional", "Objects", "LocalDate", "Year", "IntStream", "Scanner",

    "Exception", "RuntimeException", "ArithmeticException", "NullPointerException",
    "ArrayIndexOutOfBoundsException", "IndexOutOfBoundsException",
    "IllegalArgumentException", "IllegalStateException", "ClassNotFoundException",
    "NoClassDefFoundError", "NumberFormatException", "FileNotFoundException",
    "IOException", "InterruptedException", "CloneNotSupportedException",
    "AssertionError", "StackOverflowError", "OutOfMemoryError", "SecurityException",
    "UnsupportedOperationException", "SQLException", "TimeoutException",
    "UncheckedIOException", "ConcurrentModificationException","ObjectOutputStream", "FileInputStream"
}


def extract_cbo_features(java_code):
    """
    Extracts  quality related to CBO (Coupling Between Objects) for Java.
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
                        cbo_features["constructor_injections"] += 0.5

            # ✅ Detect Setter Injections (Methods starting with "set")
            if isinstance(node, javalang.tree.MethodDeclaration):
                if "set" in node.name.lower():
                    for param in node.parameters:
                        if hasattr(param, "type") and hasattr(param.type, "name"):
                            class_name = param.type.name
                            cbo_features["class_dependencies"].add(class_name)
                            cbo_features["setter_injections"] += 0.5

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
                            cbo_features["interface_implementations"] += 0.5

            # ✅ Detect Method Parameters as Dependencies (NEW FIX)
            if isinstance(node, javalang.tree.MethodDeclaration):
                for param in node.parameters:
                    if hasattr(param, "type") and hasattr(param.type, "name"):
                        class_name = param.type.name
                        cbo_features["class_dependencies"].add(class_name)

    except Exception as e:
        print(f"Error parsing Java code: {e}")

    # Convert class_dependencies from a set to a count--
    cbo_features["class_dependencies"] = len(cbo_features["class_dependencies"])
    return cbo_features


class CBOMetrics1:
    def __init__(self, java_code):
        """
        Parses Java code and extracts:
        - Constructor & Setter Injection (Weight = 1)
        - Direct Object Instantiation (new Keyword) (Weight = 3)
        - Static Method Calls (Weight = 1)
        - Static Variable Usages (Weight = 1)
        - Interface Implementations (Weight = 1)
        - Global Variable References (Weight = 1)
        """
        self.tree = javalang.parse.parse(java_code)
        self.dependencies = set()
        self.primitive_types = {"byte", "short", "int", "long", "float", "double", "boolean", "char"}
        self.built_in_classes = {
            "Thread", "String", "HashMap", "BufferedReader", "FileReader", "StopWatch", "DuplicateRecordException",
            "Runnable", "System.out", "System.err", "Logger", "Math", "InputStreamReader", "URL", "StringBuilder",
            "LinkedList", "HashSet", "Stack", "Properties", "FileInputStream", "Random", "Exception",
            "RuntimeException", "ArithmeticException", "System", "Integer", "Double", "Float", "Boolean", "Character",
            "Long", "Short", "Byte", "BigInteger", "BigDecimal", "Object", "Collections", "Arrays", "List",
            "ArrayList", "HashMap", "HashSet", "LinkedList", "Map", "Set", "Sort", "PageRequest", "UUID", "Files",
            "Paths", "Runtime", "Optional", "Objects", "LocalDate", "Year", "IntStream", "Scanner",
            "NumberFormatException", "RecordNotFoundException", "BadRequestException", "InputStreamResource",
            "InvalidFileTypeException", "IllegalArgumentException", 'ArrayList', 'Hashtable', 'Queue', 'Stack',
            'SortedList', 'List', 'Dictionary', 'SortedDictionary', 'SortedList', 'Queue', 'Stack', 'HashSet',
            'SortedSet', 'ConcurrentBag', 'ConcurrentQueue', 'ConcurrentStack', 'ConcurrentDictionary', "Timer", "Random","ExecutorService", "URL"
        }
        self.constructor_injections = {}
        self.setter_injections = {}
        self.direct_instantiations = []
        self.assignment_weights = []
        self.static_method_calls = []
        self.static_variable_usages = []
        self.interface_implementations = []
        self.global_variable_references = []

    def extract_dependencies(self):
        """
        Extracts constructor injections, setter injections, direct instantiations,
        static method calls, static variable usages, interface implementations, and global variable references.
        """
        # --- Constructor Injection (DI, Weight = 1)--
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
            if class_name not in self.primitive_types and class_name not in self.built_in_classes:
                self.dependencies.add(class_name)
                self.direct_instantiations.append({
                    "line": instantiation_line,
                    "instantiation": f"new {class_name}();",
                    "weight": 2
                })

        # --- Static Method Calls (Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.MethodInvocation):
            if hasattr(node, "qualifier") and node.qualifier:
                qualifier = str(node.qualifier)
                if qualifier and qualifier[0].isupper() and qualifier not in self.built_in_classes:
                    self.dependencies.add(qualifier)
                    method_call_line = node.position.line if node.position else "Unknown"
                    self.static_method_calls.append({
                        "line": method_call_line,
                        "method_call": f"{qualifier}.{node.member}()",
                        "weight": 2
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
                        "weight": 2
                    })

        # --- Interface Implementations (Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.ClassDeclaration):
            if node.implements:
                for implemented_interface in node.implements:
                    if hasattr(implemented_interface, "name"):
                        interface_name = implemented_interface.name
                        self.dependencies.add(interface_name)
                        self.interface_implementations.append({
                            "interface": interface_name,
                            "weight": 1
                        })

        # --- Global Variable References (Weight = 1) ---
        for _, node in self.tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                field_name = declarator.name
                global_variable_line = node.position.line if node.position else "Unknown"
                self.global_variable_references.append({
                    "line": global_variable_line,
                    "global_variable": field_name,
                    "weight": 1
                })

    def calculate_cbo(self):
        """
        Calculates the Coupling Between Objects (CBO) value based on unique dependencies.
        """
        cbo_value = len(self.dependencies)
        return cbo_value

    def get_cbo_report(self):
        """
        Extracts dependencies, calculates CBO, and returns a structured report.
        """
        self.extract_dependencies()
        cbo_value = self.calculate_cbo()

        report = {
            "CBO Score": cbo_value,
            "Interface Implementations": self.interface_implementations,
            "Global Variable References": self.global_variable_references,
            "Constructor Injections": self.constructor_injections,
            "Setter Injections": self.setter_injections,
            "Direct Object Instantiations": self.direct_instantiations,
            "Static Method Calls": self.static_method_calls,
            "Static Variable Usages": self.static_variable_usages,
            "Injection Initiations": self.assignment_weights,
        }
        return report


def extract_cbo_features1(java_code):
    """
    Extracts CBO-related features from Java code using CBOMetrics1.
    Returns a dictionary with counts of each CBO factor.
    """
    cbo_metrics = CBOMetrics1(java_code)
    report = cbo_metrics.get_cbo_report()

    return {
        "direct_instantiations": len(report["Direct Object Instantiations"]),
        "static_method_calls": len(report["Static Method Calls"]),
        "static_variable_usage": len(report["Static Variable Usages"]),
        "interface_implementations": len(report["Interface Implementations"]),
        "assignment_weights": len(report["Injection Initiations"])
    }

def process_java_files(json_file, output_csv):
    """
    Reads Java code examples from a JSON file, extracts CBO features,
    and  them to a CSV file with dynamically assigned CBO labels.
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        java_data = json.load(file)

    csv_data = []

    for entry in java_data:
        file_name = entry["file_name"]
        java_code = entry["java_code"]

        cbo_features = extract_cbo_features1(java_code)

        WEIGHT_INTERFACE = 0.5

        # Apply decrement by 1 to specific factors
        interface_implementations = WEIGHT_INTERFACE * max(0, cbo_features["interface_implementations"])
        # constructor_injections = WEIGHT_INTERFACE *  max(0, cbo_features["constructor_injections"])
        # setter_injections = WEIGHT_INTERFACE * max(0, cbo_features["setter_injections"])
        assignment_weights = WEIGHT_INTERFACE * max(0, cbo_features["assignment_weights"])

        # Collect all CBO factor values (after decrement)
        cbo_values = [
            # cbo_features["class_dependencies"],
            cbo_features["direct_instantiations"],
            cbo_features["static_method_calls"],
            cbo_features["static_variable_usage"],
            interface_implementations,  # Decrement applied
            assignment_weights
            # constructor_injections,  # Decrement applied
            # setter_injections,  # Decrement applied
            # cbo_features["global_variable_references"]
        ]

        csv_data.append([file_name, *cbo_values])  # Unpacking feature values

    df = pd.DataFrame(csv_data, columns=[
        "file_name", "direct_instantiations", "static_method_calls",
        "static_variable_usage", "interface_implementations", "assignment_weights"
    ])

    # Compute dynamic CBO threshold (90th percentile)
    cbo_sum = df.iloc[:, 1:].sum(axis=1)  # Sum of all CBO features per file (after decrement adjustments)
    # print("scbo_sum>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", cbo_sum)
    dynamic_threshold = cbo_sum.quantile(0.85)  # 90th percentile for thresholding

    df["cbo_label"] = cbo_sum.apply(lambda x: 1 if x > dynamic_threshold else 0)

    # Save updated
    df.to_csv(output_csv, index=False)

    print(f"CSV file '{output_csv}' generated successfully with dynamic threshold: {dynamic_threshold}")


def extract_cbo_features_csharp(csharp_code):
    """
    Extracts software quality metrics related to CBO (Coupling Between Objects) in C#.
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

    # ✅ Exclude built-in classes like Console, Math, and System
    excluded_classes = {"Console", "Math", "System", "Thread", "Task", "Runtime"}

    # ✅ Regex Patterns for Feature Extraction
    instantiation_pattern = r'new\s+([A-Z][\w]*)\s*\('  # Detects `new ClassName()`
    static_method_pattern = r'\b([A-Z][\w]*)\s*\.\s*([A-Z]\w*)\s*\('  # Detects `ClassName.Method()`
    static_variable_pattern = r'([A-Z][\w]*)\.\w+\s*='  # Detects `ClassName.Variable = Value`
    constructor_pattern = r'\bpublic\s+[A-Z][\w]*\s*\(([^)]*)\)'  # Constructor Injection
    setter_pattern = r'public\s+void\s+set[A-Z]\w*\s*\(([^)]*)\)'  # Setter Injection
    class_pattern = r'class\s+([A-Z][\w]*)'  # Class Declaration
    interface_pattern = r'class\s+\w+\s*:\s*([\w\s,]+)'  # Implements Interfaces

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


def calculate_cbo_csharp1(lines):
    """
    Calculates Coupling Between Objects (CBO) in C# code using Pygments tokenization.
    Now includes interface implementation detection.

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
    interface_implementations = []  # Track implemented interfaces

    primitive_types = {"byte", "short", "int", "long", "float", "double", "bool", "char"}
    built_in_classes = {'Console', 'String', 'int', 'float', 'double', 'bool', 'char', 'byte', 'short', 'long',
                        'Math', 'Object', 'Thread', 'Runtime', 'Optional', 'Task', 'Action', 'Func', 'Exception',
                        'SystemException', 'InvalidOperationException', 'ArgumentException',
                        'NullReferenceException', 'IndexOutOfRangeException', 'DivideByZeroException',
                        'FormatException', 'OverflowException', 'StackOverflowException', 'OutOfMemoryException',
                        'IOException', 'FileNotFoundException', 'UnauthorizedAccessException', 'TimeoutException', 'File', 'List', 'ArrayList'}

    direct_instantiation_pattern = re.compile(r"\bnew\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    static_variable_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
    static_method_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\(")

    interface_pattern = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_,\s]+)")  # Interface detection

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        tokens = list(lex(stripped_line, CSharpLexer()))

        if not stripped_line:
            continue

        # ✅ Detect Interface Implementations (Weight = 1)
        match = interface_pattern.search(stripped_line)
        if match:
            class_name = match.group(1)
            interfaces = match.group(2).split(",")

            for interface in interfaces:
                interface = interface.strip()
                if interface and interface not in primitive_types and interface not in built_in_classes:
                    dependencies.add(interface)
                    interface_implementations.append({
                        "class": class_name,
                        "interface": interface,
                        "line": line_number,
                        "weight": 1
                    })

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

        # ✅ Direct Object Instantiation (Weight = 3)
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

        # ✅ Static Method Calls (Weight = 1)
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
        "Interface Implementations": interface_implementations,
        "Constructor Injections": constructor_injections,
        "Setter Injections": setter_injections,
        "Direct Object Instantiations": direct_instantiations,
        "Static Method Calls": static_method_calls,
        "Static Variable Usages": static_variable_usages,
        "Injection Initiations": injection_initiations
    }



def process_csharp_files(json_file, output_csv):
    """
    Reads C# code examples from a JSON file, extracts CBO features,
    and writes them to a CSV file with dynamically assigned CBO labels.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            csharp_data = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error: Unable to read JSON file '{json_file}'. Reason: {e}")
        return

    csv_data = []

    for entry in csharp_data:
        file_name = entry.get("file_name", "unknown_file")
        csharp_code = entry.get("csharp_code", None)

        if csharp_code is None:
            print(f"⚠️ Warning: 'csharp_code' not found in entry {entry}. Skipping...")
            continue  # Skip this entry

        cbo_features = extract_cbo_features_csharp(csharp_code)

        # Collect all CBO factor values
        cbo_values = [
            cbo_features["class_dependencies"],
            cbo_features["direct_instantiations"],
            cbo_features["static_method_calls"],
            cbo_features["static_variable_usage"],
            cbo_features["interface_implementations"],
            cbo_features["constructor_injections"],
            cbo_features["setter_injections"],
            cbo_features["global_variable_references"]
        ]

        # Store extracted values
        csv_data.append([
            file_name,
            *cbo_values  # Unpacking feature values
        ])

    # Convert data to Pandas DataFrame
    df = pd.DataFrame(csv_data, columns=[
        "file_name", "class_dependencies", "direct_instantiations", "static_method_calls",
        "static_variable_usage", "interface_implementations", "constructor_injections",
        "setter_injections", "global_variable_references"
    ])

    # Compute dynamic CBO threshold (75th percentile)
    cbo_sum = df.iloc[:, 2:].sum(axis=1)  # Sum of all CBO features per file
    dynamic_threshold = cbo_sum.quantile(0.90)  # 75th percentile

    # Assign CBO label dynamically
    df["cbo_label"] = cbo_sum.apply(lambda x: 1 if x > dynamic_threshold else 0)

    # Save updated CSV
    df.to_csv(output_csv, index=False)

    print(f"✅ CSV file '{output_csv}' generated successfully with dynamic threshold: {dynamic_threshold}")


def process_csharp_files1(json_file, output_csv):
    """
    Reads C# code examples from a JSON file, extracts CBO features using calculate_cbo_csharp1,
    and writes them to a CSV file with dynamically assigned CBO labels.

    - Uses "Injection Initiations" instead of constructor/setter injection.
    - Computes dynamic CBO threshold (90th percentile) for labeling.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            csharp_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error: Unable to read JSON file '{json_file}'. Reason: {e}")
        return

    csv_data = []

    for entry in csharp_data:
        file_name = entry.get("file_name", "unknown_file")
        csharp_code = entry.get("csharp_code", None)

        if csharp_code is None:
            print(f"⚠️ Warning: 'csharp_code' not found in entry {entry}. Skipping...")
            continue  # Skip this entry

        # Extract CBO features using calculate_cbo_csharp1
        cbo_features = calculate_cbo_csharp1(csharp_code.split('\n'))

        WEIGHT_INTERFACE = 0.5

        # Ensure numeric values
        direct_instantiations = len(cbo_features.get("Direct Object Instantiations", []))
        static_method_calls = len(cbo_features.get("Static Method Calls", []))
        static_variable_usage = len(cbo_features.get("Static Variable Usages", []))
        interface_implementations = len(cbo_features.get("Interface Implementations", [])) * WEIGHT_INTERFACE
        injection_initiations = len(cbo_features.get("Injection Initiations", [])) * WEIGHT_INTERFACE

        # Collect all CBO factor values
        cbo_values = [
            direct_instantiations,
            static_method_calls,
            static_variable_usage,
            interface_implementations,
            injection_initiations,
        ]

        # Store extracted values
        csv_data.append([
            file_name,
            *cbo_values  # Unpacking feature values
        ])

    # Convert data to Pandas DataFrame
    df = pd.DataFrame(csv_data, columns=[
        "file_name", "direct_instantiations", "static_method_calls",
        "static_variable_usage", "interface_implementations", "injection_initiations"
    ])

    # Compute dynamic CBO threshold (90th percentile)
    cbo_sum = df.iloc[:, 1:].sum(axis=1)  # Sum of all CBO features per file
    dynamic_threshold = cbo_sum.quantile(0.90)  # 90th percentile

    # Assign CBO label dynamically
    df["cbo_label"] = cbo_sum.apply(lambda x: 1 if x > dynamic_threshold else 0)

    # Save updated CSV
    df.to_csv(output_csv, index=False)

    print(f"✅ CSV file '{output_csv}' generated successfully with dynamic threshold: {dynamic_threshold}")
