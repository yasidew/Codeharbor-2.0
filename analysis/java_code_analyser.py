import javalang
import re

class JavaCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # Java code to be analyzed

    def custom_rules_check(self):
        # Parse the Java code into an abstract syntax tree (AST)
        tree = javalang.parse.parse(self.code)
        # Check custom rules (e.g., method length)
        self._check_method_length(tree)
        self._check_too_many_parameters(tree)
        self._check_cyclomatic_complexity(tree)
        self._check_unused_variables(tree)
        self._check_nesting_depth(tree)
        self._check_hard_coded_secrets(tree)
        self._check_sensitive_data_in_logs(tree)
        self._check_inefficient_data_structures(tree)
        self._check_long_chained_calls(tree)
        self._check_global_variable_overuse(tree)
        self._check_deprecated_libraries(tree)
        self._check_code_duplication()
        self._check_lack_of_comments(tree)
        self._check_magic_numbers(tree)
        self._check_exception_handling(tree)
        self._check_recursive_methods(tree)
        self._check_unreachable_code(tree)
        self._check_missing_default_case(tree)
        self._check_circular_imports(tree)
        self._check_deprecated_syntax(tree)
        self._check_unused_imports(tree)

    def _check_method_length(self, tree):
        # Check if any method in the code has more than 10 lines
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if len(node.body) > 10:
                print(f"Method {node.name} is too long at line {node.position.line}")

    def _check_too_many_parameters(self, tree):
        # Check if any method has more than 5 parameters
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if len(node.parameters) > 5:
                print(f"Method {node.name} has too many parameters at line {node.position.line}")

    def _check_cyclomatic_complexity(self, tree):
        # Custom rule to check cyclomatic complexity of methods
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            complexity = sum(1 for n in node.body if isinstance(n, (javalang.tree.IfStatement, javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)))
            if complexity > 10:
                print(f"Method {node.name} at line {node.position.line} has high cyclomatic complexity: {complexity}")

    def _check_unused_variables(self, tree):
        # Detect unused variables
        declared = []
        used = set()
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            declared.append(node.name)
        for path, node in tree.filter(javalang.tree.MemberReference):
            used.add(node.member)
        unused = [var for var in declared if var not in used]
        if unused:
            print(f"Unused variables: {unused}")

    def _check_nesting_depth(self, node, depth=0, max_depth=4):
        # Check for excessive nesting in the code
        if depth > max_depth:
            print(f"Excessive nesting found at line {node.position.line}")
        for child in node.children:
            if isinstance(child, javalang.ast.Node):
                self._check_nesting_depth(child, depth + 1, max_depth)

    def _check_hard_coded_secrets(self, tree):
        # Custom rule to check for hardcoded secrets in the code
        secret_patterns = [r"API_KEY\s*=\s*[\"'].*[\"']", r"PASSWORD\s*=\s*[\"'].*[\"']"]
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            if isinstance(node.initializer, javalang.tree.Literal):
                for pattern in secret_patterns:
                    if re.search(pattern, node.initializer.value):
                        print(f"Hardcoded secret detected at line {node.position.line}: {node.initializer.value}")

    def _check_sensitive_data_in_logs(self, tree):
        # Check for sensitive data exposed in logs
        sensitive_keywords = ["password", "token", "api_key"]
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member == "println":
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and any(kw in arg.value.lower() for kw in sensitive_keywords):
                        print(f"Sensitive data exposed in print statement at line {node.position.line}: {arg.value}")

    def _check_inefficient_data_structures(self, tree):
        # Check for inefficient data structures
        for path, node in tree.filter(javalang.tree.BinaryOperation):
            if node.operator == "in" and isinstance(node.right, javalang.tree.ArrayInitializer):
                print(f"Inefficient membership check using an array at line {node.position.line}")

    def _check_long_chained_calls(self, tree):
        # Check for long chained method calls
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            chain_length = 0
            current = node
            while isinstance(current, javalang.tree.MethodInvocation):
                chain_length += 1
                current = current.qualifier
            if chain_length > 3:
                print(f"Long chained method call of length {chain_length} at line {node.position.line}")

    def _check_global_variable_overuse(self, tree):
        # Check for excessive use of global variables
        global_count = sum(1 for path, node in tree.filter(javalang.tree.FieldDeclaration))
        if global_count > 5:
            print(f"Excessive use of global variables: {global_count}")

    def _check_deprecated_libraries(self, tree):
        # Check for deprecated libraries
        deprecated_libs = ["java.util.Random", "java.lang.System"]  # Example: add more based on research
        for path, node in tree.filter(javalang.tree.Import):
            if node.path in deprecated_libs:
                print(f"Deprecated library {node.path} imported at line {node.position.line}")

    def _check_code_duplication(self):
        lines = self.code.split("\n")
        duplicates = {}
        for i in range(len(lines)):
            snippet = "\n".join(lines[i:i + 5])  # Compare blocks of 5 lines
            if snippet in duplicates:
                duplicates[snippet].append(i + 1)
            else:
                duplicates[snippet] = [i + 1]
        for snippet, occurrences in duplicates.items():
            if len(occurrences) > 1:
                print(f"Duplicate code block found at lines {occurrences}")

    def _check_lack_of_comments(self, tree):
        # Check for lack of comments/docstrings (no proper documentation)
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not node.documentation:
                print(f"Method '{node.name}' at line {node.position.line} lacks a docstring.")

    def _check_magic_numbers(self, tree, allowed=[0, 1, -1]):
        # Check for magic numbers in the code (means hard-coded numbers)
        for path, node in tree.filter(javalang.tree.Literal):
            if isinstance(node.value, (int, float)) and node.value not in allowed:
                print(f"Magic number {node.value} found at line {node.position.line}. Consider defining it as a constant.")

    def _check_exception_handling(self, tree):
        # Check for unused imports in the code
        for path, node in tree.filter(javalang.tree.TryStatement):
            if not node.catches:
                print(f"Try block at line {node.position.line} lacks exception handling.")
            for handler in node.catches:
                if handler.parameter.name == "Exception":
                    print(f"Overly broad exception handling found at line {handler.position.line}")

    def _check_recursive_methods(self, tree):
        # Check for recursive methods without a proper base case to avoid infinite recursion
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            for sub_node in node.body:
                if isinstance(sub_node, javalang.tree.MethodInvocation) and sub_node.member == node.name:
                    print(f"Recursive method '{node.name}' detected at line {node.position.line}")

    def _check_unreachable_code(self, tree):
        # Check for unreachable code after return, raise, or break statements
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            for i, stmt in enumerate(node.body[:-1]):
                if isinstance(stmt, (javalang.tree.ReturnStatement, javalang.tree.ThrowStatement, javalang.tree.BreakStatement)):
                    unreachable = node.body[i+1:]
                    if unreachable:
                        print(f"Unreachable code detected after line {stmt.position.line}")

    def _check_missing_default_case(self, tree):
        # Check for missing default case in switch-statements
        for path, node in tree.filter(javalang.tree.SwitchStatement):
            has_default = any(isinstance(n, javalang.tree.SwitchStatementCase) and n.case is None for n in node.cases)
            if not has_default:
                print(f"Missing default case for switch-statement at line {node.position.line}")

    def _check_circular_imports(self, tree):
        # Check for circular imports in the code
        imports = []
        for path, node in tree.filter(javalang.tree.Import):
            imports.append(node.path)
        if len(imports) != len(set(imports)):
            print("Circular import detected.")

    def _check_deprecated_syntax(self, tree):
        # Custom rule to check for deprecated syntax
        deprecated_nodes = (javalang.tree.PrintStatement, javalang.tree.ExecStatement)  # Example: old Java syntax
        for path, node in tree.filter(deprecated_nodes):
            print(f"Deprecated syntax {type(node).__name__} found at line {node.position.line}")

    def _check_unused_imports(self, tree):
        # Check for unused imports in the code
        imports = {node.path: node.position.line for path, node in tree.filter(javalang.tree.Import)}
        used = set()
        for path, node in tree.filter(javalang.tree.MemberReference):
            used.add(node.member)
        unused = [imp for imp in imports if imp not in used]
        for imp in unused:
            print(f"Unused import '{imp}' at line {imports[imp]}")

# Example usage
if __name__ == '__main__':
    with open('path_to_your_java_file.java', 'r') as file:
        code = file.read()  # Read the Java code to be analyzed

    analyzer = JavaCodeAnalyzer(code)
    analyzer.custom_rules_check()  # Run custom rules check (e.g., method length)