import json

import pylint.lint
import ast
import re


class CodeAnalyzer:
    def __init__(self, code):
        self.defined_variables = [] # List to store defined variables
        self.code = code  # Code to be analyzed
        self.recommendations =[]  # List to store recommendations

    def generate_recommendations(self):
        # Parse the code into an AST
        try:
            tree = ast.parse(self.code)
        except SyntaxError as e:
            return [f"Syntax Error: {e}"]



        # Run pylint and custom rules check
        self.run_pylint()
        self.custom_rules_check(tree)
        return self.recommendations

    def run_pylint(self):
        # Pylint options for checking general coding standards
        pylint_opts = ['--disable=all', '--enable=E,W,C,R', '--output-format=json']
        pylint_output = pylint.lint.Run(pylint_opts, do_exit=False)
        pylint_json = json.loads(pylint_output.linter.reporter.data)
        for issue in pylint_json:
            self.recommendations.append(f"Pylint: {issue['message']} at line {issue['line']}")
        # pylint.lint.Run(pylint_opts)

    def custom_rules_check(self):
        # Parse the code into an abstract syntax tree (AST)
        tree = ast.parse(self.code)
        # Check custom rules (e.g., function length)
        self._check_function_length(tree)
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
        self._check_recursive_functions(tree)
        self._check_unreachable_code(tree)
        self._check_missing_default_case(tree)
        self._check_circular_imports(tree)
        self._check_deprecated_syntax(tree)
        self._check_unused_imports(tree)
        self._check_sql_injection(tree)
        self._check_command_injection(tree)
        self._check_xss(tree)
        self._check_weak_cryptographic_practices(tree)
        self._check_insecure_file_handling(tree)
        self._check_session_security(tree)
        self._check_unvalidated_redirects(tree)
        self._check_package_naming_convention(self.code.splitlines())
        self._check_class_naming_convention(tree)
        self._check_method_naming_convention(tree)
        self._check_thread_safety_violations(tree)
        self._check_inefficient_loops(tree)
        self._check_potential_null_pointers(tree)
        self._check_resource_leaks(tree)
        self._check_empty_catch_blocks(tree)



    def _check_function_length(self, tree):
        # recommendations = []
        # Check if any function in the code has more than 10 lines
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 10:
                    self.recommendations.append({
                        "rule": "Function Length",
                        "message": f"Function {node.name} is too long at line {node.lineno}",
                        "line": node.lineno,
                    })
        # return  recommendations
        # print(f"Function {node.name} is too long at line {node.lineno}")

    def _check_too_many_parameters(self, tree):
        # recommendations = []
        # Check if any function has more than 5 parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    self.recommendations.append({
                        "rule": "Too Many Parameters",
                        "message": f"Function {node.name} has too many parameters at line {node.lineno}",
                        "line": node.lineno,
                    })
        # return recommendations
                    # print(f"Function {node.name} has too many parameters at line {node.lineno}")

    # Custom rule to check cyclomatic complexity of functions
    def _check_cyclomatic_complexity(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.BoolOp)))
                if complexity > 10:
                    self.recommendations.append({
                        "rule": "Cyclomatic Complexity",
                        "message": f"Function {node.name} has high cyclomatic complexity at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Function {node.name} at line {node.lineno} has high cyclomatic complexity: {complexity}")

    # detect unused variables or imports
    def _check_unused_variables(self, tree):
        declared = []
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                declared.append(node.targets[0].id if hasattr(node.targets[0], 'id') else None)
            elif isinstance(node, ast.Name):
                used.add(node.id)

        unused = [var for var in declared if var not in used]
        if unused:
            self.recommendations.append({
                "rule": "Unused Variables",
                "message": f"Unused variables: {unused}",
                "line": None,
            })
            # print(f"Unused variables: {unused}")

    # Check for excessive nesting in the code
    def _check_nesting_depth(self, node, depth=0, max_depth=4):
        if depth > max_depth:
            self.recommendations.append({
                "rule": "Excessive Nesting",
                "message": f"Excessive nesting found at line {node.lineno}",
                "line": node.lineno,
            })
            # print(f"Excessive nesting found at line {node.lineno}")
        for child in ast.iter_child_nodes(node):
            self._check_nesting_depth(child, depth + 1, max_depth)

    # Custom rule to check for hardcoded secrets in the code
    def _check_hard_coded_secrets(self, tree):
        secret_patterns = [r"API_KEY\s*=\s*[\"'].*[\"']", r"PASSWORD\s*=\s*[\"'].*[\"']"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Str):
                for pattern in secret_patterns:
                    if re.search(pattern, node.value.s):
                        # print(f"Hardcoded secret detected at line {node.lineno}: {node.value.s}")
                        self.recommendations.append({
                            "rule": "Hardcoded Secrets",
                            "message": f"Hardcoded secret detected at line {node.lineno}: {node.value.s}",
                            "line": node.lineno,
                        })

    # Check for sensitive data exposed in logs
    def _check_sensitive_data_in_logs(self, tree):
        sensitive_keywords = ["password", "token", "api_key"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                for arg in node.args:
                    if isinstance(arg, ast.Str) and any(kw in arg.s.lower() for kw in sensitive_keywords):
                        self.recommendations.append({
                            "rule": "Sensitive Data in Logs",
                            "message": f"Sensitive data exposed in print statement at line {node.lineno}: {arg.s}",
                            "line": node.lineno,
                        })
                        # print(f"Sensitive data exposed in print statement at line {node.lineno}: {arg.s}")


    # Check for inefficient data structures
    def _check_inefficient_data_structures(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In):
                if isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.List):
                    self.recommendations.append({
                        "rule": "Inefficient Data Structures",
                        "message": f"Inefficient membership check using a list at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Inefficient membership check using a list at line {node.lineno}")

    # Check for long chained method calls
    def _check_long_chained_calls(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain_length = 0
                current = node
                while isinstance(current, ast.Attribute):
                    chain_length += 1
                    current = current.value
                if chain_length > 3:
                    self.recommendations.append({
                        "rule": "Long Chained Method Calls",
                        "message": f"Long chained method call of length {chain_length} at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Long chained method call of length {chain_length} at line {node.lineno}")

    # Check for excessive use of global variables
    def _check_global_variable_overuse(self, tree):
        global_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Global))
        if global_count > 5:
            self.recommendations.append({
                "rule": "Excessive Use of Global Variables",
                "message": f"Excessive use of global variables: {global_count}",
                "line": None,
            })
            # print(f"Excessive use of global variables: {global_count}")

    # Check for deprecated libraries
    def _check_deprecated_libraries(self, tree):
        deprecated_libs = ["random", "os.system"]  # Example: add more based on research
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in deprecated_libs:
                        self.recommendations.append({
                            "rule": "Deprecated Libraries",
                            "message": f"Deprecated library {alias.name} imported at line {node.lineno}",
                            "line": node.lineno,
                        })
                        # print(f"Deprecated library {alias.name} imported at line {node.lineno}")

    # Check for code duplication
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
                self.recommendations.append({
                    "rule": "Code Duplication",
                    "message": f"Duplicate code block found at lines {occurrences}",
                    "line": None,
                })
                # print(f"Duplicate code block found at lines {occurrences}")

    # Check for lack of comments/docstrings (no proper documentation)
    def _check_lack_of_comments(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                    self.recommendations.append({
                        "rule": "Lack of Comments",
                        "message": f"{type(node).__name__} '{node.name}' at line {node.lineno} lacks a docstring.",
                        "line": node.lineno,
                    })
                    # print(f"{type(node).__name__} '{node.name}' at line {node.lineno} lacks a docstring.")

    # Check for magic numbers in the code (means hard-coded numbers)
    def _check_magic_numbers(self, tree, allowed=[0, 1, -1]):
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in allowed:
                    self.recommendations.append({
                        "rule": "Magic Numbers",
                        "message": f"Magic number {node.value} found at line {node.lineno}. Consider defining it as a constant.",
                        "line": node.lineno,
                    })
                    # print(f"Magic number {node.value} found at line {node.lineno}. Consider defining it as a constant.")

    # Check for unused imports in the code
    def _check_exception_handling(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                if not node.handlers:
                    self.recommendations.append({
                        "rule": "Exception Handling",
                        "message": f"Try block at line {node.lineno} lacks exception handling.",
                        "line": node.lineno,
                    })
                    # print(f"Try block at line {node.lineno} lacks exception handling.")
                for handler in node.handlers:
                    if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                        self.recommendations.append({
                            "rule": "Exception Handling",
                            "message": f"Overly broad exception handling found at line {handler.lineno}",
                            "line": handler.lineno,
                        })
                        # print(f"Overly broad exception handling found at line {handler.lineno}")

    # Check for recursive functions without a proper base case to avoid infinite recursion
    def _check_recursive_functions(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Call) and isinstance(sub_node.func,
                                                                     ast.Name) and sub_node.func.id == node.name:
                        self.recommendations.append({
                            "rule": "Recursive Functions",
                            "message": f"Recursive function '{node.name}' detected at line {node.lineno}",
                            "line": node.lineno,
                        })
                        # print(f"Recursive function '{node.name}' detected at line {node.lineno}")


    # Check for unreachable code after return, raise, or break statements
    def _check_unreachable_code(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Break)):
                        unreachable = node.body[i+1:]
                        if unreachable:
                            self.recommendations.append({
                                "rule": "Unreachable Code",
                                "message": f"Unreachable code detected after line {stmt.lineno}",
                                "line": stmt.lineno,
                            })
                            # print(f"Unreachable code detected after line {stmt.lineno}")


    # Check for missing default case in if-statements
    def _check_missing_default_case(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                has_else = any(isinstance(n, ast.If) and n.orelse for n in ast.walk(node))
                if not has_else:
                    self.recommendations.append({
                        "rule": "Missing Default Case",
                        "message": f"Missing default case for if-statement at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Missing default case for if-statement at line {node.lineno}")


    # Check for circular imports in the code
    def _check_circular_imports(self, tree):
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        if len(imports) != len(set(imports)):
            self.recommendations.append({
                "rule": "Circular Imports",
                "message": "Circular import detected.",
                "line": None,
            })
            # print("Circular import detected.")


    # Custom rule to check for deprecated syntax
    def _check_deprecated_syntax(self, tree):
        deprecated_nodes = (ast.Print, ast.Exec)  # Example: old Python 2 syntax
        for node in ast.walk(tree):
            if isinstance(node, deprecated_nodes):
                self.recommendations.append({
                    "rule": "Deprecated Syntax",
                    "message": f"Deprecated syntax {type(node).__name__} found at line {node.lineno}",
                    "line": node.lineno,
                })
                # print(f"Deprecated syntax {type(node).__name__} found at line {node.lineno}")


    # Check for unused imports in the code
    def _check_unused_imports(self, tree):
        imports = {node.names[0].name: node.lineno for node in ast.walk(tree) if isinstance(node, ast.Import)}
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
        unused = [imp for imp in imports if imp not in used]
        for imp in unused:
            self.recommendations.append({
                "rule": "Unused Imports",
                "message": f"Unused import '{imp}' at line {imports[imp]}",
                "line": imports[imp],
            })
            # print(f"Unused import '{imp}' at line {imports[imp]}")


    def _check_sql_injection(self, tree):
        # Check for potential SQL injection vulnerabilities
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ["execute", "executemany"]:
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            self.recommendations.append({
                                "rule": "SQL Injection",
                                "message": f"Potential SQL injection vulnerability detected at line {node.lineno}",
                                "line": node.lineno,
                            })
                            # print(f"Potential SQL injection vulnerability detected at line {node.lineno}")


    def _check_command_injection(self, tree):
        # Check for potential command injection vulnerabilities
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["system", "popen", "call", "check_call", "check_output"]:
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            self.recommendations.append({
                                "rule": "Command Injection",
                                "message": f"Potential command injection vulnerability detected at line {node.lineno}",
                                "line": node.lineno,
                            })
                            # print(f"Potential command injection vulnerability detected at line {node.lineno}")


    def _check_xss(self, tree):
        # Check for potential cross-site scripting (XSS) vulnerabilities
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ["write", "send", "sendall"]:
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            self.recommendations.append({
                                "rule": "Cross-Site Scripting",
                                "message": f"Potential XSS vulnerability detected at line {node.lineno}",
                                "line": node.lineno,
                            })
                            # print(f"Potential XSS vulnerability detected at line {node.lineno}")


    def _check_weak_cryptographic_practices(self, tree):
        # Check for weak cryptographic practices
        weak_crypto_funcs = ["md5", "sha1"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in weak_crypto_funcs:
                    self.recommendations.append({
                        "rule": "Weak Cryptographic Practices",
                        "message": f"Weak cryptographic practice detected at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Weak cryptographic practice detected at line {node.lineno}")


    def _check_insecure_file_handling(self, tree):
        # Check for insecure file handling practices
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["open", "os.remove", "os.rename"]:
                    if any(isinstance(arg, ast.Str) and ".." in arg.s for arg in node.args):
                        self.recommendations.append({
                            "rule": "Insecure File Handling",
                            "message": f"Insecure file handling practice detected at line {node.lineno}",
                            "line": node.lineno,
                        })
                    # print(f"Insecure file handling practice detected at line {node.lineno}")


    def _check_session_security(self, tree):
        # Check for session security issues
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
                if node.targets[0].attr in ["session"]:
                    self.recommendations.append({
                        "rule": "Session Security",
                        "message": f"Potential session security issue detected at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Potential session security issue detected at line {node.lineno}")


    def _check_unvalidated_redirects(self, tree):
        # Check for unvalidated redirects
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["redirect", "url_for"]:
                    self.recommendations.append({
                        "rule": "Unvalidated Redirects",
                        "message": f"Potential unvalidated redirect detected at line {node.lineno}",
                        "line": node.lineno,
                    })
                    # print(f"Potential unvalidated redirect detected at line {node.lineno}")

    def _check_package_naming_convention(self, code_lines):
        # Check for package naming convention (lower-case)
        package_pattern = re.compile(r"package\s+([\w\.]+);")
        for i, line in enumerate(code_lines):
            match = package_pattern.match(line.strip())
            if match:
                package_name = match.group(1)
                if not package_name.islower():
                    self.recommendations.append({
                        "rule": "Package Naming Convention",
                        "message": f"Package name '{package_name}' at line {i+1} does not follow lower-case naming convention.",
                        "line": i + 1,
                    })

    def _check_class_naming_convention(self, tree):
        # Check for class naming convention (CamelCase)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self.recommendations.append({
                        "rule": "Class Naming Convention",
                        "message": f"Class name '{node.name}' at line {node.lineno} does not follow CamelCase convention.",
                        "line": node.lineno,
                    })

    def _check_method_naming_convention(self, tree):
        # Check for method naming convention (camelCase)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r"^[a-z][a-zA-Z0-9]*$", node.name):
                    self.recommendations.append({
                        "rule": "Method Naming Convention",
                        "message": f"Method name '{node.name}' at line {node.lineno} does not follow camelCase convention.",
                        "line": node.lineno,
                    })

    def _check_thread_safety_violations(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and "lock" not in [n.id for n in ast.walk(node)]:
                if isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
                    self.recommendations.append({
                        "rule": "Thread Safety Violations",
                        "message": f"Possible unsynchronized access to shared resource '{node.attr}' at line {node.lineno}. Consider using locks or synchronization.",
                        "line": node.lineno,
                    })

    def _check_inefficient_loops(self, tree):
        def is_nested_loop(node):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.For, ast.While)):
                        return True
            return False

        for node in ast.walk(tree):
            if is_nested_loop(node):
                self.recommendations.append({
                    "rule": "Inefficient Loops",
                    "message": f"Nested loop detected at line {node.lineno}. Consider optimizing the loop structure.",
                    "line": node.lineno,
                })

    def _check_potential_null_pointers(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id not in self.defined_variables:
                        self.recommendations.append({
                            "rule": "Potential NullPointerExceptions",
                            "message": f"Variable '{node.value.id}' at line {node.lineno} might be null or undefined. Add proper null checks.",
                            "line": node.lineno,
                        })

    def _check_resource_leaks(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if any(res in ast.dump(node.func) for res in ["open", "socket", "connect", "accept", "bind"]):
                    is_closed = any(
                        isinstance(child, ast.Call) and "close" in ast.dump(child.func)
                        for child in ast.iter_child_nodes(node)
                    )
                    if not is_closed:
                        self.recommendations.append({
                            "rule": "Resource Leaks",
                            "message": f"Resource opened at line {node.lineno} is not properly closed. Ensure to close resources like files or sockets.",
                            "line": node.lineno,
                        })


    def _check_empty_catch_blocks(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if not handler.body:
                        self.recommendations.append({
                            "rule": "Empty Catch Blocks",
                            "message": f"Empty catch block at line {handler.lineno}. Avoid silently swallowing exceptions.",
                            "line": handler.lineno,
                        })





# Example usage
# if __name__ == '__main__':
#     with open('path_to_your_file.py', 'r') as file:
#         code = file.read()  # Read the Python code to be analyzed
#
#     analyzer = CodeAnalyzer(code)
#     analyzer.run_pylint()  # Run Pylint for general coding standards
#     analyzer.custom_rules_check()  # Run custom rules check (e.g., function length)
