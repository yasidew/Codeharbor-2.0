import javalang
import re

class JavaCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # Java code to be analyzed
        self.recommendations = []  # Recommendations based on the analysis

    def generate_recommendations(self):
        self.custom_rules_check()
        return self.recommendations

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



    def _check_method_length(self, tree):
        # Check if any method in the code has more than 10 lines
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if len(node.body) > 10:
                self.recommendations.append({
                    "rule": "Method Length",
                    "message": f"Method {node.name} is too long at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Method {node.name} is too long at line {node.position.line}")

    def _check_too_many_parameters(self, tree):
        # Check if any method has more than 5 parameters
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if len(node.parameters) > 5:
                self.recommendations.append({
                    "rule": "Too Many Parameters",
                    "message": f"Method {node.name} has too many parameters at line {node.position.line}",
                    "line": node.position.line,
                })

                # print(f"Method {node.name} has too many parameters at line {node.position.line}")

    def _check_cyclomatic_complexity(self, tree):
        # Custom rule to check cyclomatic complexity of methods
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            complexity = sum(1 for n in node.body if isinstance(n, (javalang.tree.IfStatement, javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)))
            if complexity > 10:
                self.recommendations.append({
                    "rule": "High Cyclomatic Complexity",
                    "message": f"Method {node.name} has high cyclomatic complexity at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Method {node.name} at line {node.position.line} has high cyclomatic complexity: {complexity}")

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
            self.recommendations.append({
                "rule": "Unused Variables",
                "message": f"Unused variables: {unused}",
                "line": None,
            })
            # print(f"Unused variables: {unused}")

    def _check_nesting_depth(self, node, depth=0, max_depth=4):
        # Check for excessive nesting in the code
        if depth > max_depth:
            self.recommendations.append({
                "rule": "Excessive Nesting",
                "message": f"Excessive nesting found at line {node.position.line}",
                "line": node.position.line,
            })
            # print(f"Excessive nesting found at line {node.position.line}")
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
                        self.recommendations.append({
                            "rule": "Hardcoded Secret",
                            "message": f"Hardcoded secret detected at line {node.position.line}: {node.initializer.value}",
                            "line": node.position.line,
                        })
                        # print(f"Hardcoded secret detected at line {node.position.line}: {node.initializer.value}")

    def _check_sensitive_data_in_logs(self, tree):
        # Check for sensitive data exposed in logs
        sensitive_keywords = ["password", "token", "api_key"]
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member == "println":
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and any(kw in arg.value.lower() for kw in sensitive_keywords):
                        self.recommendations.append({
                            "rule": "Sensitive Data in Logs",
                            "message": f"Sensitive data exposed in print statement at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })
                        # print(f"Sensitive data exposed in print statement at line {node.position.line}: {arg.value}")

    def _check_inefficient_data_structures(self, tree):
        # Check for inefficient data structures
        for path, node in tree.filter(javalang.tree.BinaryOperation):
            if node.operator == "in" and isinstance(node.right, javalang.tree.ArrayInitializer):
                self.recommendations.append({
                    "rule": "Inefficient Data Structure",
                    "message": f"Inefficient membership check using an array at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Inefficient membership check using an array at line {node.position.line}")

    def _check_long_chained_calls(self, tree):
        # Check for long chained method calls
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            chain_length = 0
            current = node
            while isinstance(current, javalang.tree.MethodInvocation):
                chain_length += 1
                current = current.qualifier
            if chain_length > 3:
                self.recommendations.append({
                    "rule": "Long Chained Method Calls",
                    "message": f"Long chained method call of length {chain_length} at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Long chained method call of length {chain_length} at line {node.position.line}")

    def _check_global_variable_overuse(self, tree):
        # Check for excessive use of global variables
        global_count = sum(1 for path, node in tree.filter(javalang.tree.FieldDeclaration))
        if global_count > 5:
            self.recommendations.append({
                "rule": "Excessive Global Variables",
                "message": f"Excessive use of global variables: {global_count}",
                "line": None,
            })
            # print(f"Excessive use of global variables: {global_count}")

    def _check_deprecated_libraries(self, tree):
        # Check for deprecated libraries
        deprecated_libs = ["java.util.Random", "java.lang.System"]  # Example: add more based on research
        for path, node in tree.filter(javalang.tree.Import):
            if node.path in deprecated_libs:
                self.recommendations.append({
                    "rule": "Deprecated Library",
                    "message": f"Deprecated library {node.path} imported at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Deprecated library {node.path} imported at line {node.position.line}")

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

    def _check_lack_of_comments(self, tree):
        # Check for lack of comments/docstrings (no proper documentation)
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not node.documentation:
                self.recommendations.append({
                    "rule": "Lack of Comments",
                    "message": f"Method '{node.name}' at line {node.position.line} lacks a docstring.",
                    "line": node.position.line,
                })
                # print(f"Method '{node.name}' at line {node.position.line} lacks a docstring.")

    def _check_magic_numbers(self, tree, allowed=[0, 1, -1]):
        # Check for magic numbers in the code (means hard-coded numbers)
        for path, node in tree.filter(javalang.tree.Literal):
            if isinstance(node.value, (int, float)) and node.value not in allowed:
                self.recommendations.append({
                    "rule": "Magic Numbers",
                    "message": f"Magic number {node.value} found at line {node.position.line}. Consider defining it as a constant.",
                    "line": node.position.line,
                })
                # print(f"Magic number {node.value} found at line {node.position.line}. Consider defining it as a constant.")

    def _check_exception_handling(self, tree):
        # Check for unused imports in the code
        for path, node in tree.filter(javalang.tree.TryStatement):
            if not node.catches:
                self.recommendations.append({
                    "rule": "Exception Handling",
                    "message": f"Try block at line {node.position.line} lacks exception handling.",
                    "line": node.position.line,
                })
                # print(f"Try block at line {node.position.line} lacks exception handling.")
            for handler in node.catches:
                if handler.parameter.name == "Exception":
                    self.recommendations.append({
                        "rule": "Overly Broad Exception Handling",
                        "message": f"Overly broad exception handling found at line {handler.position.line}",
                        "line": handler.position.line,
                    })
                    # print(f"Overly broad exception handling found at line {handler.position.line}")

    def _check_recursive_methods(self, tree):
        # Check for recursive methods without a proper base case to avoid infinite recursion
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            for sub_node in node.body:
                if isinstance(sub_node, javalang.tree.MethodInvocation) and sub_node.member == node.name:
                    self.recommendations.append({
                        "rule": "Recursive Method",
                        "message": f"Recursive method '{node.name}' detected at line {node.position.line}",
                        "line": node.position.line,
                    })
                    # print(f"Recursive method '{node.name}' detected at line {node.position.line}")

    def _check_unreachable_code(self, tree):
        # Check for unreachable code after return, raise, or break statements
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            for i, stmt in enumerate(node.body[:-1]):
                if isinstance(stmt, (javalang.tree.ReturnStatement, javalang.tree.ThrowStatement, javalang.tree.BreakStatement)):
                    unreachable = node.body[i+1:]
                    if unreachable:
                        self.recommendations.append({
                            "rule": "Unreachable Code",
                            "message": f"Unreachable code detected after line {stmt.position.line}",
                            "line": stmt.position.line,
                        })
                        # print(f"Unreachable code detected after line {stmt.position.line}")

    def _check_missing_default_case(self, tree):
        # Check for missing default case in switch-statements
        for path, node in tree.filter(javalang.tree.SwitchStatement):
            has_default = any(isinstance(n, javalang.tree.SwitchStatementCase) and n.case is None for n in node.cases)
            if not has_default:
                self.recommendations.append({
                    "rule": "Missing Default Case",
                    "message": f"Missing default case for switch-statement at line {node.position.line}",
                    "line": node.position.line,
                })
                # print(f"Missing default case for switch-statement at line {node.position.line}")

    def _check_circular_imports(self, tree):
        # Check for circular imports in the code
        imports = []
        for path, node in tree.filter(javalang.tree.Import):
            imports.append(node.path)
        if len(imports) != len(set(imports)):
            self.recommendations.append({
                "rule": "Circular Imports",
                "message": "Circular import detected.",
                "line": None,
            })
            # print("Circular import detected.")

    def _check_deprecated_syntax(self, tree):
        # Custom rule to check for deprecated syntax
        deprecated_nodes = (javalang.tree.PrintStatement, javalang.tree.ExecStatement)  # Example: old Java syntax
        for path, node in tree.filter(deprecated_nodes):
            self.recommendations.append({
                "rule": "Deprecated Syntax",
                "message": f"Deprecated syntax {type(node).__name__} found at line {node.position.line}",
                "line": node.position.line,
            })
            # print(f"Deprecated syntax {type(node).__name__} found at line {node.position.line}")

    def _check_unused_imports(self, tree):
        # Check for unused imports in the code
        imports = {node.path: node.position.line for path, node in tree.filter(javalang.tree.Import)}
        used = set()
        for path, node in tree.filter(javalang.tree.MemberReference):
            used.add(node.member)
        unused = [imp for imp in imports if imp not in used]
        for imp in unused:
            self.recommendations.append({
                "rule": "Unused Import",
                "message": f"Unused import '{imp}' at line {imports[imp]}",
                "line": imports[imp],
            })
            # print(f"Unused import '{imp}' at line {imports[imp]}")

    def _check_sql_injection(self, tree):
        # Check for potential SQL injection vulnerabilities
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() in ["executequery", "executeupdate", "execute", "preparestatement", "executemany", ]:
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and ("SELECT" in arg.value or "INSERT" in arg.value):
                        self.recommendations.append({
                            "rule": "SQL Injection",
                            "message": f"Potential SQL injection vulnerability at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })

    def _check_command_injection(self, tree):
        # Check for command injection vulnerabilities
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() in ["exec", "execcommand", "system", "popen", "call", "check_call", "check_output"]:
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal):
                        self.recommendations.append({
                            "rule": "Command Injection",
                            "message": f"Potential command injection vulnerability at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })

    def _check_xss(self, tree):
        # Check for Cross-Site Scripting (XSS) vulnerabilities
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() in ["write", "print", "println", "send", "sendall", "sendto", "broadcast"]:
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and ("<script>" in arg.value or "</script>" in arg.value):
                        self.recommendations.append({
                            "rule": "Cross-Site Scripting (XSS)",
                            "message": f"Potential XSS vulnerability at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })

    def _check_weak_cryptographic_practices(self, tree):
        # Check for usage of weak cryptographic algorithms
        weak_algorithms = ["MD5", "SHA1"]
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() == "getinstance":
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and arg.value.strip('"') in weak_algorithms:
                        self.recommendations.append({
                            "rule": "Weak Cryptographic Practices",
                            "message": f"Weak cryptographic algorithm used at line {node.position.line}: {arg.value.strip('\"')}",
                            "line": node.position.line,
                        })

    def _check_insecure_file_handling(self, tree):
        # Check for insecure file handling practices
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() in ["delete", "rename", "write", "open", "read", "copy", "move", "create"]:
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and not arg.value.startswith("secure_"):
                        self.recommendations.append({
                            "rule": "Insecure File Handling",
                            "message": f"Insecure file operation detected at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })

    def _check_session_security(self, tree):
        # Check for insecure session management
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() == "setmaxinactiveinterval":
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and int(arg.value) > 3600:
                        self.recommendations.append({
                            "rule": "Session Security",
                            "message": f"Insecure session timeout setting at line {node.position.line}: {arg.value} seconds",
                            "line": node.position.line,
                        })

    def _check_unvalidated_redirects(self, tree):
        # Check for unvalidated redirects
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member.lower() in ["sendredirect", "redirect", "url_for", "forward"]:
                for arg in node.arguments:
                    if isinstance(arg, javalang.tree.Literal) and not arg.value.startswith("https://trusted-domain.com"):
                        self.recommendations.append({
                            "rule": "Unvalidated Redirects",
                            "message": f"Unvalidated redirect detected at line {node.position.line}: {arg.value}",
                            "line": node.position.line,
                        })


    def _check_package_naming_convention(self, lines):
        # Check for package naming conventions
        for line in lines:
            if line.strip().startswith("package"):
                package_name = line.split("package")[1].strip("; ").strip()
                if not re.match(r"^[a-z]+(\.[a-z][a-z0-9]*)*$", package_name):
                    self.recommendations.append({
                        "rule": "Package Naming Convention",
                        "message": f"Package name '{package_name}' does not follow naming conventions.",
                        "line": lines.index(line) + 1,
                    })

    def _check_class_naming_convention(self, tree):
        # Check if class names follow PascalCase
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                self.recommendations.append({
                    "rule": "Class Naming Convention",
                    "message": f"Class name '{node.name}' does not follow PascalCase naming conventions.",
                    "line": node.position.line,
                })

    def _check_method_naming_convention(self, tree):
        # Check if method names follow camelCase
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not re.match(r"^[a-z][a-zA-Z0-9]*$", node.name):
                self.recommendations.append({
                    "rule": "Method Naming Convention",
                    "message": f"Method name '{node.name}' does not follow camelCase naming conventions.",
                    "line": node.position.line,
                })

    def _check_thread_safety_violations(self, tree):
        # Check for thread-safety issues (e.g., unsynchronized shared variables)
        for path, node in tree.filter(javalang.tree.FieldDeclaration):
            if 'static' in node.modifiers and 'volatile' not in node.modifiers:
                self.recommendations.append({
                    "rule": "Thread Safety Violation",
                    "message": f"Static field {node.declarators[0].name} may not be thread-safe. Consider using 'volatile' or synchronization.",
                    "line": node.position.line,
                })

    def _check_inefficient_loops(self, tree):
        # Check for nested loops causing performance bottlenecks
        for path, node in tree.filter(javalang.tree.ForStatement):
            if any(isinstance(child, javalang.tree.ForStatement) for child in node.body):
                self.recommendations.append({
                    "rule": "Inefficient Loops",
                    "message": f"Nested loop detected at line {node.position.line}. Consider optimizing.",
                    "line": node.position.line,
                })

    def _check_potential_null_pointers(self, tree):
        # Check for potential null pointer exceptions
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if isinstance(node.qualifier, javalang.tree.MemberReference) and not node.qualifier.position:
                self.recommendations.append({
                    "rule": "Potential NullPointerException",
                    "message": f"Possible null dereference for {node.qualifier.member} at line {node.position.line}. Ensure proper null checks.",
                    "line": node.position.line,
                })

    def _check_resource_leaks(self, tree):
        # Check for unclosed resources like files, sockets, etc.
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in {"open", "getConnection"}:
                if not any("close" in arg.member for arg in node.arguments if isinstance(arg, javalang.tree.MethodInvocation)):
                    self.recommendations.append({
                        "rule": "Resource Leak",
                        "message": f"Resource opened using {node.member} at line {node.position.line} is not properly closed.",
                        "line": node.position.line,
                    })

    def _check_empty_catch_blocks(self, tree):
        # Check for empty catch blocks
        for path, node in tree.filter(javalang.tree.TryStatement):
            for catch_clause in node.catches:
                if not catch_clause.block or len(catch_clause.block) == 0:
                    self.recommendations.append({
                        "rule": "Empty Catch Block",
                        "message": f"Empty catch block detected at line {catch_clause.block.position.line}. Handle exceptions properly.",
                        "line": catch_clause.block.position.line,
                    })




# # Example usage
# if __name__ == '__main__':
#     with open('path_to_your_java_file.java', 'r') as file:
#         code = file.read()  # Read the Java code to be analyzed
#
#     analyzer = JavaCodeAnalyzer(code)
#     analyzer.custom_rules_check()  # Run custom rules check (e.g., method length)