import json
import os
import tempfile
import pylint.lint
import ast
import re


class PythonCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # Python code to analyze
        self.recommendations = []  # List to store findings

    def generate_recommendations(self):
        # Parse the code into an AST
        try:
            tree = ast.parse(self.code)
        except SyntaxError as e:
            self.recommendations.append({"error": f"Syntax Error: {e}"})
            return self.recommendations

        # Run pylint and custom rules
        self.run_pylint()
        self.custom_rules_check(tree)
        return self._deduplicate_recommendations()

    def run_pylint(self):
        """Run pylint analysis and collect results."""
        pylint_opts = ["--disable=all", "--enable=E,W,C,R", "--output-format=json"]
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
                temp_file.write(self.code.encode())
                temp_file.close()
                pylint_opts.append(temp_file.name)
                results = pylint.lint.Run(pylint_opts, exit=False).linter.reporter.messages
                for issue in results:
                    self.recommendations.append(
                        {
                            "rule": "Pylint",
                            "message": issue.msg,
                            "line": issue.line,
                        }
                    )
        except Exception as e:
            self.recommendations.append({"error": f"Pylint error: {e}"})
        finally:
            os.unlink(temp_file.name)

    def custom_rules_check(self, tree):
        """Run custom static analysis checks."""
        checks = [
            self._check_function_length,
            self._check_too_many_parameters,
            self._check_nesting_depth,
            self._check_unused_variables,
            self._check_lack_of_comments,
            self._check_hard_coded_secrets,
            self._check_sql_injection,
            self._check_insecure_file_handling,
            self._check_weak_cryptography,
            self._check_empty_exception_handling,
            self._check_magic_numbers,
            self._check_long_chained_calls,
            self._check_deprecated_libraries,
            self._check_excessive_global_variables,
            self._check_resource_leaks,
            self._check_circular_imports,
            self._check_exception_messages,
            self._check_unreachable_code,
            self._check_variable_shadowing,
            self._check_naming_conventions,
            self._check_mutable_defaults,
            self._check_unencrypted_communication,
            self._check_dangerous_functions,
            self._check_dead_code,
            self._check_duplicate_code,
            self._check_excessive_classes_or_functions,
            self._check_dependency_inversion,
            self._check_inefficient_complexity,
            self._check_yagni_violation,
            self._check_hardcoded_file_paths,


        ]
        for check in checks:
            try:
                check(tree)
            except Exception as e:
                self.recommendations.append({"error": f"Error in {check.__name__}: {e}"})

    def _deduplicate_recommendations(self):
        """Remove duplicate recommendations."""
        unique_recommendations = []
        seen = set()
        for rec in self.recommendations:
            key = (rec.get("rule"), rec.get("line"), rec.get("message"))
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        return unique_recommendations

    def _check_function_length(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 10:
                    self.recommendations.append(
                        {
                            "rule": "Function Length",
                            "message": f"Function {node.name} is too long.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_too_many_parameters(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    self.recommendations.append(
                        {
                            "rule": "Too Many Parameters",
                            "message": f"Function {node.name} has too many parameters.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_nesting_depth(self, tree):
        """Detect excessive nesting."""
        max_depth = 4

        def calculate_depth(node, current_depth=0):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                current_depth += 1
            return max(
                [current_depth] + [calculate_depth(child, current_depth) for child in ast.iter_child_nodes(node)]
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                if calculate_depth(node) > max_depth:
                    self.recommendations.append(
                        {
                            "rule": "Excessive Nesting",
                            "message": f"Nesting exceeds {max_depth} levels.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_unused_variables(self, tree):
        """Check for unused variables."""
        declared = set()
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        declared.add(target.id)
            elif isinstance(node, ast.Name):
                used.add(node.id)

        unused = declared - used
        if unused:
            self.recommendations.append(
                {
                    "rule": "Unused Variables",
                    "message": f"Unused variables detected: {', '.join(unused)}.",
                }
            )

    def _check_lack_of_comments(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Str)
                ):
                    self.recommendations.append(
                        {
                            "rule": "Lack of Comments",
                            "message": f"Function {node.name} lacks a docstring.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_hard_coded_secrets(self, tree):
        secret_patterns = [r'(?i)(password|api_key|token)\s*=\s*[\'"].*[\'"]']
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Str):
                for pattern in secret_patterns:
                    if re.search(pattern, node.value.s):
                        self.recommendations.append(
                            {
                                "rule": "Hardcoded Secrets",
                                "message": f"Hardcoded secret detected: {node.value.s}",
                                "line": getattr(node, "lineno", "unknown"),
                            }
                        )

    def _check_sql_injection(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"execute", "executemany"}:
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            self.recommendations.append(
                                {
                                    "rule": "SQL Injection",
                                    "message": "Potential SQL injection vulnerability in dynamic SQL query.",
                                    "line": getattr(node, "lineno", "unknown"),
                                }
                            )

    def _check_insecure_file_handling(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                if not any(kw.arg == "encoding" for kw in getattr(node, "keywords", [])):
                    self.recommendations.append(
                        {
                            "rule": "Insecure File Handling",
                            "message": "File opened without specifying encoding.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_weak_cryptography(self, tree):
        weak_algorithms = {"md5", "sha1"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in weak_algorithms:
                    self.recommendations.append(
                        {
                            "rule": "Weak Cryptography",
                            "message": f"Usage of weak cryptographic algorithm: {node.func.attr}.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_empty_exception_handling(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None or (hasattr(handler.type, "id") and handler.type.id == "Exception"):
                        if not handler.body:
                            self.recommendations.append(
                                {
                                    "rule": "Empty Exception Handling",
                                    "message": "Empty or overly broad exception handling detected.",
                                    "line": getattr(node, "lineno", "unknown"),
                                }
                            )

    def _check_magic_numbers(self, tree, allowed_constants={0, 1, -1}):
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in allowed_constants:
                    self.recommendations.append(
                        {
                            "rule": "Magic Numbers",
                            "message": f"Magic number {node.value} found. Consider defining it as a constant.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_long_chained_calls(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain_length = 0
                current = node
                while isinstance(current, ast.Attribute):
                    chain_length += 1
                    current = current.value
                if chain_length > 3:
                    self.recommendations.append(
                        {
                            "rule": "Long Chained Method Calls",
                            "message": f"Long chained method call detected with length {chain_length}.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_deprecated_libraries(self, tree):
        deprecated_libraries = {"optparse": "Use argparse instead", "imp": "Use importlib instead"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in deprecated_libraries:
                        self.recommendations.append(
                            {
                                "rule": "Deprecated Libraries",
                                "message": f"Library '{alias.name}' is deprecated. {deprecated_libraries[alias.name]}",
                                "line": getattr(node, "lineno", "unknown"),
                            }
                        )

    def _check_excessive_global_variables(self, tree):
        global_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Global))
        if global_count > 5:
            self.recommendations.append(
                {
                    "rule": "Excessive Global Variables",
                    "message": f"Too many global variables ({global_count}) used in the code.",
                }
            )

    def _check_resource_leaks(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
                is_closed = any(
                    isinstance(child, ast.Call) and getattr(child.func, "attr", "") == "close"
                    for child in ast.iter_child_nodes(node)
                )
                if not is_closed:
                    self.recommendations.append(
                        {
                            "rule": "Resource Leaks",
                            "message": "File opened but not closed properly.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_circular_imports(self, tree):
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        if len(imports) != len(set(imports)):
            self.recommendations.append(
                {
                    "rule": "Circular Imports",
                    "message": "Circular import detected in the code.",
                }
            )

    def _check_exception_messages(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                if isinstance(node.exc, ast.Call) and not isinstance(node.exc.args[0], ast.Str):
                    self.recommendations.append(
                        {
                            "rule": "Improper Exception Messages",
                            "message": "Exceptions should include meaningful messages.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_unreachable_code(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Break)):
                        unreachable = node.body[i + 1:]
                        if unreachable:
                            self.recommendations.append(
                                {
                                    "rule": "Unreachable Code",
                                    "message": f"Unreachable code detected after line {stmt.lineno}.",
                                }
                            )

    def _check_variable_shadowing(self, tree):
        builtins = dir(__builtins__)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in builtins:
                self.recommendations.append(
                    {
                        "rule": "Variable Shadowing",
                        "message": f"Variable '{node.id}' shadows a built-in name.",
                        "line": getattr(node, "lineno", "unknown"),
                    }
                )

    def _check_naming_conventions(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Name)):
                if not re.match(r"^[a-z_][a-z0-9_]*$", node.name):
                    self.recommendations.append(
                        {
                            "rule": "Non-Pythonic Naming",
                            "message": f"'{node.name}' does not follow Python's snake_case naming convention.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_mutable_defaults(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.defaults:
                    if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                        self.recommendations.append(
                            {
                                "rule": "Mutable Default Argument",
                                "message": f"Mutable default argument detected in function '{node.name}'",
                                "line": getattr(node, "lineno", "unknown"),
                            }
                        )


    def _check_unencrypted_communication(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.attr in {"get", "post", "put", "delete"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Str) and arg.s.startswith("http://"):
                            self.recommendations.append(
                                {
                                    "rule": "Unencrypted Communication",
                                    "message": f"Unencrypted HTTP communication detected: {arg.s}",
                                    "line": getattr(node, "lineno", "unknown"),
                                }
                            )

    def _check_dangerous_functions(self, tree):
        dangerous_funcs = {"eval", "exec"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_funcs:
                    self.recommendations.append(
                        {
                            "rule": "Dangerous Function Usage",
                            "message": f"Usage of dangerous function '{node.func.id}' detected.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_excessive_classes_or_functions(self, tree):
        class_count = 0
        function_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, ast.FunctionDef):
                function_count += 1

        if class_count > 10:  # Example threshold
            self.recommendations.append(
                {
                    "rule": "Too Many Classes",
                    "message": f"File contains {class_count} classes, which might indicate excessive responsibility.",
                    "line": "unknown",
                }
            )
        if function_count > 20:  # Example threshold
            self.recommendations.append(
                {
                    "rule": "Too Many Functions",
                    "message": f"File contains {function_count} functions, which might indicate excessive responsibility.",
                    "line": "unknown",
                }
            )


    def _check_duplicate_code(self, tree):
        def get_code_snippet(node):
            return ast.unparse(node)

        snippets = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                snippet = get_code_snippet(node)
                if snippet in snippets:
                    self.recommendations.append(
                        {
                            "rule": "Duplicate Code",
                            "message": f"Duplicate code detected: {node.name} matches another block.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )
                else:
                    snippets[snippet] = node



    def _check_dead_code(self, tree):
        defined_functions = set()
        called_functions = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)

        unused_functions = defined_functions - called_functions
        for func in unused_functions:
            self.recommendations.append(
                {
                    "rule": "Dead Code",
                    "message": f"Function '{func}' is defined but never called.",
                    "line": "unknown",
                }
            )


    def _check_dependency_inversion(self, tree):
        """Check for direct dependency violations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.islower():
                self.recommendations.append(
                    {
                        "rule": "Dependency Inversion Violation",
                        "message": f"Direct instantiation detected: {node.func.id}. Consider using abstractions or dependency injection.",
                        "line": getattr(node, "lineno", "unknown"),
                    }
                )


    def _check_inefficient_complexity(self, tree):
        """Detect nested loops with potential performance issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                nested_loops = sum(1 for child in ast.walk(node) if isinstance(child, (ast.For, ast.While)))
                if nested_loops > 2:  # Example threshold
                    self.recommendations.append(
                        {
                            "rule": "Inefficient Algorithmic Complexity",
                            "message": f"Nested loops detected with depth {nested_loops}. Consider optimizing.",
                            "line": getattr(node, "lineno", "unknown"),
                        }
                    )

    def _check_yagni_violation(self, tree):
        """Check for unused functions or imports."""
        defined_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        called_functions = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

        unused_functions = defined_functions - called_functions
        for func in unused_functions:
            self.recommendations.append(
                {
                    "rule": "YAGNI Violation",
                    "message": f"Function '{func}' is defined but never used.",
                    "line": "unknown",
                }
            )


    def _check_hardcoded_file_paths(self, tree):
        """Detect hardcoded file paths in the code."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "/" in node.value or "\\" in node.value:
                    self.recommendations.append(
                        {
                            "rule": "Hardcoded File Path",
                            "message": f"Hardcoded file path detected: '{node.value}'. Use dynamic path generation instead.",
                            "line": getattr(node, 'lineno', 'unknown'),
                        }
                    )




# Example Usage
if __name__ == "__main__":
    code = """your Python code here"""
    analyzer = PythonCodeAnalyzer(code)
    recommendations = analyzer.generate_recommendations()
    print(json.dumps(recommendations, indent=2))
