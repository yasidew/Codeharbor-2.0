# import json
# import re
#
#
# class JavaScriptCodeAnalyzer:
#     def __init__(self, code):
#         self.code = code  # JavaScript code to analyze
#         self.recommendations = []  # List to store findings
#
#     # def generate_recommendations(self):
#     #     try:
#     #         import esprima
#     #         tree = esprima.parseScript(self.code, tolerant=True, comment=True)
#     #         self.custom_rules_check(tree)
#     #     except Exception as e:
#     #         self.recommendations.append({"error": f"Syntax Error: {e}"})
#     #         return self.recommendations
#
#     def generate_recommendations(self):
#         """Run the analysis and generate recommendations."""
#         try:
#             import esprima
#             if not self.code.strip():
#                 self.recommendations.append({"error": "No code provided for analysis."})
#                 return self.recommendations
#
#             # Parse the JavaScript code into an AST
#             tree = esprima.parseScript(self.code, tolerant=True, comment=True)
#
#             # Apply custom rules
#             self.custom_rules_check(tree)
#         except Exception as e:
#             self.recommendations.append({"error": f"Syntax Error: {e}"})
#
#         # Deduplicate recommendations
#         self._deduplicate_recommendations()
#
#         return self.recommendations
#
#
#     def custom_rules_check(self, tree):
#         # Define the checks to run
#         checks = [
#             self._check_function_length,
#             self._check_too_many_parameters,
#             self._check_nesting_depth,
#             self._check_unused_variables,
#             self._check_lack_of_comments,
#             self._check_hard_coded_secrets,
#             self._check_sql_injection,
#             self._check_insecure_file_handling,
#             self._check_weak_cryptography,
#             self._check_empty_exception_handling,
#             self._check_magic_numbers,
#             self._check_long_chained_calls,
#             self._check_deprecated_libraries,
#             self._check_excessive_global_variables,
#             self._check_resource_leaks,
#             self._check_circular_imports,
#             self._check_exception_messages,
#             self._check_unreachable_code,
#             self._check_variable_shadowing,
#             self._check_naming_conventions,
#         ]
#
#         # Run each check
#         for check in checks:
#             try:
#                 check(tree)
#             except Exception as e:
#                 self.recommendations.append({"error": f"Error in {check.__name__}: {e}"})
#
#         return self._deduplicate_recommendations()
#
#     # def _deduplicate_recommendations(self):
#     #     """Remove duplicate recommendations."""
#     #     unique_recommendations = []
#     #     seen = set()
#     #     for rec in self.recommendations:
#     #         key = (rec.get("rule"), rec.get("line"), rec.get("message"))
#     #         if key not in seen:
#     #             seen.add(key)
#     #             unique_recommendations.append(rec)
#     #     return unique_recommendations
#
#     def _deduplicate_recommendations(self):
#         """Remove duplicate recommendations and group 'Lack of Comments' by functions or methods."""
#         unique = []
#         seen = set()
#         lack_of_comments = []
#
#         for rec in self.recommendations:
#             key = (rec.get("rule"), rec.get("line"), rec.get("message"))
#             if key not in seen:
#                 seen.add(key)
#                 if rec.get("rule") == "Lack of Comments":
#                     lack_of_comments.append(rec)
#                 else:
#                     unique.append(rec)
#
#         # Group all 'Lack of Comments' into one recommendation
#         if lack_of_comments:
#             unique.append({
#                 "rule": "Lack of Comments",
#                 "message": f"{len(lack_of_comments)} functions/methods lack comments. Please review.",
#                 "details": lack_of_comments  # Provide detailed breakdown if needed
#             })
#
#         self.recommendations = unique
#
#
#     def _check_function_length(self, tree):
#         """Check if functions are too long."""
#         for node in self._walk(tree):
#             if node.type == "FunctionDeclaration":
#                 body_length = len(node.body.body)
#                 if body_length > 20:  # Example threshold
#                     self.recommendations.append(
#                         {
#                             "rule": "Function Length",
#                             "message": f"Function '{node.id.name}' has {body_length} lines. Consider refactoring.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_too_many_parameters(self, tree):
#         """Check if functions have too many parameters."""
#         for node in self._walk(tree):
#             if node.type == "FunctionDeclaration":
#                 param_count = len(node.params)
#                 if param_count > 5:  # Example threshold
#                     self.recommendations.append(
#                         {
#                             "rule": "Too Many Parameters",
#                             "message": f"Function '{node.id.name}' has {param_count} parameters. Consider reducing them.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_nesting_depth(self, tree):
#         """Check if nesting depth is excessive."""
#         max_depth = 4
#
#         def calculate_depth(node, current_depth=0):
#             if node.type in ["IfStatement", "ForStatement", "WhileStatement"]:
#                 current_depth += 1
#             max_child_depth = max(
#                 (calculate_depth(child, current_depth) for child in getattr(node, "body", []) or []),
#                 default=0,
#             )
#             return max(current_depth, max_child_depth)
#
#         for node in self._walk(tree):
#             if node.type in ["IfStatement", "ForStatement", "WhileStatement"]:
#                 depth = calculate_depth(node)
#                 if depth > max_depth:
#                     self.recommendations.append(
#                         {
#                             "rule": "Excessive Nesting",
#                             "message": f"Nesting depth of {depth} exceeds the allowed limit of {max_depth}.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_unused_variables(self, tree):
#         """Check for unused variables."""
#         declared = set()
#         used = set()
#
#         for node in self._walk(tree):
#             if node.type == "VariableDeclarator":
#                 declared.add(node.id.name)
#             elif node.type == "Identifier":
#                 used.add(node.name)
#
#         unused = declared - used
#         for variable in unused:
#             self.recommendations.append(
#                 {
#                     "rule": "Unused Variables",
#                     "message": f"Variable '{variable}' is declared but not used.",
#                 }
#             )
#
#     def _check_lack_of_comments(self, tree):
#         """Check for lack of comments in functions."""
#         for node in self._walk(tree):
#             if node.type == "FunctionDeclaration":
#                 if not hasattr(node, "leadingComments") or not node.leadingComments:
#                     self.recommendations.append(
#                         {
#                             "rule": "Lack of Comments",
#                             "message": f"Function '{node.id.name}' lacks comments or documentation.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_hard_coded_secrets(self, tree):
#         """Check for hardcoded secrets like passwords or API keys."""
#         secret_patterns = [r'(?i)(password|apiKey|token)\s*[:=]\s*["\'].*["\']']
#         for node in self._walk(tree):
#             if node.type == "Literal" and isinstance(node.value, str):
#                 for pattern in secret_patterns:
#                     if re.search(pattern, node.value):
#                         self.recommendations.append(
#                             {
#                                 "rule": "Hardcoded Secrets",
#                                 "message": f"Hardcoded secret detected: '{node.value}'.",
#                                 "line": node.loc.start.line,
#                             }
#                         )
#
#
#     def _check_sql_injection(self, tree):
#         """Check for potential SQL injection vulnerabilities."""
#         for node in self._walk(tree):
#             if node.type == "CallExpression" and hasattr(node.callee, "property"):
#                 if node.callee.property.name in ["query", "execute"]:
#                     for arg in node.arguments:
#                         if arg.type == "BinaryExpression" and arg.operator == "+":
#                             self.recommendations.append(
#                                 {
#                                     "rule": "SQL Injection",
#                                     "message": "Dynamic query detected. Use parameterized queries to prevent SQL injection.",
#                                     "line": node.loc.start.line,
#                                 }
#                             )
#
#     def _check_insecure_file_handling(self, tree):
#         """Check for insecure file handling."""
#         for node in self._walk(tree):
#             if node.type == "CallExpression" and hasattr(node.callee, "name"):
#                 if node.callee.name in ["fs.open", "fs.readFile", "fs.writeFile"]:
#                     args = [arg.value for arg in node.arguments if arg.type == "Literal"]
#                     if "encoding" not in args:
#                         self.recommendations.append(
#                             {
#                                 "rule": "Insecure File Handling",
#                                 "message": "File operation detected without specifying encoding. Specify an encoding for safety.",
#                                 "line": node.loc.start.line,
#                             }
#                         )
#
#     def _check_weak_cryptography(self, tree):
#         """Check for usage of weak cryptographic algorithms."""
#         weak_algorithms = ["md5", "sha1"]
#         for node in self._walk(tree):
#             if node.type == "CallExpression" and hasattr(node.callee, "property"):
#                 if node.callee.property.name in weak_algorithms:
#                     self.recommendations.append(
#                         {
#                             "rule": "Weak Cryptography",
#                             "message": f"Usage of weak cryptographic algorithm '{node.callee.property.name}'. Use a stronger algorithm like SHA256.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_empty_exception_handling(self, tree):
#         """Check for empty exception handling."""
#         for node in self._walk(tree):
#             if node.type == "TryStatement":
#                 if not node.handler or not node.handler.body.body:
#                     self.recommendations.append(
#                         {
#                             "rule": "Empty Exception Handling",
#                             "message": "Empty or overly broad exception handling detected.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_magic_numbers(self, tree, allowed_constants={0, 1, -1}):
#         """Check for magic numbers."""
#         for node in self._walk(tree):
#             if node.type == "Literal" and isinstance(node.value, (int, float)):
#                 if node.value not in allowed_constants:
#                     self.recommendations.append(
#                         {
#                             "rule": "Magic Numbers",
#                             "message": f"Magic number {node.value} found. Consider defining it as a constant.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_long_chained_calls(self, tree):
#         """Check for long chained method calls."""
#         for node in self._walk(tree):
#             if node.type == "MemberExpression":
#                 chain_length = 0
#                 current = node
#                 while current.type == "MemberExpression":
#                     chain_length += 1
#                     current = current.object
#                 if chain_length > 3:  # Example threshold
#                     self.recommendations.append(
#                         {
#                             "rule": "Long Chained Method Calls",
#                             "message": f"Detected long chained method call with a length of {chain_length}. Consider breaking it into smaller calls.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_deprecated_libraries(self, tree):
#         """Check for usage of deprecated libraries."""
#         deprecated_libraries = {
#             "fs.promises": "Deprecated. Use async/await with fs directly.",
#             "crypto.createCipher": "Deprecated. Use crypto.createCipheriv instead.",
#         }
#         for node in self._walk(tree):
#             if node.type == "MemberExpression" and node.object.type == "Identifier":
#                 library = f"{node.object.name}.{node.property.name}"
#                 if library in deprecated_libraries:
#                     self.recommendations.append(
#                         {
#                             "rule": "Deprecated Libraries",
#                             "message": f"Usage of deprecated library '{library}'. {deprecated_libraries[library]}",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_excessive_global_variables(self, tree):
#         """Check for excessive use of global variables."""
#         global_vars = set()
#         for node in self._walk(tree):
#             if node.type == "VariableDeclaration" and node.kind == "var":  # 'var' declarations are global by default
#                 for decl in node.declarations:
#                     global_vars.add(decl.id.name)
#
#         if len(global_vars) > 5:  # Example threshold
#             self.recommendations.append(
#                 {
#                     "rule": "Excessive Global Variables",
#                     "message": f"Excessive use of global variables detected ({len(global_vars)}). Avoid using globals.",
#                 }
#             )
#
#     def _check_resource_leaks(self, tree):
#         """Check for resource leaks, such as open files or connections."""
#         for node in self._walk(tree):
#             if node.type == "CallExpression" and hasattr(node.callee, "name"):
#                 if node.callee.name in ["open", "readFile", "createReadStream"]:
#                     has_close = False
#                     for sibling in getattr(node, "parent", {}).get("body", []):
#                         if sibling.type == "CallExpression" and hasattr(sibling.callee, "property"):
#                             if sibling.callee.property.name == "close":
#                                 has_close = True
#                                 break
#                     if not has_close:
#                         self.recommendations.append(
#                             {
#                                 "rule": "Resource Leaks",
#                                 "message": "Resource (e.g., file or stream) opened without being properly closed.",
#                                 "line": node.loc.start.line,
#                             }
#                         )
#
#     def _check_circular_imports(self, tree):
#         """Check for circular imports."""
#         imports = set()
#         for node in self._walk(tree):
#             if node.type == "ImportDeclaration":
#                 imports.add(node.source.value)
#
#         # Simulate detecting circular imports (actual detection might require a dependency graph)
#         if len(imports) != len(set(imports)):
#             self.recommendations.append(
#                 {
#                     "rule": "Circular Imports",
#                     "message": "Circular import detected. Avoid circular dependencies between modules.",
#                 }
#             )
#
#     def _check_exception_messages(self, tree):
#         """Check if exceptions include meaningful messages."""
#         for node in self._walk(tree):
#             if node.type == "ThrowStatement":
#                 if (
#                         node.argument.type == "NewExpression"
#                         and hasattr(node.argument.callee, "name")
#                         and not node.argument.arguments
#                 ):
#                     self.recommendations.append(
#                         {
#                             "rule": "Improper Exception Messages",
#                             "message": f"Exception thrown without a meaningful message.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#     def _check_unreachable_code(self, tree):
#         """Check for unreachable code."""
#         for node in self._walk(tree):
#             if node.type == "BlockStatement":
#                 for i, stmt in enumerate(node.body[:-1]):
#                     if stmt.type in ["ReturnStatement", "ThrowStatement", "BreakStatement", "ContinueStatement"]:
#                         for unreachable in node.body[i + 1 :]:
#                             self.recommendations.append(
#                                 {
#                                     "rule": "Unreachable Code",
#                                     "message": f"Unreachable code detected after line {stmt.loc.start.line}.",
#                                     "line": unreachable.loc.start.line,
#                                 }
#                             )
#                         break
#
#     def _check_variable_shadowing(self, tree):
#         """Check for variable shadowing with built-in names."""
#         built_ins = {"eval", "arguments", "NaN", "undefined", "Infinity"}
#         for node in self._walk(tree):
#             if node.type == "VariableDeclarator" and node.id.name in built_ins:
#                 self.recommendations.append(
#                     {
#                         "rule": "Variable Shadowing",
#                         "message": f"Variable '{node.id.name}' shadows a built-in JavaScript name.",
#                         "line": node.loc.start.line,
#                     }
#                 )
#
#     def _check_naming_conventions(self, tree):
#         """Check if variables and functions follow camelCase naming conventions."""
#         def is_camel_case(name):
#             return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name))
#
#         for node in self._walk(tree):
#             if node.type in ["FunctionDeclaration", "VariableDeclarator"]:
#                 name = node.id.name if node.type == "VariableDeclarator" else node.id.name
#                 if not is_camel_case(name):
#                     self.recommendations.append(
#                         {
#                             "rule": "Non-JavaScript Naming",
#                             "message": f"'{name}' does not follow JavaScript's camelCase naming convention.",
#                             "line": node.loc.start.line,
#                         }
#                     )
#
#
#
#
#     def _walk(self, node):
#         """Helper method to recursively walk through the AST."""
#         yield node
#         for child in getattr(node, "body", []) or []:
#             yield from self._walk(child)
#
#
# # Example Usage
# if __name__ == "__main__":
#     js_code = """
#     function authenticateUser(username, password) {
#         if (username === "admin" && password === "secret") {
#             return true;
#         }
#         return false;
#     }
#
#     var unusedVar = "test";
#     """
#     analyzer = JavaScriptCodeAnalyzer(js_code)
#     recommendations = analyzer.generate_recommendations()
#     print(json.dumps(recommendations, indent=2))


import json
import re
import subprocess

class JavaScriptCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # JavaScript code to analyze
        self.recommendations = []  # List to store findings


    def generate_recommendations(self):
        """Run the analysis and generate recommendations."""
        try:
            import esprima
            if not self.code.strip():
                self.recommendations.append({"rule": "No Code", "message": "No code provided for analysis."})
                return self.recommendations

            # Parse the JavaScript code into an AST
            tree = esprima.parseScript(self.code, tolerant=True, comment=True)

            # Apply custom rules
            self.custom_rules_check(tree)
        except Exception as e:
            self.recommendations.append({"rule": "Syntax Error", "message": f"Syntax Error: {e}"})

        # Deduplicate recommendations
        self._deduplicate_recommendations()

        return self.recommendations

    # def generate_recommendations(self):
    #     """Run the analysis and generate recommendations."""
    #     try:
    #         if not self.code.strip():
    #             self.recommendations.append({"rule": "No Code", "message": "No code provided for analysis."})
    #             return self.recommendations
    #
    #         # Write the code to a temporary file
    #         with open("temp.js", "w") as temp_file:
    #             temp_file.write(self.code)
    #
    #         # Run ESLint on the temporary file
    #         result = subprocess.run(["eslint", "temp.js", "--format", "json"], capture_output=True, text=True)
    #         eslint_output = json.loads(result.stdout)
    #
    #         # Process ESLint output
    #         for message in eslint_output[0].get("messages", []):
    #             self.recommendations.append({
    #                 "rule": message["ruleId"],
    #                 "message": message["message"],
    #                 "line": message["line"],
    #             })
    #
    #     except Exception as e:
    #         self.recommendations.append({"rule": "Syntax Error", "message": f"Syntax Error: {e}"})
    #
    #     # Deduplicate recommendations
    #     self._deduplicate_recommendations()
    #
    #     return self.recommendations


    def custom_rules_check(self, tree):
        # Define the checks to run
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
        ]

        # Run each check
        for check in checks:
            try:
                check(tree)
            except Exception as e:
                self.recommendations.append({"rule": f"Error in {check.__name__}", "message": str(e)})

    def _deduplicate_recommendations(self):
        """Remove duplicate recommendations and group 'Lack of Comments' by functions or methods."""
        unique = []
        seen = set()
        lack_of_comments = []

        for rec in self.recommendations:
            key = (rec.get("rule"), rec.get("line"), rec.get("message"))
            if key not in seen:
                seen.add(key)
                if rec.get("rule") == "Lack of Comments":
                    lack_of_comments.append(rec)
                else:
                    unique.append(rec)

        # Group all 'Lack of Comments' into one recommendation
        if lack_of_comments:
            unique.append({
                "rule": "Lack of Comments",
                "message": f"{len(lack_of_comments)} functions/methods lack comments. Please review.",
                "details": lack_of_comments  # Provide detailed breakdown if needed
            })

        self.recommendations = unique

    # def _deduplicate_recommendations(self):
    #     """Remove duplicate recommendations and group 'Lack of Comments' by functions or methods."""
    #     unique = []
    #     seen = set()
    #     lack_of_comments = []
    #
    #     for rec in self.recommendations:
    #         key = (rec.get("rule"), rec.get("line"), rec.get("message"))
    #         if key not in seen:
    #             seen.add(key)
    #             if rec.get("rule") == "Lack of Comments":
    #                 lack_of_comments.append(rec)
    #             else:
    #                 unique.append(rec)
    #
    #     # Group all 'Lack of Comments' into one recommendation
    #     if lack_of_comments:
    #         unique.append({
    #             "rule": "Lack of Comments",
    #             "message": f"{len(lack_of_comments)} functions/methods lack comments. Please review.",
    #             "details": lack_of_comments  # Provide detailed breakdown if needed
    #         })
    #
    #     self.recommendations = unique

    def _check_function_length(self, tree):
        """Check if functions are too long."""
        for node in self._walk(tree):
            if node.type == "FunctionDeclaration":
                body_length = len(node.body.body)
                line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                if body_length > 20:  # Example threshold
                    self.recommendations.append(
                        {
                            "rule": "Function Length",
                            "message": f"Function '{node.id.name}' has {body_length} lines. Consider refactoring.",
                            "line": line,
                        }
                    )

    def _check_too_many_parameters(self, tree):
        """Check if functions have too many parameters."""
        for node in self._walk(tree):
            if node.type == "FunctionDeclaration":
                param_count = len(node.params)
                line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'

                if param_count > 5:  # Example threshold
                    self.recommendations.append(
                        {
                            "rule": "Too Many Parameters",
                            "message": f"Function '{node.id.name}' has {param_count} parameters. Consider reducing them.",
                            "line": line,
                        }
                    )

    # def _check_nesting_depth(self, tree):
    #     """Check if nesting depth is excessive."""
    #     max_depth = 4
    #
    #     def calculate_depth(node, current_depth=0):
    #         if node.type in ["IfStatement", "ForStatement", "WhileStatement"]:
    #             current_depth += 1
    #         max_child_depth = max(
    #             (calculate_depth(child, current_depth) for child in getattr(node, "body", []) or []),
    #             default=0,
    #         )
    #         return max(current_depth, max_child_depth)
    #
    #     for node in self._walk(tree):
    #         if node.type in ["IfStatement", "ForStatement", "WhileStatement"]:
    #             depth = calculate_depth(node)
    #             line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
    #
    #             if depth > max_depth:
    #                 self.recommendations.append(
    #                     {
    #                         "rule": "Excessive Nesting",
    #                         "message": f"Nesting depth of {depth} exceeds the allowed limit of {max_depth}.",
    #                         "line": line,
    #                     }
    #                 )

    def _check_nesting_depth(self, tree):
        """Check if nesting depth is excessive."""
        max_depth = 4

        def calculate_depth(node, current_depth=0):
            if not hasattr(node, 'body') or not isinstance(node.body, list):
                return current_depth
            return max(
                (calculate_depth(child, current_depth + 1) for child in node.body),
                default=current_depth
            )

        for node in self._walk(tree):
            if node.type in ["IfStatement", "ForStatement", "WhileStatement"]:
                depth = calculate_depth(node)
                line = getattr(node.loc, 'start', {}).get('line', 'unknown')
                if depth > max_depth:
                    self.recommendations.append(
                        {
                            "rule": "Excessive Nesting",
                            "message": f"Nesting depth of {depth} exceeds the allowed limit of {max_depth}.",
                            "line": line,
                        }
                    )


    def _check_unused_variables(self, tree):
        """Check for unused variables."""
        declared = set()
        used = set()

        for node in self._walk(tree):
            if node.type == "VariableDeclarator":
                declared.add(node.id.name)
            elif node.type == "Identifier":
                used.add(node.name)

        unused = declared - used
        for variable in unused:
            self.recommendations.append(
                {
                    "rule": "Unused Variables",
                    "message": f"Variable '{variable}' is declared but not used.",
                }
            )

    def _check_lack_of_comments(self, tree):
        """Check for lack of comments in functions."""
        for node in self._walk(tree):
            if node.type == "FunctionDeclaration":
                if not hasattr(node, "leadingComments") or not node.leadingComments:
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'

                    self.recommendations.append(
                        {
                            "rule": "Lack of Comments",
                            "message": f"Function '{node.id.name}' lacks comments or documentation.",
                            "line": line,
                        }
                    )

    def _check_hard_coded_secrets(self, tree):
        """Check for hardcoded secrets like passwords or API keys."""
        secret_patterns = [r'(?i)(password|apiKey|token)\s*[:=]\s*["\'].*["\']']
        for node in self._walk(tree):
            if node.type == "Literal" and isinstance(node.value, str):
                for pattern in secret_patterns:
                    if re.search(pattern, node.value):
                        line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'

                        self.recommendations.append(
                            {
                                "rule": "Hardcoded Secrets",
                                "message": f"Hardcoded secret detected: '{node.value}'.",
                                "line": line,
                            }
                        )

    # def _check_sql_injection(self, tree):
    #     """Check for potential SQL injection vulnerabilities."""
    #     for node in self._walk(tree):
    #         if node.type == "CallExpression" and hasattr(node.callee, "property"):
    #             if node.callee.property.name in ["query", "execute"]:
    #                 for arg in node.arguments:
    #                     if arg.type == "BinaryExpression" and arg.operator == "+":
    #                         line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
    #                         self.recommendations.append(
    #                             {
    #                                 "rule": "SQL Injection",
    #                                 "message": "Dynamic query detected. Use parameterized queries to prevent SQL injection.",
    #                                 "line":line,
    #                             }
    #                         )

    def _check_sql_injection(self, tree):
        for node in self._walk(tree):
            if node.type == "CallExpression":
                # Ensure `callee` is a MemberExpression with a property
                if hasattr(node.callee, "type") and node.callee.type == "MemberExpression":
                    property_name = getattr(node.callee.property, "name", None)
                    if property_name in ["query", "execute"]:
                        # Perform SQL injection detection
                        for arg in node.arguments:
                            if arg.type == "BinaryExpression" and getattr(arg, "operator", None) == "+":
                                line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                                self.recommendations.append(
                                    {
                                        "rule": "SQL Injection",
                                        "message": "Dynamic query detected. Use parameterized queries to prevent SQL injection.",
                                        "line": line,
                                    }
                                )


    def _check_insecure_file_handling(self, tree):
        """Check for insecure file handling."""
        for node in self._walk(tree):
            if node.type == "CallExpression" and hasattr(node.callee, "name"):
                if node.callee.name in ["fs.open", "fs.readFile", "fs.writeFile"]:
                    args = [arg.value for arg in node.arguments if arg.type == "Literal"]
                    if "encoding" not in args:
                        line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                        self.recommendations.append(
                            {
                                "rule": "Insecure File Handling",
                                "message": "File operation detected without specifying encoding. Specify an encoding for safety.",
                                "line": line,
                            }
                        )

    # def _check_weak_cryptography(self, tree):
    #     """Check for usage of weak cryptographic algorithms."""
    #     weak_algorithms = ["md5", "sha1"]
    #     for node in self._walk(tree):
    #         if node.type == "CallExpression" and hasattr(node.callee, "property"):
    #             if node.callee.property.name in weak_algorithms:
    #                 line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
    #                 self.recommendations.append(
    #                     {
    #                         "rule": "Weak Cryptography",
    #                         "message": f"Usage of weak cryptographic algorithm '{node.callee.property.name}'. Use a stronger algorithm like SHA256.",
    #                         "line": line,
    #                     }
    #                 )
    def _check_weak_cryptography(self, tree):
        weak_algorithms = ["md5", "sha1"]
        for node in self._walk(tree):
            if node.type == "CallExpression":
                # Ensure `callee` is a MemberExpression
                if hasattr(node.callee, "type") and node.callee.type == "MemberExpression":
                    property_name = getattr(node.callee.property, "name", None)
                    if property_name in weak_algorithms:
                        line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                        self.recommendations.append(
                            {
                                "rule": "Weak Cryptography",
                                "message": f"Usage of weak cryptographic algorithm '{property_name}'. Use a stronger algorithm like SHA256.",
                                "line": line,
                            }
                        )


    def _check_empty_exception_handling(self, tree):
        """Check for empty exception handling."""
        for node in self._walk(tree):
            if node.type == "TryStatement":
                if not node.handler or not node.handler.body.body:
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                    self.recommendations.append(
                        {
                            "rule": "Empty Exception Handling",
                            "message": "Empty or overly broad exception handling detected.",
                            "line": line,
                        }
                    )

    def _check_magic_numbers(self, tree, allowed_constants={0, 1, -1}):
        """Check for magic numbers."""
        for node in self._walk(tree):
            if node.type == "Literal" and isinstance(node.value, (int, float)):
                if node.value not in allowed_constants:
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                    self.recommendations.append(
                        {
                            "rule": "Magic Numbers",
                            "message": f"Magic number {node.value} found. Consider defining it as a constant.",
                            "line": line,
                        }
                    )

    def _check_long_chained_calls(self, tree):
        """Check for long chained method calls."""
        for node in self._walk(tree):
            if node.type == "MemberExpression":
                chain_length = 0
                current = node
                while current.type == "MemberExpression":
                    chain_length += 1
                    current = current.object
                if chain_length > 3:  # Example threshold
                    self.recommendations.append(
                        {
                            "rule": "Long Chained Method Calls",
                            "message": f"Detected long chained method call with a length of {chain_length}. Consider breaking it into smaller calls.",
                            "line": node.loc.start.line,
                        }
                    )

    def _check_deprecated_libraries(self, tree):
        """Check for usage of deprecated libraries."""
        deprecated_libraries = {
            "fs.promises": "Deprecated. Use async/await with fs directly.",
            "crypto.createCipher": "Deprecated. Use crypto.createCipheriv instead.",
        }
        for node in self._walk(tree):
            if node.type == "MemberExpression" and node.object.type == "Identifier":
                library = f"{node.object.name}.{node.property.name}"
                if library in deprecated_libraries:
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'

                    self.recommendations.append(
                        {
                            "rule": "Deprecated Libraries",
                            "message": f"Usage of deprecated library '{library}'. {deprecated_libraries[library]}",
                            "line": line,
                        }
                    )

    def _check_excessive_global_variables(self, tree):
        """Check for excessive use of global variables."""
        global_vars = set()
        for node in self._walk(tree):
            if node.type == "VariableDeclaration" and node.kind == "var":  # 'var' declarations are global by default
                for decl in node.declarations:
                    global_vars.add(decl.id.name)

        if len(global_vars) > 5:  # Example threshold
            self.recommendations.append(
                {
                    "rule": "Excessive Global Variables",
                    "message": f"Excessive use of global variables detected ({len(global_vars)}). Avoid using globals.",
                }
            )

    def _check_resource_leaks(self, tree):
        """Check for resource leaks, such as open files or connections."""
        for node in self._walk(tree):
            if node.type == "CallExpression" and hasattr(node.callee, "name"):
                if node.callee.name in ["open", "readFile", "createReadStream"]:
                    has_close = False
                    for sibling in getattr(node, "parent", {}).get("body", []):
                        if sibling.type == "CallExpression" and hasattr(sibling.callee, "property"):
                            if sibling.callee.property.name == "close":
                                has_close = True
                                break
                    if not has_close:
                        line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'

                        self.recommendations.append(
                            {
                                "rule": "Resource Leaks",
                                "message": "Resource (e.g., file or stream) opened without being properly closed.",
                                "line": line,
                            }
                        )

    def _check_circular_imports(self, tree):
        """Check for circular imports."""
        imports = set()
        for node in self._walk(tree):
            if node.type == "ImportDeclaration":
                imports.add(node.source.value)

        # Simulate detecting circular imports (actual detection might require a dependency graph)
        if len(imports) != len(set(imports)):
            self.recommendations.append(
                {
                    "rule": "Circular Imports",
                    "message": "Circular import detected. Avoid circular dependencies between modules.",
                }
            )

    def _check_exception_messages(self, tree):
        """Check if exceptions include meaningful messages."""
        for node in self._walk(tree):
            if node.type == "ThrowStatement":
                if hasattr(node, "argument") and hasattr(node.argument, "callee"):
                    callee_name = getattr(node.argument.callee, "name", None)
                    if callee_name and not node.argument.arguments:
                        line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                        self.recommendations.append(
                            {
                                "rule": "Improper Exception Messages",
                                "message": "Exception thrown without a meaningful message.",
                                "line": line,
                            }
                        )


    def _check_unreachable_code(self, tree):
        """Check for unreachable code."""
        for node in self._walk(tree):
            if node.type == "BlockStatement" and hasattr(node, "body"):
                for i, stmt in enumerate(node.body[:-1]):
                    if stmt.type in ["ReturnStatement", "ThrowStatement", "BreakStatement", "ContinueStatement"]:
                        for unreachable in node.body[i + 1:]:
                            line_stmt = getattr(stmt.loc, 'start', {}).get('line', 'unknown') if hasattr(stmt, 'loc') else 'unknown'
                            line_unreachable = getattr(unreachable.loc, 'start', {}).get('line', 'unknown') if hasattr(unreachable, 'loc') else 'unknown'
                            self.recommendations.append(
                                {
                                    "rule": "Unreachable Code",
                                    "message": f"Unreachable code detected after line {line_stmt}.",
                                    "line": line_unreachable,
                                }
                            )
                        break


    def _check_variable_shadowing(self, tree):
        """Check for variable shadowing with built-in names."""
        built_ins = {"eval", "arguments", "NaN", "undefined", "Infinity"}
        for node in self._walk(tree):
            if node.type == "VariableDeclarator":
                name = getattr(node.id, "name", None)
                if name in built_ins:
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                    self.recommendations.append(
                        {
                            "rule": "Variable Shadowing",
                            "message": f"Variable '{name}' shadows a built-in JavaScript name.",
                            "line": line,
                        }
                    )


    def _check_naming_conventions(self, tree):
        """Check if variables and functions follow camelCase naming conventions."""
        def is_camel_case(name):
            return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name))

        for node in self._walk(tree):
            if node.type in ["FunctionDeclaration", "VariableDeclarator"]:
                name = getattr(node.id, "name", None)
                if name and not is_camel_case(name):
                    line = getattr(node.loc, 'start', {}).get('line', 'unknown') if hasattr(node, 'loc') else 'unknown'
                    self.recommendations.append(
                        {
                            "rule": "Non-JavaScript Naming",
                            "message": f"'{name}' does not follow JavaScript's camelCase naming convention.",
                            "line": line,
                        }
                    )


    # def _walk(self, node):
    #     """Helper method to recursively walk through the AST."""
    #     yield node
    #     for child in getattr(node, "body", []) or []:
    #         yield from self._walk(child)

    def _walk(self, node):
        """Helper to traverse the AST."""
        if node is None:
            return
        yield node
        for key in dir(node):
            value = getattr(node, key, None)
            if isinstance(value, list):
                for child in value:
                    if hasattr(child, "type"):
                        yield from self._walk(child)
            elif hasattr(value, "type"):
                yield from self._walk(value)



# Example Usage
if __name__ == "__main__":
    js_code = """
    function authenticateUser(username, password) {
        if (username === "admin" && password === "secret") {
            return true;
        }
        return false;
    }

    var unusedVar = "test";
    """
    analyzer = JavaScriptCodeAnalyzer(js_code)
    recommendations = analyzer.generate_recommendations()
    print(json.dumps(recommendations, indent=2))