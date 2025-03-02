

import json
import javalang
import re


class JavaCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # Java code to be analyzed
        self.recommendations = []  # Recommendations based on the analysis

    def deduplicate_recommendations(self):
        """Remove duplicate recommendations, and group 'Lack of Comments' by method."""
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
                "message": f"{len(lack_of_comments)} methods lack comments. Please review.",
                "details": lack_of_comments  # Provide detailed breakdown if needed
            })

        self.recommendations = unique



    # def safe_get_line(node):
    #     return getattr(node.position, "line", "unknown")


    def generate_recommendations(self):
        """Run the analysis and generate recommendations."""
        try:
            # Parse the Java code into an abstract syntax tree (AST)
            tree = javalang.parse.parse(self.code)
            # Apply custom rules
            self.custom_rules_check(tree)
        except javalang.parser.JavaSyntaxError as e:
            self.recommendations.append({"error": f"Syntax Error: {str(e)}"})
        except Exception as e:
            self.recommendations.append({"error": f"Unexpected error during parsing: {str(e)}"})

        # Deduplicate recommendations
        self.deduplicate_recommendations()

        return self.recommendations

    def custom_rules_check(self, tree):
        """Run specific custom checks on the AST."""
        checks = [
            self._check_method_length,
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
        for check in checks:
            try:
                check(tree)
            except Exception as e:
                self.recommendations.append({"error": f"Error in {check.__name__}: {str(e)}"})

    def _check_method_length(self, tree):
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.body and len(node.body) > 10:
                self.recommendations.append({
                    "rule": "Method Length",
                    "message": f"Method '{node.name}' exceeds 10 lines.",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_too_many_parameters(self, tree):
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if len(node.parameters) > 5:
                self.recommendations.append({
                    "rule": "Too Many Parameters",
                    "message": f"Method '{node.name}' has too many parameters ({len(node.parameters)}).",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_nesting_depth(self, tree):
        def calculate_depth(node, depth=0):
            if hasattr(node, 'children'):
                return max(
                    [calculate_depth(child, depth + 1) for child in node.children if isinstance(child, list)] + [depth])
            return depth

        max_allowed_depth = 4
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.body:
                nesting_depth = calculate_depth(node.body)
                if nesting_depth > max_allowed_depth:
                    self.recommendations.append({
                        "rule": "Nesting Depth",
                        "message": f"Method '{node.name}' exceeds allowed nesting depth ({nesting_depth}).",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_unused_variables(self, tree):
        declared = set()
        used = set()

        for path, node in tree:
            if isinstance(node, javalang.tree.VariableDeclarator):
                declared.add(node.name)
            elif isinstance(node, javalang.tree.MemberReference):
                used.add(node.member)

        unused = declared - used
        if unused:
            self.recommendations.append({
                "rule": "Unused Variables",
                "message": f"Unused variables detected: {', '.join(unused)}.",
            })

    def _check_lack_of_comments(self, tree):
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not node.documentation:
                self.recommendations.append({
                    "rule": "Lack of Comments",
                    "message": f"Method '{node.name}' lacks a docstring or comment.",
                    "line": getattr(node.position, "line", "unknown"),
                })

    # def _check_lack_of_comments(self, tree):
    #     """Detect methods missing comments and summarize them."""
    #     methods_without_comments = []
    #     for path, node in tree.filter(javalang.tree.MethodDeclaration):
    #         if not node.documentation:
    #             methods_without_comments.append(node.name)
    #
    #     if methods_without_comments:
    #         self.recommendations.append({
    #             "rule": "Lack of Comments",
    #             "message": f"The following methods lack comments: {', '.join(methods_without_comments)}.",
    #         })


    def _check_hard_coded_secrets(self, tree):
        secret_patterns = [r'(?i)(password|api_key|token)\s*=\s*[\'"].*[\'"]']
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            if node.initializer and isinstance(node.initializer, javalang.tree.Literal):
                for pattern in secret_patterns:
                    if re.search(pattern, node.initializer.value):
                        self.recommendations.append({
                            "rule": "Hardcoded Secrets",
                            "message": f"Hardcoded secret found in variable '{node.name}'.",
                            "line": getattr(node.position, "line", "unknown"),
                        })

    def _check_sql_injection(self, tree):
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in {"execute", "executemany"}:
                if any(isinstance(arg, javalang.tree.BinaryOperation) for arg in node.arguments):
                    self.recommendations.append({
                        "rule": "SQL Injection",
                        "message": f"Potential SQL injection vulnerability in method '{node.member}'.",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_insecure_file_handling(self, tree):
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in {"FileReader", "FileWriter", "BufferedReader", "BufferedWriter"}:
                if not any(arg.value == "UTF-8" for arg in getattr(node.arguments, "children", [])):
                    self.recommendations.append({
                        "rule": "Insecure File Handling",
                        "message": f"Insecure file handling detected in method '{node.member}'.",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_weak_cryptography(self, tree):
        """Detect weak cryptographic practices."""
        weak_algorithms = {"MD5", "SHA1"}
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in weak_algorithms:
                self.recommendations.append({
                    "rule": "Weak Cryptography",
                    "message": f"Usage of weak cryptographic algorithm: {node.member}.",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_empty_exception_handling(self, tree):
        """Detect empty or overly broad exception handling."""
        for path, node in tree.filter(javalang.tree.TryStatement):
            if not node.block or all(not handler.block for handler in node.catches):
                self.recommendations.append({
                    "rule": "Empty Exception Handling",
                    "message": "Empty or overly broad exception handling detected.",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_magic_numbers(self, tree):
        """Detect magic numbers."""
        allowed_constants = {0, 1, -1}
        for path, node in tree.filter(javalang.tree.Literal):
            if isinstance(node.value, str) and node.value.isdigit():
                value = int(node.value)
                if value not in allowed_constants:
                    self.recommendations.append({
                        "rule": "Magic Numbers",
                        "message": f"Magic number {value} found. Consider defining it as a constant.",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_long_chained_calls(self, tree):
        """Detect long chained method calls."""
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            chain_length = 0
            current = node
            while isinstance(current, javalang.tree.MethodInvocation):
                chain_length += 1
                current = current.qualifier
            if chain_length > 3:
                self.recommendations.append({
                    "rule": "Long Chained Method Calls",
                    "message": f"Long chained method call detected with length {chain_length}.",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_deprecated_libraries(self, tree):
        """Detect usage of deprecated libraries."""
        deprecated_libraries = {"java.util.Vector": "Consider using java.util.ArrayList instead."}
        for path, node in tree.filter(javalang.tree.Type):
            if node.name in deprecated_libraries:
                self.recommendations.append({
                    "rule": "Deprecated Libraries",
                    "message": f"Library '{node.name}' is deprecated. {deprecated_libraries[node.name]}",
                    "line": getattr(node.position, "line", "unknown"),
                })

    def _check_excessive_global_variables(self, tree):
        """Detect excessive use of global variables."""
        global_vars = []
        for path, node in tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                global_vars.append(declarator.name)
        if len(global_vars) > 5:
            self.recommendations.append({
                "rule": "Excessive Global Variables",
                "message": f"Excessive global variables found: {', '.join(global_vars)}.",
            })

    def _check_resource_leaks(self, tree):
        """Detect resource leaks, such as opened resources not being closed."""
        for path, node in tree.filter(javalang.tree.MethodInvocation):
            if node.member in {"FileInputStream", "FileOutputStream", "Socket"}:
                if not any(
                        "close" in getattr(call.member, "lower", lambda: "")()
                        for call_path, call in tree.filter(javalang.tree.MethodInvocation)
                ):
                    self.recommendations.append({
                        "rule": "Resource Leaks",
                        "message": f"Resource opened with '{node.member}' but not properly closed.",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_circular_imports(self, tree):
        """Detect circular imports."""
        imports = set()
        for path, node in tree.filter(javalang.tree.Import):
            imports.add(node.path)
        if len(imports) != len(set(imports)):
            self.recommendations.append({
                "rule": "Circular Imports",
                "message": "Circular import detected.",
            })

    def _check_exception_messages(self, tree):
        """Ensure exception messages are meaningful."""
        for path, node in tree.filter(javalang.tree.ThrowStatement):
            if node.expression and isinstance(node.expression, javalang.tree.MethodInvocation):
                args = node.expression.arguments
                if not args or not isinstance(args[0], javalang.tree.Literal):
                    self.recommendations.append({
                        "rule": "Exception Messages",
                        "message": "Exception thrown without a meaningful message.",
                        "line": getattr(node.position, "line", "unknown"),
                    })

    def _check_unreachable_code(self, tree):
        """Detect unreachable code."""
        for path, node in tree.filter(javalang.tree.BlockStatement):
            has_return_or_throw = False
            for child in node.children:
                if isinstance(child, (javalang.tree.ReturnStatement, javalang.tree.ThrowStatement)):
                    has_return_or_throw = True
                elif has_return_or_throw:
                    self.recommendations.append({
                        "rule": "Unreachable Code",
                        "message": f"Unreachable code detected after line {getattr(child.position, 'line', 'unknown')}.",
                    })

    def _check_variable_shadowing(self, tree):
        """Detect shadowing of variables."""
        variables = {}
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            if node.name in variables:
                self.recommendations.append({
                    "rule": "Variable Shadowing",
                    "message": f"Variable '{node.name}' is shadowing another variable.",
                    "line": getattr(node.position, "line", "unknown"),
                })
            else:
                variables[node.name] = node.position.line

    def _check_naming_conventions(self, tree):
        """Ensure variables and methods follow Java naming conventions."""
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not re.match(r"^[a-z][a-zA-Z0-9]*$", node.name):
                self.recommendations.append({
                    "rule": "Naming Conventions",
                    "message": f"Method name '{node.name}' does not follow camelCase naming convention.",
                    "line": getattr(node.position, "line", "unknown"),
                })
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            if not re.match(r"^[a-z][a-zA-Z0-9]*$", node.name):
                self.recommendations.append({
                    "rule": "Naming Conventions",
                    "message": f"Variable name '{node.name}' does not follow camelCase naming convention.",
                    "line": getattr(node.position, "line", "unknown"),
                })


# Example Usage
if __name__ == "__main__":
    java_code = """
        import java.sql.Connection;
        import java.sql.DriverManager;
        import java.sql.Statement;

        public class VulnerableJavaApp {
            private static final String DB_PASSWORD = "password123"; // Hardcoded secret

            public void fetchData(String userInput) {
                try {
                    Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/db", "user", DB_PASSWORD);
                    Statement stmt = conn.createStatement();
                    String sql = "SELECT * FROM users WHERE name = '" + userInput + "'";
                    stmt.execute(sql); // SQL injection
                } catch (Exception e) {
                    // Empty catch block
                }
            }
        }
    """
    analyzer = JavaCodeAnalyzer(java_code)
    recommendations = analyzer.generate_recommendations()
    print(json.dumps(recommendations, indent=2))
