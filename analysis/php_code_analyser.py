from subprocess import Popen, PIPE

from phply import phpast
from phply.phpparse import make_parser
import re
import json


class PHPCodeAnalyzer:
    def __init__(self, code):
        self.code = code  # PHP code to analyze
        self.recommendations = []  # List to store recommendations
        # self.parser = make_parser()  # Create a PHP parser

    def validate_php_syntax(self):
        """Validate PHP syntax using `php -l`."""
        process = Popen(["php", "-l"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate(input=self.code.encode())
        return process.returncode == 0, stderr.decode()

    # def generate_recommendations(self):
    #     """Run the analysis and generate recommendations."""
    #     try:
    #         # Parse the PHP code into an AST
    #         tree = self.parser.parse(self.code)
    #         # Apply custom rules
    #         self.custom_rules_check(tree)
    #     except Exception as e:
    #         self.recommendations.append({"rule": "Parsing Error", "message": f"Error parsing code: {e}"})
    #
    #     # Deduplicate recommendations
    #     self.deduplicate_recommendations()
    #
    #     return self.recommendations

    def generate_recommendations(self):
        """Run the analysis and generate recommendations."""
        is_valid, error_message = self.validate_php_syntax()
        if not is_valid:
            self.recommendations.append({
                "rule": "Syntax Error",
                "message": f"PHP syntax error: {error_message}",
                "line": "unknown",
            })
            return self.recommendations

        try:
            # Simulate the analysis process
            # Placeholder for actual PHP analysis logic
            self.recommendations.extend(self.custom_rules_check())
        except Exception as e:
            self.recommendations.append({"rule": "Parsing Error", "message": f"Error parsing code: {e}"})

        self.deduplicate_recommendations()

        return self.recommendations

    def deduplicate_recommendations(self):
        """Remove duplicate recommendations."""
        unique = []
        seen = set()

        for rec in self.recommendations:
            key = (rec.get("rule"), rec.get("line"), rec.get("message"))
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        self.recommendations = unique

    # def custom_rules_check(self, tree):
    #     """Run specific custom checks on the AST."""
    #     for node in tree:
    #         try:
    #             if isinstance(node, phpast.Function):
    #                 self._check_function_length(node)
    #                 self._check_too_many_parameters(node)
    #                 self._check_lack_of_comments(node)
    #                 self._check_nesting_depth(node)
    #             if isinstance(node, phpast.Assignment):
    #                 self._check_hard_coded_secrets(node)
    #                 self._check_unused_variables(node)
    #             if isinstance(node, phpast.MethodCall):
    #                 self._check_sql_injection(node)
    #                 self._check_insecure_file_handling(node)
    #                 self._check_weak_cryptography(node)
    #             if isinstance(node, phpast.Try):
    #                 self._check_empty_exception_handling(node)
    #             if isinstance(node, phpast.Global):
    #                 self._check_excessive_global_variables(node)
    #             if isinstance(node, phpast.Throw):
    #                 self._check_exception_messages(node)
    #             if isinstance(node, phpast.Return):
    #                 self._check_unreachable_code(node)
    #             if isinstance(node, phpast.Variable):
    #                 self._check_variable_shadowing(node)
    #                 self._check_naming_conventions(node)
    #         except Exception as e:
    #             self.recommendations.append({"rule": "Rule Error", "message": f"Error processing node: {e}"})

    def custom_rules_check(self, tree):
        """Run specific custom checks on the AST."""
        for node in tree:
            try:
                # Check for specific node types and run relevant rules
                if isinstance(node, phpast.Function):
                    self._run_safe(self._check_function_length, node)
                    self._run_safe(self._check_too_many_parameters, node)
                    self._run_safe(self._check_lack_of_comments, node)
                    self._run_safe(self._check_nesting_depth, node)
                elif isinstance(node, phpast.Assignment):
                    self._run_safe(self._check_hard_coded_secrets, node)
                    self._run_safe(self._check_unused_variables, node)
                elif isinstance(node, phpast.MethodCall):
                    self._run_safe(self._check_sql_injection, node)
                    self._run_safe(self._check_insecure_file_handling, node)
                    self._run_safe(self._check_weak_cryptography, node)
                elif isinstance(node, phpast.Try):
                    self._run_safe(self._check_empty_exception_handling, node)
                elif isinstance(node, phpast.Global):
                    self._run_safe(self._check_excessive_global_variables, node)
                elif isinstance(node, phpast.Throw):
                    self._run_safe(self._check_exception_messages, node)
                elif isinstance(node, phpast.Return):
                    self._run_safe(self._check_unreachable_code, node)
                elif isinstance(node, phpast.Variable):
                    self._run_safe(self._check_variable_shadowing, node)
                    self._run_safe(self._check_naming_conventions, node)
                else:
                    # Log unknown node types for debugging
                    self.recommendations.append({
                        "rule": "Unknown Node Type",
                        "message": f"Encountered unknown node type: {type(node).__name__}.",
                        "line": getattr(node, "lineno", "unknown"),
                    })
            except Exception as e:
                self.recommendations.append({
                    "rule": "Node Processing Error",
                    "message": f"Error processing node of type {type(node).__name__}: {e}",
                    "line": getattr(node, "lineno", "unknown"),
                })

    def _run_safe(self, rule_check_function, node):
        """Run a rule check function safely and catch any exceptions."""
        try:
            rule_check_function(node)
        except Exception as e:
            self.recommendations.append({
                "rule": rule_check_function.__name__,
                "message": f"Error in rule '{rule_check_function.__name__}': {e}",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_function_length(self, node):
        """Check if functions are too long."""
        if len(node.body.nodes) > 20:  # Example threshold
            self.recommendations.append({
                "rule": "Function Length",
                "message": f"Function '{node.name}' exceeds 20 lines.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_too_many_parameters(self, node):
        """Check if functions have too many parameters."""
        if len(node.params) > 5:
            self.recommendations.append({
                "rule": "Too Many Parameters",
                "message": f"Function '{node.name}' has too many parameters ({len(node.params)}).",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_lack_of_comments(self, node):
        """Detect functions without comments."""
        if not getattr(node, "doc", None):
            self.recommendations.append({
                "rule": "Lack of Comments",
                "message": f"Function '{node.name}' lacks comments or documentation.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_hard_coded_secrets(self, node):
        """Detect hardcoded secrets like passwords or API keys."""
        secret_patterns = [r'(?i)(password|api_key|token)\s*=\s*[\'"].*[\'"]']
        if isinstance(node.expr, phpast.String):
            for pattern in secret_patterns:
                if re.search(pattern, node.expr.value):
                    self.recommendations.append({
                        "rule": "Hardcoded Secrets",
                        "message": f"Hardcoded secret detected: '{node.expr.value}'.",
                        "line": getattr(node, "lineno", "unknown"),
                    })

    def _check_nesting_depth(self, node):
        """Detect excessive nesting."""

        def calculate_depth(body, depth=0):
            return max(
                (calculate_depth(child.body, depth + 1) for child in body if hasattr(child, 'body')),
                default=depth
            )

        depth = calculate_depth(node.body.nodes)
        if depth > 4:  # Example threshold
            self.recommendations.append({
                "rule": "Nesting Depth",
                "message": f"Function '{node.name}' exceeds allowed nesting depth ({depth}).",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_sql_injection(self, node):
        """Detect potential SQL injection vulnerabilities."""
        if node.name == "query" and isinstance(node.params[0], phpast.BinaryOp):
            self.recommendations.append({
                "rule": "SQL Injection",
                "message": "Dynamic query detected. Use prepared statements to prevent SQL injection.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_empty_exception_handling(self, node):
        """Detect empty exception handling blocks."""
        for handler in node.catches:
            if not handler.body:
                self.recommendations.append({
                    "rule": "Empty Exception Handling",
                    "message": "Empty or overly broad exception handling detected.",
                    "line": getattr(handler, "lineno", "unknown"),
                })

    def _check_unused_variables(self, node):
        """Detect unused variables."""
        declared = set(node.name)
        used = set(ref.name for ref in node.refs)
        unused = declared - used
        if unused:
            self.recommendations.append({
                "rule": "Unused Variables",
                "message": f"Unused variables detected: {', '.join(unused)}.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_weak_cryptography(self, node):
        """Detect weak cryptographic practices."""
        weak_algorithms = {"md5", "sha1"}
        if node.name in weak_algorithms:
            self.recommendations.append({
                "rule": "Weak Cryptography",
                "message": f"Usage of weak cryptographic algorithm: {node.name}.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_insecure_file_handling(self, node):
        """Detect insecure file handling."""
        insecure_methods = ["fopen", "file_get_contents", "file_put_contents"]
        if node.name in insecure_methods and "UTF-8" not in node.args:
            self.recommendations.append({
                "rule": "Insecure File Handling",
                "message": f"Insecure file handling detected in method '{node.name}'.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_long_chained_calls(self, node):
        """Detect long chained method calls."""
        chain_length = 0
        while isinstance(node, phpast.MethodCall):
            chain_length += 1
            node = node.receiver
        if chain_length > 3:  # Example threshold
            self.recommendations.append({
                "rule": "Long Chained Method Calls",
                "message": f"Long chained method call detected with length {chain_length}.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_excessive_global_variables(self, node):
        """Detect excessive use of global variables."""
        if len(node.names) > 5:  # Example threshold
            self.recommendations.append({
                "rule": "Excessive Global Variables",
                "message": "Excessive global variables detected.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_exception_messages(self, node):
        """Ensure exception messages are meaningful."""
        if not node.expr.args:
            self.recommendations.append({
                "rule": "Exception Messages",
                "message": "Exception thrown without a meaningful message.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_unreachable_code(self, node):
        """Detect unreachable code."""
        if hasattr(node, "children") and node.children:
            for child in node.children:
                if isinstance(child, (phpast.Return, phpast.Throw)):
                    self.recommendations.append({
                        "rule": "Unreachable Code",
                        "message": f"Unreachable code detected after line {getattr(child, 'lineno', 'unknown')}.",
                        "line": getattr(child, "lineno", "unknown"),
                    })

    def _check_variable_shadowing(self, node):
        """Detect shadowing of variables."""
        if node.name in {"_SERVER", "_POST", "_GET"}:
            self.recommendations.append({
                "rule": "Variable Shadowing",
                "message": f"Variable '{node.name}' shadows a built-in PHP superglobal.",
                "line": getattr(node, "lineno", "unknown"),
            })

    def _check_naming_conventions(self, node):
        """Ensure variables and methods follow PHP naming conventions."""
        def is_snake_case(name):
            return bool(re.match(r"^[a-z][a-z0-9_]*$", name))

        if not is_snake_case(node.name):
            self.recommendations.append({
                "rule": "Non-PHP Naming",
                "message": f"'{node.name}' does not follow PHP's snake_case naming convention.",
                "line": getattr(node, "lineno", "unknown"),
            })


# Example Usage
if __name__ == "__main__":
    php_code = """
    <?php
    $password = "hardcoded_secret"; // Hardcoded secret

    function fetchData($username, $password) {
        $query = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
        mysqli_query($query); // SQL Injection
    }

    try {
        // Empty exception handling
    } catch (Exception $e) {}
    ?>
    """
    analyzer = PHPCodeAnalyzer(php_code)
    recommendations = analyzer.generate_recommendations()
    print(json.dumps(recommendations, indent=2))
