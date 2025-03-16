from django.test import TestCase, Client
from django.urls import reverse
from code_analysis.models import CodeSnippet, Project  # ✅ DB Models
from Drinks.views import analyze_code_view, ai_code_analysis  # ✅ Analysis logic

import json

class CodeAnalysisTest(TestCase):
    """
    Unit tests for the code analysis system.
    """
    databases = {'default', 'code_analysis'}

    def setUp(self):
        """Initialize test data before each test."""
        self.client = Client()
        self.test_project = Project.objects.create(name="Test Project")
        self.test_code_snippet = "def test_function():\n    print('Hello, World!')"
        self.analysis_url = reverse('analyze_code')  # ✅ Ensure this matches your `urls.py`

    def test_analyze_code_view_successful(self):
        """✅ Test if analyze_code_view successfully processes input and returns suggestions."""
        response = self.client.post(self.analysis_url, {
            'code': self.test_code_snippet,
            'project_name': 'Test Project'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Analysis Results")
        self.assertContains(response, "suggestion")  # Ensure suggestions are returned

    def test_ai_code_analysis_function(self):
        """✅ Test if AI model generates a suggestion for a given code snippet."""
        suggestion = ai_code_analysis(self.test_code_snippet)
        self.assertIsInstance(suggestion, str)  # Ensure output is a string
        self.assertGreater(len(suggestion), 10)  # Ensure AI provides a meaningful response

    def test_code_snippet_storage(self):
        """✅ Test if analyzed code snippets are correctly stored in the database."""
        snippet = CodeSnippet.objects.create(
            project=self.test_project,
            snippet=self.test_code_snippet,
            ai_suggestion="Use print() efficiently.",
            model_suggestion="Refactor using a logger."
        )

        saved_snippet = CodeSnippet.objects.get(id=snippet.id)
        self.assertEqual(saved_snippet.snippet, self.test_code_snippet)
        self.assertEqual(saved_snippet.ai_suggestion, "Use print() efficiently.")
        self.assertEqual(saved_snippet.model_suggestion, "Refactor using a logger.")


    def test_reject_non_python_code(self):
        """✅ Test if the system rejects non-Python code with an appropriate error message."""
        non_python_code = "<html><body><h1>Not Python Code</h1></body></html>"
        response = self.client.post(self.analysis_url, {
            'code': non_python_code,
            'project_name': 'Test Project'
        })

        self.assertEqual(response.status_code, 200)  # View should render the page, not crash
        self.assertContains(response, "Error: The uploaded file `Pasted Code` is not valid Python code.")


    def test_empty_code_submission(self):
        """✅ Test if the view returns an error when no code is provided."""
        response = self.client.post(self.analysis_url, {'code': '', 'project_name': 'Test Project'})

        self.assertEqual(response.status_code, 200)  # Should render the page, not fail
        self.assertContains(response, "No code provided for analysis.")  # Check error message


    # def test_invalid_code_submission(self):
    #     """❌ Test if the view returns an error when no code is provided."""
    #     response = self.client.post(self.analysis_url, {'code': ''})
    #     self.assertContains(response, "No code provided for analysis.")

if __name__ == "__main__":
    import unittest
    unittest.main()
