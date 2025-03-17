from django.test import TestCase, Client
from django.urls import reverse
from code_analysis.models import JavaCodeSnippet, JavaProject  # ✅ Java models

class JavaCodeAnalysisTest(TestCase):
    """
    Unit tests for Java code analysis system.
    """
    databases = {'default', 'code_analysis'}

    def setUp(self):
        """Initialize test data before each test."""
        self.client = Client()
        self.test_project = JavaProject.objects.create(name="Test Java Project")
        self.test_java_code = """ 
        public class TestClass {
            public void testMethod() {
                System.out.println("Hello, Java!");
            }
        }
        """
        self.analysis_url = reverse('java_code_analysis')  # ✅ Ensure it matches `urls.py`

    def test_analyze_java_code_view_successful(self):
        """✅ Test if java_code_analysis successfully processes input and returns suggestions."""
        response = self.client.post(self.analysis_url, {
            'code': self.test_java_code,
            'project_name': 'Test Java Project'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Analysis Results")
        self.assertContains(response, "suggestion")  # Ensure suggestions are returned

    def test_reject_non_java_code(self):
        """✅ Test if the system rejects non-Java code with an appropriate error message."""
        non_java_code = "<html><body><h1>Not Java Code</h1></body></html>"
        response = self.client.post(self.analysis_url, {
            'code': non_java_code,
            'project_name': 'Test Java Project'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "🚨 The pasted code is not valid Java!")  # ✅ Match actual message


    def test_java_code_snippet_storage(self):
        """✅ Test if analyzed Java code snippets are correctly stored in the database."""
        snippet = JavaCodeSnippet.objects.create(
            project=self.test_project,
            snippet=self.test_java_code,
            ai_suggestion="Use best practices for logging.",
            model_suggestion="Consider refactoring for better readability."
        )

        saved_snippet = JavaCodeSnippet.objects.get(id=snippet.id)
        self.assertEqual(saved_snippet.snippet, self.test_java_code)
        self.assertEqual(saved_snippet.ai_suggestion, "Use best practices for logging.")
        self.assertEqual(saved_snippet.model_suggestion, "Consider refactoring for better readability.")
