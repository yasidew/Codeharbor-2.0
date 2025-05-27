import time
from django.test import TestCase, Client
from django.urls import reverse
from code_analysis.models import Project

class CodeAnalysisPerformanceTest(TestCase):
    """
    Performance tests for the code analysis system.
    """
    databases = {'default', 'code_analysis'}

    def setUp(self):
        """Initialize test data before each test."""
        self.client = Client()
        self.test_project = Project.objects.create(name="Performance Test Project")
        self.analysis_url = reverse('analyze_code')

        # Large Python code snippet (50x duplication to simulate large input)
        self.large_code_snippet = "\n".join([
                                                "def large_test_function():",
                                                "    for i in range(1000):",
                                                "        print(f'Iteration: {i}')"
                                            ] * 50)  # Simulating a large codebase

    def test_performance_large_code(self):
        """✅ Test how long the analysis takes for a large Python codebase."""
        start_time = time.time()

        response = self.client.post(self.analysis_url, {
            'code': self.large_code_snippet,
            'project_name': 'Performance Test Project'
        })

        end_time = time.time()
        duration = end_time - start_time
        print(f"🚀 Performance Test: Analyzing large code took {duration:.2f} seconds")

        self.assertEqual(response.status_code, 200)
        self.assertLess(duration, 20.0, "❌ Performance test failed! Took too long.")  # Adjust threshold as needed
