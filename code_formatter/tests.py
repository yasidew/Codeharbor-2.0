from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import Guideline, CodeRefactoringRecord, DesignPatternResource
import json
import base64
import os


class RefactorViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.refactor_url = reverse('refactor_view')

    def test_refactor_view_get(self):
        """Test if refactor view renders correctly."""
        response = self.client.get(self.refactor_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'code_formatter/refactor.html')


class UploadCodeTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('upload_code')

    def test_upload_code_with_valid_file(self):
        """Test file upload with a valid text file."""
        file_content = "print('Hello, World!')"
        uploaded_file = SimpleUploadedFile("test.py", file_content.encode('utf-8'), content_type="text/plain")

        response = self.client.post(self.upload_url, {'file': uploaded_file})

        self.assertEqual(response.status_code, 200)
        self.assertIn("code", response.json())
        self.assertEqual(response.json()["code"], file_content)

    def test_upload_code_without_file(self):
        """Test file upload failure when no file is provided."""
        response = self.client.post(self.upload_url)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


class RefactorCodeTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.refactor_url = reverse('refactor_code')

    def test_refactor_code_with_valid_input(self):
        """Test refactoring functionality with valid input."""
        test_code = "def hello():\n    print('Hello, World!')"
        response = self.client.post(
            self.refactor_url,
            json.dumps({'code': test_code, 'use_guidelines': False}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("refactored_code", response.json())
        self.assertIn("original_loc", response.json())

    def test_refactor_code_with_missing_input(self):
        """Test refactoring failure with missing code input."""
        response = self.client.post(
            self.refactor_url,
            json.dumps({'use_guidelines': True}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


class GetGuidelinesTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.guideline_url = reverse('get_guidelines')

    def test_get_guidelines_empty(self):
        """Test fetching guidelines when none exist."""
        response = self.client.get(self.guideline_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["guidelines"], [])

    def test_get_guidelines_with_data(self):
        """Test fetching guidelines when some exist."""
        Guideline.objects.create(pattern="Factory", rule="Use dependency injection.")

        response = self.client.get(self.guideline_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["guidelines"]), 1)


class DefineGuidelinesTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.define_guideline_url = reverse('define_guidelines')

    def test_define_guidelines_get(self):
        """Test rendering define guidelines page."""
        response = self.client.get(self.define_guideline_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'code_formatter/define_guidelines.html')

    def test_define_guidelines_post(self):
        """Test creating a new guideline."""
        response = self.client.post(self.define_guideline_url, {'pattern': 'Singleton', 'rule': 'Ensure single instance.'})
        self.assertEqual(response.status_code, 302)  # Should redirect


class DeleteGuidelineTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.guideline = Guideline.objects.create(pattern="Factory", rule="Use dependency injection.")
        self.delete_url = reverse('delete_guideline', args=[self.guideline.id])

    def test_delete_guideline(self):
        """Test deleting an existing guideline."""
        response = self.client.post(self.delete_url)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Guideline.objects.filter(id=self.guideline.id).exists())


class FetchGitHubFileTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.github_file_url = reverse('fetch_github_file')

    def test_fetch_github_file_with_missing_data(self):
        """Test GitHub file fetch with missing fields."""
        response = self.client.post(
            self.github_file_url,
            json.dumps({'repo_url': 'https://github.com/user/repo'}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


class CreateGitHubPRTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.github_pr_url = reverse('create_github_pr')

    def test_create_github_pr_with_missing_data(self):
        """Test creating GitHub PR with missing fields."""
        response = self.client.post(
            self.github_pr_url,
            json.dumps({'repo_url': 'https://github.com/user/repo', 'file_path': ''}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


class AddResourceTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.add_resource_url = reverse('add_resource')

    def test_add_resource_with_valid_data(self):
        """Test adding a new design pattern resource."""
        response = self.client.post(
            self.add_resource_url,
            {'title': 'Strategy Pattern', 'description': 'A way to manage behaviors dynamically.'}
        )

        self.assertEqual(response.status_code, 302)  # Should redirect


class ListResourcesTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.list_resources_url = reverse('list_resources')
        DesignPatternResource.objects.create(title="Factory Pattern", description="Creates objects without specifying exact class.")

    def test_list_resources(self):
        """Test retrieving design pattern resources."""
        response = self.client.get(self.list_resources_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Factory Pattern")


class RefactorAndCompareTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.refactor_compare_url = reverse('refactor_and_compare')

    def test_refactor_and_compare_with_valid_input(self):
        """Test code refactoring and comparison."""
        test_code = "def add(a, b): return a + b"

        response = self.client.post(
            self.refactor_compare_url,
            json.dumps({'code': test_code}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("after_code", response.json())

    def test_refactor_and_compare_with_missing_code(self):
        """Test code refactoring and comparison failure with missing code."""
        response = self.client.post(
            self.refactor_compare_url,
            json.dumps({}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
