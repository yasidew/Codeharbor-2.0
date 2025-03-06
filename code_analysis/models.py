from django.db import models

# class CodeAnalysis(models.Model):
#     project_name = models.CharField(max_length=255)  # Track per project
#     code_snippet = models.TextField()  # Original code
#     ai_suggestion = models.TextField()  # AI-generated suggestion
#     model_suggestion = models.TextField()  # Your trained model's suggestion
#     timestamp = models.DateTimeField(auto_now_add=True)  # When analyzed
#
#     def __str__(self):
#         return f"{self.project_name} - {self.timestamp}"


from django.db import models

class Project(models.Model):
    """Stores project metadata separately."""
    name = models.CharField(max_length=255, unique=True, default="Untitled Project")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class CodeSnippet(models.Model):
    """Stores individual code snippets and their analysis results."""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="snippets")
    snippet = models.TextField()  # Store individual code snippets
    ai_suggestion = models.TextField()  # Store AI suggestion for this snippet
    model_suggestion = models.TextField()  # Store CodeT5 suggestion for this snippet
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Snippet from {self.project.name} - {self.created_at}"
