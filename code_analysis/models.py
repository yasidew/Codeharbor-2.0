


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


class CodeAnalysisHistory(models.Model):
    """Stores past code analysis results for trend comparison."""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="analysis_history")
    analyzed_at = models.DateTimeField(auto_now_add=True)
    lines_of_code = models.IntegerField()
    duplicate_code_percentage = models.FloatField()
    complexity_score = models.FloatField()
    security_issues = models.IntegerField(default=0)  # Store security issue count

    def __str__(self):
        return f"Analysis for {self.project.name} - {self.analyzed_at}"

############################ java ######################################

class JavaProject(models.Model):
    """Stores project metadata for Java analysis."""
    name = models.CharField(max_length=255, unique=True, default="Untitled Java Project")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class JavaCodeSnippet(models.Model):
    """Stores individual Java code snippets and their analysis results."""
    project = models.ForeignKey(JavaProject, on_delete=models.CASCADE, related_name="java_snippets")
    snippet = models.TextField()  # Store Java code snippets
    ai_suggestion = models.TextField()  # Store AI suggestion for this snippet
    model_suggestion = models.TextField()  # Store CodeT5 suggestion for this snippet
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Java Snippet from {self.project.name} - {self.created_at}"