
from django.contrib import admin
from .models import Project, CodeSnippet, CodeAnalysisHistory, JavaProject, JavaCodeSnippet


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")  # Display project name and creation time
    search_fields = ("name",)  # Allow searching by project name
    ordering = ("-created_at",)  # Order by latest created projects

@admin.register(CodeSnippet)
class CodeSnippetAdmin(admin.ModelAdmin):
    list_display = ("project", "created_at")  # Show project name and created time
    search_fields = ("project__name", "snippet")  # Search by project name and snippet
    list_filter = ("project",)  # Add filter by project


@admin.register(CodeAnalysisHistory)
class CodeAnalysisHistoryAdmin(admin.ModelAdmin):
    list_display = ("project", "analyzed_at", "complexity_score", "duplicate_code_percentage", "security_issues")
    search_fields = ("project__name",)
    list_filter = ("project", "analyzed_at")


### 🔹 **Java Code Analysis Admin** ###
@admin.register(JavaProject)
class JavaProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")  # Display project name and creation time
    search_fields = ("name",)  # Allow searching by project name
    ordering = ("-created_at",)  # Order by latest created projects


@admin.register(JavaCodeSnippet)
class JavaCodeSnippetAdmin(admin.ModelAdmin):
    list_display = ("project", "created_at")  # Show project name and created time
    search_fields = ("project__name", "snippet")  # Search by project name and snippet
    list_filter = ("project",)  # Add filter by project

