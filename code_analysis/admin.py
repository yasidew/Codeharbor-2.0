# from django.contrib import admin
# from .models import CodeAnalysis
#
# admin.site.register(CodeAnalysis)
# from django.contrib import admin
#
# # Register your models here.
from django.contrib import admin
from .models import Project, CodeSnippet

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

