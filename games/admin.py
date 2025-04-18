# # from django.contrib import admin
# #
# # # Register your models here.
# # from django.contrib import admin
# # from .models import GitHubChallenge  # Import your model
# #
# # @admin.register(GitHubChallenge)
# # class GitHubChallengeAdmin(admin.ModelAdmin):
# #     list_display = ("title", "repo_url", "file_url", "html_preview", "created_at")  # ✅ Added html_preview
# #
# #     def html_preview(self, obj):
# #         return obj.html_code[:100] + "..."  # ✅ Show first 100 chars
# #     html_preview.short_description = "HTML Code Preview"  # ✅ Column Name
# #
# #     list_display_links = ("title",)
# #     search_fields = ("title", "repo_url")
# #     list_filter = ("created_at",)
#
#
# from django.contrib import admin
# from .models import GitHubChallenge, GitHubScraperGame, GitGameScore  # Import your model
#
# from django.contrib import admin
# from .models import GitHubChallenge, GitHubScraperGame, GitGameScore  # Import your model
#
# @admin.register(GitHubChallenge)
# class GitHubChallengeAdmin(admin.ModelAdmin):
#     list_display = ("title", "difficulty", "repo_url", "file_url", "html_preview", "created_at")  # ✅ Added "difficulty"
#
#     def html_preview(self, obj):
#         return obj.html_code[:100] + "..."  # ✅ Show first 100 chars
#     html_preview.short_description = "HTML Code Preview"  # ✅ Column Name
#
#     list_display_links = ("title",)
#     search_fields = ("title", "repo_url", "difficulty")  # ✅ Added "difficulty" to search
#     list_filter = ("difficulty", "created_at")  # ✅ Added "difficulty" to filters
#
#
# @admin.register(GitHubScraperGame)
# class GitHubScraperGameAdmin(admin.ModelAdmin):
#     list_display = ('name', 'description', 'created_at')
#     search_fields = ('name',)
#
# @admin.register(GitGameScore)
# class UserGameScoreAdmin(admin.ModelAdmin):
#     list_display = ('id', 'user_id', 'game_id', 'github_challenge_id', 'score', 'attempts', 'last_played')
#     search_fields = ('user__username', 'game__name')
#     list_filter = ('game', 'score')


from django.contrib import admin
from .models import GitHubChallenge, GitHubScraperGame, GitGameScore

@admin.register(GitHubChallenge)
class GitHubChallengeAdmin(admin.ModelAdmin):
    list_display = ("title", "difficulty", "repo_url", "file_url", "html_preview", "created_at")  # ✅ "difficulty" added

    def html_preview(self, obj):
        return obj.html_code[:100] + "..."  # ✅ Show first 100 chars
    html_preview.short_description = "HTML Code Preview"

    list_display_links = ("title",)
    search_fields = ("title", "repo_url", "difficulty")  # ✅ "difficulty" in search
    list_filter = ("difficulty", "created_at")  # ✅ Filter by difficulty

@admin.register(GitHubScraperGame)
class GitHubScraperGameAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at')
    search_fields = ('name',)

@admin.register(GitGameScore)
class UserGameScoreAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'game_id', 'github_challenge_id', 'score', 'critical_score', 'serious_score', 'moderate_score', 'minor_score', 'attempts', 'last_played')
    search_fields = ('user__username', 'game__name')
    list_filter = ('game', 'score', 'critical_score', 'serious_score', 'moderate_score', 'minor_score')  # ✅ Filter by severity scores
