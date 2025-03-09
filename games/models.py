
# from django.contrib.auth.models import User
# from django.db import models
#
# DIFFICULTY_CHOICES = [
#     ('easy', 'Easy'),
#     ('medium', 'Medium'),
#     ('hard', 'Hard'),
# ]
#
# class GitHubChallenge(models.Model):
#     title = models.CharField(max_length=255)
#     repo_url = models.URLField()
#     file_url = models.URLField()
#     html_code = models.TextField()
#     difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)  # ✅ Added difficulty
#     created_at = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return f"{self.title} ({self.difficulty})"
#
#
# class GitHubScraperGame(models.Model):
#     """Model for storing details about the accessibility checking game."""
#     name = models.CharField(max_length=255, unique=True)
#     description = models.TextField()
#     created_at = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return self.name
#
# class GitGameScore(models.Model):
#     """Model to store the user's progress, scores, and game participation."""
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to Django's built-in User model
#     game = models.ForeignKey(GitHubScraperGame, on_delete=models.CASCADE)
#     score = models.FloatField(default=0.0)
#     attempts = models.IntegerField(default=0)
#     last_played = models.DateTimeField(auto_now=True)
#
#     def __str__(self):
#         return f"{self.user.username} - {self.game.name} - Score: {self.score}"


from django.contrib.auth.models import User
from django.db import models

DIFFICULTY_CHOICES = [
    ('easy', 'Easy'),
    ('medium', 'Medium'),
    ('hard', 'Hard'),
]

class GitHubChallenge(models.Model):
    """Model for storing GitHub challenges with difficulty levels."""
    title = models.CharField(max_length=255)
    repo_url = models.URLField()
    file_url = models.URLField()
    html_code = models.TextField()
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)  # ✅ Added difficulty
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.difficulty})"

class GitHubScraperGame(models.Model):
    """Model for storing details about the accessibility checking game."""
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class GitGameScore(models.Model):
    """Model to store the user's progress, scores, game participation, and challenge ID."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to Django's built-in User model
    game = models.ForeignKey(GitHubScraperGame, on_delete=models.CASCADE)
    github_challenge = models.ForeignKey(GitHubChallenge, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(default=0.0)
    attempts = models.IntegerField(default=0)
    last_played = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.game.name} - {self.github_challenge.title} - Score: {self.score}"
