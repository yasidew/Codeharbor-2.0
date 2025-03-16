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
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES)  # ✅ Difficulty Level
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
    """Model to store user progress, scores, challenge ID, and categorized violation scores."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to Django's built-in User model
    game = models.ForeignKey(GitHubScraperGame, on_delete=models.CASCADE)
    github_challenge = models.ForeignKey(GitHubChallenge, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(default=0.0)  # ✅ Overall Score
    attempts = models.IntegerField(default=0)
    last_played = models.DateTimeField(auto_now=True)

    # ✅ Separate fields for different severity scores
    critical_score = models.FloatField(default=0.0)
    serious_score = models.FloatField(default=0.0)
    moderate_score = models.FloatField(default=0.0)
    minor_score = models.FloatField(default=0.0)

    def __str__(self):
        return (
            f"User ID: {self.user.id} | Game ID: {self.game.id} | Challenge ID: {self.github_challenge.id if self.github_challenge else 'None'} | "
            f"Score: {self.score} | Critical: {self.critical_score} | Serious: {self.serious_score} | Moderate: {self.moderate_score} | Minor: {self.minor_score}")
