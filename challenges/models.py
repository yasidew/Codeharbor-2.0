from django.contrib.auth.models import User
from django.db import models

from django.db import models
from django.contrib.auth.models import User

class Challenges(models.Model):
    CATEGORY_CHOICES = [
        ('storytelling', 'Storytelling'),
        ('debugging', 'Debugging'),
        ('quiz', 'Quiz'),
        ('design', 'Design'),
        ('performance', 'Performance'),
    ]
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    template = models.TextField()
    difficulty = models.CharField(
        max_length=20,
        choices=[('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')]
    )
    month = models.IntegerField()  # Month number (1-12)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.category} - {self.difficulty} - Month {self.month}"





class Submission(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    challenge = models.ForeignKey(Challenges, on_delete=models.CASCADE)
    code = models.TextField()  # Store the submitted code
    score = models.IntegerField(null=True, blank=True)  # Accessibility score
    feedback = models.TextField(null=True, blank=True)  # Feedback from the checker
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.challenge} - {self.score}"



