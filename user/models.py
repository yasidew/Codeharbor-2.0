from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from datetime import timedelta, date


#new#
class UserProfile(models.Model):
    """Stores user-specific data like avatar, badges, and additional info."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to="media/avatars/", default="media/avatars/default.png")
    bio = models.TextField(blank=True, null=True)
    badges = models.JSONField(default=list)  # Stores earned badges as a list
    current_streak = models.IntegerField(default=0)
    longest_streak = models.IntegerField(default=0)

    def __str__(self):
        return self.user.username
