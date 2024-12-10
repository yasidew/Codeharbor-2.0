# from django.db import models
#
# # Create your models here.
# from django.db import models
# from django.contrib.auth.models import User
#
# class AccessibilityScore(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     score = models.FloatField()
#     date = models.DateTimeField(auto_now_add=True)
#     html_file_name = models.CharField(max_length=255, null=True, blank=True)  # Optional: Store the file name for reference
#
#     def __str__(self):
#         return f"Accessibility Score for {self.user.username} - {self.score}"
