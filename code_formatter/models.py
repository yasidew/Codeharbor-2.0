from django.db import models

# Create your models here.
from django.db import models

class Guideline(models.Model):
    company_name = models.CharField(max_length=255)
    pattern = models.CharField(max_length=50, choices=[('Factory', 'Factory'), ('Strategy', 'Strategy'), ('Observer', 'Observer')])
    rule = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.company_name} - {self.pattern}"
