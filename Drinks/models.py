from django.db import models

class Drink(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=500)

    def __str__(self):
        return self.name + ' ' + self.description


class JavaFile(models.Model):
    """ Stores only Java file name and total WCC value """
    filename = models.CharField(max_length=255, unique=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_wcc = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.filename} - WCC: {self.total_wcc}"
