from django.db import models

class Drink(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=500)

    def __str__(self):
        return self.name + ' ' + self.description

class JavaFile(models.Model):
    filename = models.CharField(max_length=255, unique=True)
    java_code = models.TextField()
    analysis_date = models.DateTimeField(auto_now_add=True)
    total_wcc = models.IntegerField(default=0)  # Store Weighted Code Complexity (WCC)

    def __str__(self):
        return f"{self.filename} - WCC: {self.total_wcc} - {self.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}"

class MethodComplexity(models.Model):
    java_file = models.ForeignKey(JavaFile, on_delete=models.CASCADE, related_name="methods")
    method_name = models.CharField(max_length=255)
    total_complexity = models.IntegerField(default=0)
    category = models.CharField(max_length=50, choices=[
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High')
    ], default='Low')

    # Additional complexity metrics
    size = models.IntegerField(default=0)
    control_structure_complexity = models.IntegerField(default=0)
    nesting_weight = models.IntegerField(default=0)
    inheritance_weight = models.IntegerField(default=0)
    compound_condition_weight = models.IntegerField(default=0)
    try_catch_weight = models.IntegerField(default=0)
    thread_weight = models.IntegerField(default=0)
    cbo_weight = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.method_name} ({self.category}) - {self.java_file.filename}"


class CSharpFile(models.Model):
    filename = models.CharField(max_length=255, unique=True)
    csharp_code = models.TextField()
    analysis_date = models.DateTimeField(auto_now_add=True)
    total_wcc = models.IntegerField(default=0)  # Weighted Code Complexity (WCC)

    def __str__(self):
        return f"{self.filename} - WCC: {self.total_wcc} - {self.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}"


class CSharpMethodComplexity(models.Model):
    csharp_file = models.ForeignKey(CSharpFile, on_delete=models.CASCADE, related_name="methods")
    method_name = models.CharField(max_length=255)
    total_complexity = models.IntegerField(default=0)
    size = models.IntegerField(default=0)
    control_structure_complexity = models.IntegerField(default=0)
    nesting_weight = models.IntegerField(default=0)
    inheritance_weight = models.IntegerField(default=0)
    compound_condition_weight = models.IntegerField(default=0)
    try_catch_weight = models.IntegerField(default=0)
    thread_weight = models.IntegerField(default=0)
    cbo_weight = models.IntegerField(default=0)
    category = models.CharField(max_length=50, choices=[
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High')
    ], default='Low')

    def __str__(self):
        return f"{self.method_name} ({self.category}) - {self.csharp_file.filename}"

