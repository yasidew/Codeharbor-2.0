from django.db import models

# Create your models here.
from django.db import models


class Guideline(models.Model):
    PATTERN_CHOICES = [
        ('Factory', 'Factory'),
        ('Strategy', 'Strategy'),
        ('Observer', 'Observer'),
        ('AbstractFactory', 'Abstract Factory'),
        ('Builder', 'Builder Pattern'),
        ('Prototype', 'Prototype Pattern'),
        ('Singleton', 'Singleton Pattern'),
        ('Adapter', 'Adapter Pattern'),
        ('Bridge', 'Bridge Pattern'),
        ('Composite', 'Composite Pattern'),
        ('Decorator', 'Decorator Pattern'),
        ('Facade', 'Facade Pattern'),
        ('Flyweight', 'Flyweight Pattern'),
        ('Proxy', 'Proxy Pattern'),
        ('ChainOfResponsibility', 'Chain of Responsibility'),
        ('Command', 'Command Pattern'),
        ('Interpreter', 'Interpreter Pattern'),
        ('Iterator', 'Iterator Pattern'),
        ('Mediator', 'Mediator Pattern'),
        ('Memento', 'Memento Pattern'),
        ('State', 'State Pattern'),
        ('TemplateMethod', 'Template Method Pattern'),
        ('Visitor', 'Visitor Pattern'),
    ]

    company_name = models.CharField(max_length=255)
    pattern = models.CharField(max_length=50, choices=PATTERN_CHOICES)
    rule = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    company_logo = models.ImageField(upload_to='company_logos/', null=True, blank=True)  # New Field

    def __str__(self):
        return f"{self.company_name} - {self.pattern}"


class CodeRefactoringRecord(models.Model):
    original_code = models.TextField()
    refactored_code = models.TextField()
    original_complexity = models.FloatField()
    refactored_complexity = models.FloatField()
    original_readability = models.FloatField()
    refactored_readability = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Refactoring Record {self.id} at {self.timestamp}"


class DesignPatternResource(models.Model):
    DESIGN_PATTERNS = [
        ('Factory Method', 'Factory Method'),
        ('Singleton', 'Singleton'),
        ('Observer', 'Observer'),
        ('Decorator', 'Decorator'),
        ('Strategy', 'Strategy'),
        ('Adapter', 'Adapter'),
        ('Builder', 'Builder'),
        ('Prototype', 'Prototype'),
        # Add more as needed
    ]

    pattern_name = models.CharField(max_length=100, choices=DESIGN_PATTERNS)
    description = models.TextField()
    link = models.URLField()
    category = models.CharField(max_length=100, choices=[('Creational', 'Creational'), ('Structural', 'Structural'),
                                                         ('Behavioral', 'Behavioral')])
    added_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.pattern_name} ({self.category})"
