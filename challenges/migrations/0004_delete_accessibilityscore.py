# Generated by Django 4.2.16 on 2024-12-02 18:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('challenges', '0003_accessibilityscore'),
    ]

    operations = [
        migrations.DeleteModel(
            name='AccessibilityScore',
        ),
    ]