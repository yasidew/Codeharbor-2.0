# Generated by Django 4.2.19 on 2025-03-03 04:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('games', '0004_githubscrapergame_alter_githubchallenge_difficulty_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='gitgamescore',
            name='game',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='games.githubscrapergame'),
        ),
    ]
