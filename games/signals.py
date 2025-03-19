from django.db.models.signals import post_save
from django.dispatch import receiver

from games.badge_assignment import award_badges
from games.models import GitGameScore


@receiver(post_save, sender=GitGameScore)
def assign_badges(sender, instance, **kwargs):
    """Automatically assigns badges when a score is updated."""
    award_badges(instance.user)
