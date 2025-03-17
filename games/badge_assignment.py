from django.db.models import Sum, F
from django.db import transaction

from games.models import Badge, UserBadge, GitGameScore


def award_badges(user):
    """Assign multiple badges based on the total critical score."""

    # Calculate the sum of critical scores
    total_critical_score = (
        GitGameScore.objects.filter(user=user)
        .aggregate(total_score=Sum(F('critical_score')))['total_score']
    )

    # Ensure it's not None
    if total_critical_score is None:
        total_critical_score = 0

    print(f"User: {user.username} | Total Critical Score: {total_critical_score}")  # Debugging Output

    # Adjusted badge criteria
    badge_criteria = [
        ("Platinum", 0),
        ("Gold", 10),
        ("Silver", 20),
        ("Bronze", 40),
        ("Participant", 100),
    ]

    with transaction.atomic():
        for badge_name, max_critical_score in badge_criteria:
            if total_critical_score <= max_critical_score:
                badge, _ = Badge.objects.get_or_create(
                    name=badge_name,
                    defaults={"description": f"Awarded for having ≤ {max_critical_score} total critical issues."}
                )

                # Ensure no duplicate badges
                if not UserBadge.objects.filter(user=user, badge=badge).exists():
                    UserBadge.objects.create(user=user, badge=badge)