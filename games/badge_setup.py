from games.models import Badge

badge_data = [
    {"name": "Platinum", "description": "Awarded for having ≤ 0 total critical issues."},
    {"name": "Gold", "description": "Awarded for having ≤ 5 total critical issues."},
    {"name": "Silver", "description": "Awarded for having ≤ 10 total critical issues."},
    {"name": "Bronze", "description": "Awarded for having ≤ 20 total critical issues."},
]

for data in badge_data:
    Badge.objects.get_or_create(name=data["name"], defaults={"description": data["description"]})
