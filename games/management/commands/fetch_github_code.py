from django.core.management.base import BaseCommand
from games.github_scraper import fetch_bad_code_from_github  # Import the scraper function

class Command(BaseCommand):
    help = "Fetches bad accessibility code from GitHub"

    def handle(self, *args, **kwargs):
        fetch_bad_code_from_github()
        self.stdout.write(self.style.SUCCESS("✅ GitHub accessibility issues fetched successfully!"))
