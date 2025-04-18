# import requests
# import subprocess
# import os
# from dotenv import load_dotenv  # Import dotenv to load environment variables
# from .models import GitHubChallenge  # Import the model from games.models
#
# # Load environment variables from .env file
# load_dotenv()
# import os
# from .models import GitHubChallenge  # Import the model from games.models
#
# # GitHub API URL to search for "accessibility" in HTML files
# GITHUB_API_URL = "https://api.github.com/search/code?q=accessibility+extension:html"
#
# # GitHub Token (Move to environment variables for security)
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
#
# AXE_JS_SCRIPT = "axe-check-new.js"  # Path to your axe-check.js file
#
#
# def run_axe_analysis(html_code):
#     """Runs axe-check.js and returns the number of accessibility violations."""
#     try:
#         # Run the axe-check.js script and pass the HTML content via stdin
#         result = subprocess.run(
#             ["node", AXE_JS_SCRIPT],
#             input=html_code,
#             text=True,
#             capture_output=True,
#             shell=True
#         )
#
#         if result.returncode == 0:
#             issue_count = int(result.stdout.strip())  # Convert output to integer
#             return issue_count
#         else:
#             print("❌ Axe analysis failed:", result.stderr)
#             return 0  # Return 0 in case of failure
#     except Exception as e:
#         print("❌ Axe-core execution error:", e)
#         return 0  # Return 0 in case of failure
#
#
# def determine_difficulty(issue_count):
#     """Determines difficulty level based on issue count."""
#     if issue_count <= 2:
#         return "easy"  # ✅ Lowercase
#     elif issue_count <= 15:
#         return "medium"  # ✅ Lowercase
#     else:
#         return "hard"  # ✅ Lowercase
#
#
#
# def fetch_bad_code_from_github():
#     headers = {
#         "Accept": "application/vnd.github.v3+json",
#         "Authorization": f"token {GITHUB_TOKEN}"  # Authenticate with token
#     }
#
#     response = requests.get(GITHUB_API_URL, headers=headers)
#
#     if response.status_code == 200:
#         data = response.json()
#         print(f"✅ Found {len(data['items'])} code snippets with accessibility issues.\n")
#
#         for item in data["items"][:5]:  # Get first 5 results
#             title = item["name"]
#             repo_url = item["repository"]["html_url"]
#             file_url = item["html_url"]
#
#             # Fetch the raw file content
#             raw_url = file_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
#             raw_response = requests.get(raw_url)
#
#             if raw_response.status_code == 200:
#                 html_code = raw_response.text[:1000]  # Store only first 1000 chars
#
#                 # Run axe-core analysis to determine accessibility issues
#                 issue_count = run_axe_analysis(html_code)
#                 difficulty = determine_difficulty(issue_count)
#
#                 # Save to the database
#                 GitHubChallenge.objects.create(
#                     title=title,
#                     repo_url=repo_url,
#                     file_url=file_url,
#                     html_code=html_code,
#                     difficulty=difficulty  # Store difficulty level
#                 )
#
#                 print(f"✅ Saved Challenge: {title} (Difficulty: {difficulty})")
#
#     else:
#         print(f"❌ GitHub API Error: {response.status_code}")
#
#
# # Run the function (Optional: Only when manually executed)
# if __name__ == "__main__":
#     fetch_bad_code_from_github()


import requests
import subprocess
import os
import random
from dotenv import load_dotenv  # Import dotenv to load environment variables
from .models import GitHubChallenge  # Import the model from games.models

# Load environment variables from .env file
load_dotenv()

# GitHub Token (Move to environment variables for security)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

AXE_JS_SCRIPT = "axe-check-new.js"  # Path to your axe-check.js file

# Accessibility-related search terms to vary the results
SEARCH_TERMS = ["aria-label", "alt attribute", "screen reader", "color contrast", "form accessibility"]

# Function to get a random page number (GitHub supports up to 1000 results, 30 per page)
def get_random_page():
    return random.randint(1, 10)  # Randomly pick a page (adjust as needed)


def run_axe_analysis(html_code):
    """Runs axe-check.js and returns the number of accessibility violations."""
    try:
        result = subprocess.run(
            ["node", AXE_JS_SCRIPT],
            input=html_code.encode("utf-8"),  # Encode input in UTF-8
            text=False,  # Disable automatic text handling
            capture_output=True,
            shell=True
        )

        if result.returncode == 0:
            output = result.stdout.decode("utf-8").strip()  # Decode output in UTF-8
            return int(output) if output.isdigit() else 0  # Convert to integer
        else:
            print("❌ Axe analysis failed:", result.stderr.decode("utf-8", errors="ignore"))
            return 0  # Return 0 in case of failure
    except Exception as e:
        print("❌ Axe-core execution error:", e)
        return 0  # Return 0 in case of failure



def determine_difficulty(issue_count):
    """Determines difficulty level based on issue count."""
    if issue_count <= 2:
        return "easy"
    elif issue_count <= 15:
        return "medium"
    else:
        return "hard"


def fetch_bad_code_from_github():
    """Fetches random accessibility-related code snippets from GitHub."""
    random_search_term = random.choice(SEARCH_TERMS)
    page = get_random_page()

    # GitHub API URL with dynamic search term and pagination
    github_api_url = f"https://api.github.com/search/code?q={random_search_term}+extension:html&page={page}&per_page=5"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    response = requests.get(github_api_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])

        if not items:
            print("⚠️ No results found on this page. Trying again...")
            return fetch_bad_code_from_github()  # Recursively call to try another page

        print(f"✅ Found {len(items)} code snippets with accessibility issues.")

        # Shuffle the results to add randomness
        random.shuffle(items)

        # Get all stored challenge URLs to avoid duplicates
        existing_urls = set(GitHubChallenge.objects.values_list("file_url", flat=True))

        for item in items:
            title = item["name"]
            repo_url = item["repository"]["html_url"]
            file_url = item["html_url"]

            # Ensure uniqueness
            if file_url in existing_urls:
                print(f"⚠️ Skipping duplicate challenge: {title}")
                continue

            # Fetch raw file content
            raw_url = file_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            raw_response = requests.get(raw_url)

            if raw_response.status_code == 200:
                html_code = raw_response.text[:1000]  # Store only first 1000 chars

                # Run axe-core analysis to determine accessibility issues
                issue_count = run_axe_analysis(html_code)
                difficulty = determine_difficulty(issue_count)

                # Save to the database
                GitHubChallenge.objects.create(
                    title=title,
                    repo_url=repo_url,
                    file_url=file_url,
                    html_code=html_code,
                    difficulty=difficulty
                )

                print(f"✅ Saved Challenge: {title} (Difficulty: {difficulty})")
            else:
                print(f"❌ Failed to fetch raw file for {title}")

    else:
        print(f"❌ GitHub API Error: {response.status_code}, Message: {response.text}")


# Run the function when manually executed
if __name__ == "__main__":
    fetch_bad_code_from_github()
