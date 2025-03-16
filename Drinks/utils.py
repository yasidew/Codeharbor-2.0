import requests

# ✅ Calculate Lines of Code (LOC)
def calculate_loc(code):
    return len(code.splitlines())

# ✅ Calculate Readability (Simple Score)
def calculate_readability(code):
    return max(1, 100 - len(code) // 10)  # Higher is better

# ✅ Extract Code from GitHub Repository
def extract_code_from_github(repo_url):
    try:
        raw_url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        response = requests.get(raw_url)
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None