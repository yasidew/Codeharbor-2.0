import requests

# Define the URL of the Flask backend
url = "http://127.0.0.1:8000/api/evaluate/"

# Define the HTML code and selected guidelines
html_code = """
<div>
    <button>Click me</button>
    <img src="image.jpg" alt="Description of image">
</div>
"""
guidelines = ["1", "2", "3"]  # Replace with the guidelines you want to evaluate

# Create the JSON payload
payload = {
    "html_code": html_code,
    "guidelines": guidelines
}

# Send the POST request
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes
    result = response.json()  # Parse the JSON response
    print("Response from server:")
    print(result)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")