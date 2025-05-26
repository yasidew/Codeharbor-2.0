import requests

# Define the URL
url = "http://127.0.0.1:8000/api/evaluate/"

# Define the HTML code and selected guidelines
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>WCAG Test Case</title>
</head>
<body>

  <h1>Accessibility Compliance Test</h1>

  <!-- Good structure for Guideline 11 -->
  <h2>Contact Form</h2>
  <form>
    <label for="username">Username:</label>
    <input type="text" id="username" name="username">
    <br><br>
    <label for="email">Email:</label>
    <input type="email" id="email" name="email">
    <button type="submit">Submit</button>
  </form>

  <!-- Guideline 1 violation: image missing alt text -->
  <h2>Image Section</h2>
  <img src="important-diagram.png">

  <!-- Guideline 2 violation: video without caption or audio description -->
  <h2>Product Demo Video</h2>
  <video controls>
    <source src="demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

</body>
</html>

"""
guidelines = ["1", "2", "11"]  # Replace with the guidelines you want to evaluate

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