import subprocess
import json

def check_accessibility(html_code):
    try:
        # Save the user's HTML code to a temporary file
        with open("temp.html", "w") as file:
            file.write(html_code)

        # Run axe CLI (make sure to install `axe-cli` with npm)
        result = subprocess.run(["axe", "temp.html", "--json"], capture_output=True, text=True)

        # Check if the tool ran successfully
        if result.returncode != 0:
            return {"score": 0, "feedback": "Error analyzing the code."}

        # Parse the JSON output from axe
        output = json.loads(result.stdout)
        violations = len(output.get("violations", []))
        total_checks = output.get("passes", 0) + violations

        # Calculate score and feedback
        score = max(0, 100 - (violations / total_checks * 100))
        feedback = "\n".join(
            [violation.get("description", "") for violation in output["violations"]]
        )
        return {"score": score, "feedback": feedback}
    except Exception as e:
        return {"score": 0, "feedback": f"An error occurred: {str(e)}"}
