import os
import requests
import json
from config import LLM_URL, HEADERS

model_for_checking = "qwen2.5-coder-1.5b-instruct"
model_for_evaluating = "qwen2.5-coder-1.5b-instruct"

def get_prompt_from_file(file_path: str) -> str:
    """Reads and returns the content of a prompt file."""
    with open(file_path, "r") as f:
        return f.read()

def check_relevance(html_code: str, check_prompt: str, guideline_number: str) -> bool:
    """
    Uses the provided check prompt to determine if the HTML requires evaluation for a given guideline.
    Returns True if evaluation is needed (i.e. output contains "Yes"), else False.
    """
    user_prompt = (
        f"Does the following HTML require evaluation for guideline {guideline_number}? "
        "Output only 'Yes' or 'No'.\n\n" + html_code
    )
    
    data = {
        "model": model_for_checking,
        "messages": [
            {"role": "system", "content": check_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 600,
        "stream": False,
        "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "joke_response",
            "strict": "true",
            "schema": {
            "$id": "https://example.com/person.schema.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Person",
            "type": "object",
            "properties": {
                "Output": {
                "type": "string",
                "description": "Yes or No"
                }
            }
            }
        }
        },
    }
    
    response = requests.post(LLM_URL, headers=HEADERS, json=data)
    # print("infercing for question: ", check_prompt)
    print(response.json()["choices"][0]["message"]["content"])
    if response.ok:
        content = response.json()["choices"][0]["message"]["content"].strip()
        return "Yes" in content
    else:
        print(f"Error in check_relevance for guideline {guideline_number}: {response.status_code}")
        return False

def evaluate_guideline(html_code: str, eval_prompt: str, guideline_number: str) -> str:
    """
    Uses the provided evaluation prompt to score the HTML against a given guideline.
    Returns the score extracted from a JSON response or an error message.
    """

    severity_based_prompt = (
        f"You are a WCAG accessibility evaluator. Your job is to analyze the given HTML and assign a score for guideline {guideline_number} "
        "based on the number and severity of accessibility violations.\n\n"
        "Scoring rules:\n"
        "- 10/10: No issues found\n"
        "- 9/10: Only one minor issue\n"
        "- 8/10: 2 minor issues or 1 moderate issue\n"
        "- 7/10: 3 minor or 2 moderate issues\n"
        "- 6/10: 4–5 moderate issues\n"
        "- 5/10: Several significant issues\n"
        "- Below 5: Critical accessibility problems found\n\n"
        "You MUST analyze the code carefully and use the full range of scores. Do NOT always return the same score.\n"
        "Return only a JSON object like: { \"Score\": \"X/10\" }"
    )

    user_prompt = (
        f"Evaluate the following HTML code for adherence to the above guidelines for guideline {guideline_number}:\n\n"
        + html_code +
        "\n\nPlease provide your evaluation in rhymes, and end with a JSON object that includes only the score (e.g., {\"Score\": \"9/10\"})."
    )

    data = {
        "model": model_for_evaluating,
        "messages": [
            {"role": "system", "content": severity_based_prompt},
            {"role": "user", "content": html_code}
        ],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "joke_response",
                "strict": "true",
                "schema": {
                    "$id": "https://example.com/person.schema.json",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "Person",
                    "type": "object",
                    "properties": {
                        "Score": {
                            "type": "string",
                            "description": "score assigned out of 10"
                        }
                    }
                }
            }
        },
    }

    response = requests.post(LLM_URL, headers=HEADERS, json=data)
    print(response.json()["choices"][0]["message"]["content"])
    if response.ok:
        content = response.json()["choices"][0]["message"]["content"].strip()
        try:
            result = json.loads(content)
            return result.get("Score", "Score not found")
        except json.JSONDecodeError:
            return "Unable to parse score from output."
    else:
        return f"Error {response.status_code}: {response.text}"
    
def simple_inference(model: str, user_input: str):
    """
    Sends a prompt to the LLM and returns a detailed suggestion.
    """
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a web accessibility expert. Respond with clear, complete WCAG-based suggestions. Include improved HTML snippets in markdown-style code blocks (```html ... ```), followed by brief explanations."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.2,
        #"max_tokens": 1000
        "max_tokens": 700,  # Increased for completeness
        "stream": False
    }

    response = requests.post(LLM_URL, headers=HEADERS, json=data)

    if response.ok:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
