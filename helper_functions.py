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
        "temperature": 0.5,
        "max_tokens": 10,
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
    user_prompt = (
        f"Evaluate the following HTML code for adherence to the above guidelines for guideline {guideline_number}:\n\n"
        + html_code +
        "\n\nPlease provide your evaluation in rhymes, and end with a JSON object that includes only the score (e.g., {\"Score\": \"9/10\"})."
    )
    
    data = {
        "model": model_for_evaluating,
        "messages": [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 15,
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
    This function sends a user input to the model and prints the response.
    """
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    response = requests.post(LLM_URL, headers=HEADERS, json=data)

    if response.ok:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
