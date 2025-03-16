import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the trained model
MODEL_PATH = os.path.join(BASE_DIR, "code-refactoring", "singleton_model")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def format_code_with_model(code):
    """
    Function to format code using the trained model.
    """
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    formatted_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return formatted_code


def refactor_code(code):
    """
    Uses OpenAI to refactor the given code and return a cleaned-up version.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an AI that refactors code for better readability and efficiency."},
                {"role": "user",
                 "content": f"Refactor the following code while maintaining its functionality:\n\n{code}"}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error in refactoring: {str(e)}"


def analyze_code(code):
    """
    Uses OpenAI to analyze the given code and provide insights on its complexity, maintainability, and readability.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an AI that analyzes code for complexity, maintainability, and readability."},
                {"role": "user",
                 "content": f"Analyze the following code and provide insights on its complexity, maintainability, and readability:\n\n{code}"}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error in analysis: {str(e)}"

# import os
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import openai
#
# # Base directory for the project
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# # Path to the trained model
# MODEL_PATH = os.path.join(BASE_DIR, "code-refactoring", "singleton_model")
#
# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
#
# def format_code_with_model(code):
#     """
#     Function to format code using the trained model.
#     """
#     inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
#     outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
#     formatted_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return formatted_code
#
#
# def refactor_code(code):
#     """
#     Uses OpenAI to refactor the given code and return a cleaned-up version.
#     """
#     try:
#         client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system",
#                  "content": "You are an AI that refactors code for better readability and efficiency."},
#                 {"role": "user",
#                  "content": f"Refactor the following code while maintaining its functionality:\n\n{code}"}
#             ],
#             temperature=0.5
#         )
#
#         return response.choices[0].message.content.strip()
#
#     except Exception as e:
#         return f"Error in refactoring: {str(e)}"
#
#
# def analyze_code(code):
#     """
#     Uses OpenAI to analyze the given code and provide insights on its complexity, maintainability, and readability.
#     """
#     try:
#         client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system",
#                  "content": "You are an AI that analyzes code for complexity, maintainability, and readability."},
#                 {"role": "user",
#                  "content": f"Analyze the following code and provide insights on its complexity, maintainability, and readability:\n\n{code}"}
#             ],
#             temperature=0.5
#         )
#
#         return response.choices[0].message.content.strip()
#
#     except Exception as e:
#         return f"Error in analysis: {str(e)}"
