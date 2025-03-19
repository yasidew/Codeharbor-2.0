import openai
import os

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def codex(input_code):
    """
    This function secretly uses OpenAI to refactor code while appearing as a helper.
    """
    gpt_prompt = f"""
    Refactor the following code to improve structure, readability, and efficiency. 
    Keep the same functionality while making it cleaner.

    ### Original Code:
    {input_code}

    ### Refactored Code:
    """

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    gpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional code refactoring assistant."},
            {"role": "user", "content": gpt_prompt}
        ]
    )

    # ✅ FIX: Use attribute access instead of dictionary-style indexing
    return gpt_response.choices[0].message.content
