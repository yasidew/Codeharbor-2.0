import json
import os
from datetime import datetime

import pandas as pd
from datetime import datetime

from django.conf import settings
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from Drinks.models import Drink
from analysis.code_analyzer import CodeAnalyzer
from analysis.java_code_analyser import JavaCodeAnalyzer
from analysis.javascript_code_analyser import JavaScriptCodeAnalyzer
from analysis.php_code_analyser import PHPCodeAnalyzer
from analysis.python_code_analyser import PythonCodeAnalyzer
from code_analysis.models import CodeSnippet, Project
from .serializers import DrinkSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .complexity_calculator import calculate_code_complexity_multiple_files, ai_recommend_refactoring, \
    calculate_code_complexity_by_method
from .complexity_calculator import calculate_code_complexity_line_by_line
from .metrics import analyze_code_complexity, load_guidelines, count_lines_of_code, count_functions_and_length, \
    count_duplicate_code_percentage, calculate_comment_density, calculate_readability_score, calculate_complexity_score, \
    categorize_value

from prettytable import PrettyTable
import statsmodels.api as sm

from challenges.models import Challenges

def home(request):
    return render(request, 'home.html')

from transformers import T5ForConditionalGeneration, AutoTokenizer
from rest_framework.decorators import api_view
import concurrent.futures
from functools import lru_cache
from dotenv import load_dotenv
import openai
import base64
from django.http import JsonResponse
from django.shortcuts import render
import json
import requests
from openai import OpenAI
import os
import torch
import ast
from transformers import AutoTokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
from django.shortcuts import render
from rest_framework.decorators import api_view


from .utils import calculate_loc, calculate_readability, extract_code_from_github

load_dotenv()
# OpenAI API Client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize model and tokenizer once
MODEL_PATH = "./models/custom_seq2seq_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global instances for reuse
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Optimize PyTorch performance
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


FLASK_API_URL = "http://127.0.0.1:5000/predict"

def call_flask_model(snippet):
    """Calls Flask API and gets prediction from T5 model."""
    try:
        response = requests.post(FLASK_API_URL, json={"snippet": snippet}, timeout=15)
        if response.status_code == 200:
            return response.json().get("generated_text", "❌ No response from model")
        else:
            return f"❌ Flask API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Flask API Request Failed: {str(e)}"


def home(request):
    return render(request, 'home.html')


def refactor_view(request):
    # Your view logic here
    return render(request, 'refactor.html')


def upload_code(request):
    return render(request, 'upload_code.html')


def refactor_code(request):
    return render(request, 'refactor_code.html')


@api_view(['GET', 'POST'])
def python_code_analysis(request):
    recommendations = {"files": {}, "pasted_code": {}}  # Initialize as a dictionary
    summary = {
        "total_vulnerabilities": 0,
        "categories": {},
        "files_analyzed": 0,
    }
    code = ""

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = PythonCodeAnalyzer(code)
                pasted_results = analyzer.generate_recommendations()
                grouped = group_recommendations_by_line(pasted_results)
                recommendations["pasted_code"] = {
                    "code": code,
                    "results": grouped,
                }
                # Update summary
                for recs in grouped.values():
                    for rec in recs:
                        summary["total_vulnerabilities"] += 1
                        summary["categories"].setdefault(rec["rule"], 0)
                        summary["categories"][rec["rule"]] += 1
            except Exception as e:
                recommendations["pasted_code"] = {
                    "error": f"Error analyzing pasted code: {str(e)}"
                }

        # Handle uploaded files
        files = request.FILES.getlist('files')
        summary["files_analyzed"] = len(files)
        if files:
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = PythonCodeAnalyzer(content)
                    file_results = analyzer.generate_recommendations()
                    grouped = group_recommendations_by_line(file_results)
                    recommendations["files"][file.name] = {
                        "content": content,
                        "results": grouped,
                    }
                    # Update summary
                    for recs in grouped.values():
                        for rec in recs:
                            summary["total_vulnerabilities"] += 1
                            summary["categories"].setdefault(rec["rule"], 0)
                            summary["categories"][rec["rule"]] += 1
                except Exception as e:
                    recommendations["files"][file.name] = {
                        "error": f"Error analyzing file: {str(e)}"
                    }

        return render(
            request,
            'python_code_analysis.html',
            {'recommendations': recommendations, 'summary': summary, 'code': code}
        )

    return render(request, 'python_code_analysis.html', {'code': code})


def group_recommendations_by_line(recommendations):
    """
    Group recommendations by line number.
    """
    grouped = {}
    for rec in recommendations:
        line = rec.get("line", "unknown")
        grouped.setdefault(line, []).append({
            "rule": rec.get("rule"),
            "message": rec.get("message"),
        })
    return grouped


@api_view(['GET', 'POST'])
def java_code_analysis(request):
    recommendations = {"files": {}, "pasted_code": {}}  # Initialize as a dictionary
    summary = {
        "total_vulnerabilities": 0,
        "categories": {},
        "files_analyzed": 0,
    }
    code = ""

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = JavaCodeAnalyzer(code)
                pasted_results = analyzer.generate_recommendations()
                # Organize recommendations by line
                recommendations["pasted_code"] = {}
                for rec in pasted_results:
                    line = rec.get('line', 'unknown')
                    recommendations["pasted_code"].setdefault(line, []).append({
                        'rule': rec.get('rule'),
                        'message': rec.get('message'),
                    })
                # Update summary
                for rec in pasted_results:
                    summary["total_vulnerabilities"] += 1
                    rule = rec.get('rule')
                    if rule:
                        summary["categories"].setdefault(rule, 0)
                        summary["categories"][rule] += 1
            except Exception as e:
                recommendations["pasted_code"] = {
                    "error": f"Error analyzing pasted code: {str(e)}"
                }

        # Handle uploaded files
        files = request.FILES.getlist('files')
        summary["files_analyzed"] = len(files)
        if files:
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = JavaCodeAnalyzer(content)
                    file_results = analyzer.generate_recommendations()
                    recommendations["files"][file.name] = {}
                    for rec in file_results:
                        line = rec.get('line', 'unknown')
                        recommendations["files"][file.name].setdefault(line, []).append({
                            'rule': rec.get('rule'),
                            'message': rec.get('message'),
                        })
                    # Update summary
                    for rec in file_results:
                        summary["total_vulnerabilities"] += 1
                        rule = rec.get('rule')
                        if rule:
                            summary["categories"].setdefault(rule, 0)
                            summary["categories"][rule] += 1
                except Exception as e:
                    recommendations["files"][file.name] = {
                        "error": f"Error analyzing file {file.name}: {str(e)}"
                    }

        return render(
            request,
            'java_code_analysis.html',
            {'recommendations': recommendations, 'summary': summary, 'code': code}
        )

    return render(request, 'java_code_analysis.html', {'code': code})


@api_view(['GET', 'POST'])
def js_code_analyser(request):
    recommendations = {"files": {}, "pasted_code": {}}  # Initialize recommendations
    summary = {
        "total_vulnerabilities": 0,
        "categories": {},
        "files_analyzed": 0,
    }
    code = ""

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = JavaScriptCodeAnalyzer(code)
                pasted_results = analyzer.generate_recommendations()
                recommendations["pasted_code"] = {}
                for rec in pasted_results:
                    line = rec.get('line', 'unknown')
                    recommendations["pasted_code"].setdefault(line, []).append({
                        'rule': rec.get('rule'),
                        'message': rec.get('message'),
                    })
                # Update summary
                for rec in pasted_results:
                    summary["total_vulnerabilities"] += 1
                    rule = rec.get('rule')
                    if rule:
                        summary["categories"].setdefault(rule, 0)
                        summary["categories"][rule] += 1
            except Exception as e:
                recommendations["pasted_code"] = {
                    "error": f"Error analyzing pasted code: {str(e)}"
                }

        # Handle uploaded files
        files = request.FILES.getlist('files')
        summary["files_analyzed"] = len(files)
        if files:
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = JavaScriptCodeAnalyzer(content)
                    file_results = analyzer.generate_recommendations()
                    recommendations["files"][file.name] = {}
                    for rec in file_results:
                        line = rec.get('line', 'unknown')
                        recommendations["files"][file.name].setdefault(line, []).append({
                            'rule': rec.get('rule'),
                            'message': rec.get('message'),
                        })
                    # Update summary
                    for rec in file_results:
                        summary["total_vulnerabilities"] += 1
                        rule = rec.get('rule')
                        if rule:
                            summary["categories"].setdefault(rule, 0)
                            summary["categories"][rule] += 1
                except Exception as e:
                    recommendations["files"][file.name] = {
                        "error": f"Error analyzing file {file.name}: {str(e)}"
                    }

        return render(
            request,
            'js_code_analyser.html',
            {'recommendations': recommendations, 'summary': summary, 'code': code}
        )

    return render(request, 'js_code_analyser.html', {'code': code})


@api_view(['GET', 'POST'])
def php_code_analyser(request):
    recommendations = {"pasted_code": {}, "files": {}}
    code = ""

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = PHPCodeAnalyzer(code)
                pasted_results = analyzer.generate_recommendations()
                for rec in pasted_results:
                    line = rec.get('line', 'unknown')
                    recommendations["pasted_code"].setdefault(line, []).append({
                        'rule': rec.get('rule'),
                        'message': rec.get('message'),
                    })
            except Exception as e:
                recommendations["pasted_code"]["error"] = f"Error analyzing pasted code: {str(e)}"

        # Handle uploaded files
        files = request.FILES.getlist('files')
        if files:
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = PHPCodeAnalyzer(content)
                    file_results = analyzer.generate_recommendations()
                    recommendations["files"][file.name] = {}
                    for rec in file_results:
                        line = rec.get('line', 'unknown')
                        recommendations["files"][file.name].setdefault(line, []).append({
                            'rule': rec.get('rule'),
                            'message': rec.get('message'),
                        })
                except Exception as e:
                    recommendations["files"][file.name] = {
                        "error": f"Error analyzing file {file.name}: {str(e)}"
                    }

        return render(
            request,
            'php_code_analyser.html',
            {'recommendations': recommendations, 'code': code}
        )

    return render(request, 'php_code_analyser.html', {'code': code})


def normalize_and_validate_indentation(code_snippet):
    """
    Normalize and validate indentation in the code snippet to prevent parsing errors.
    """
    lines = code_snippet.splitlines()
    corrected_lines = []
    indent_stack = [0]  # Track indentation levels

    for index, line in enumerate(lines):
        stripped_line = line.strip()

        # Skip empty lines
        if not stripped_line:
            corrected_lines.append("")
            continue

        # Calculate current indentation level
        current_indent = len(line) - len(line.lstrip())

        # Check for invalid indentation
        if current_indent > indent_stack[-1]:
            # Ensure a block follows structures like def, if, for, etc.
            if corrected_lines and corrected_lines[-1].rstrip().endswith((':',)):
                indent_stack.append(current_indent)
            else:
                # Force valid indentation
                current_indent = indent_stack[-1]

        elif current_indent < indent_stack[-1]:
            while indent_stack and current_indent < indent_stack[-1]:
                indent_stack.pop()

        # Fix indentation if mismatched
        if current_indent != indent_stack[-1]:
            corrected_line = " " * indent_stack[-1] + stripped_line
        else:
            corrected_line = line

        corrected_lines.append(corrected_line)

    # Ensure every block with a colon has an indented body
    normalized_code = "\n".join(corrected_lines)
    fixed_code = ensure_blocks_have_bodies(normalized_code)

    return fixed_code


def ensure_blocks_have_bodies(code_snippet):
    """
    Add placeholder 'pass' statements to ensure blocks have valid bodies.
    """
    lines = code_snippet.splitlines()
    corrected_lines = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and (index + 1 >= len(lines) or not lines[index + 1].strip()):
            corrected_lines.append(line)
            corrected_lines.append("    pass  # Placeholder for empty block")
        else:
            corrected_lines.append(line)

    return "\n".join(corrected_lines)


def split_code_snippets(code_snippet):
    """
    Split the input code snippet into individual Python functions or top-level blocks.
    """
    try:
        # Normalize and validate indentation
        normalized_code = normalize_and_validate_indentation(code_snippet)

        # Parse the normalized code into an Abstract Syntax Tree (AST)
        tree = ast.parse(normalized_code)
        snippets = []

        # Extract top-level nodes
        for node in tree.body:
            # Extract source code for each node
            if hasattr(ast, "get_source_segment"):
                snippet = ast.get_source_segment(normalized_code, node)
                print(f"Snippet: {snippet}")
            else:
                # Fallback: Use line numbers if available
                start_line = getattr(node, "lineno", None)
                end_line = getattr(node, "end_lineno", None)

                if start_line and end_line:
                    snippet_lines = normalized_code.splitlines()[start_line - 1:end_line]
                    snippet = "\n".join(snippet_lines)
                else:
                    # Fallback for cases where neither method works
                    snippet = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)

            snippets.append(snippet)

        return snippets
    except SyntaxError as e:
        print(f"Error parsing code snippets: {e}")
        return []  # Return an empty list if parsing fails


# def split_code_snippets(code, language="python"):
#     """
#     Splits code into logical blocks for analysis.
#
#     - Uses AST-based parsing for Python.
#     - Uses regex-based parsing for other languages (Java, JS, C++, etc.).
#     """
#
#     if language.lower() == "python":
#         try:
#             tree = ast.parse(code)
#             snippets = []
#
#             for node in tree.body:
#                 # Extract source code for each node
#                 snippet = ast.get_source_segment(code, node) if hasattr(ast, "get_source_segment") else ast.unparse(node)
#                 snippets.append(snippet)
#
#             return snippets
#         except SyntaxError:
#             print("Syntax error while parsing Python code.")
#             return []
#
#     # Generic Regex-based splitting for other languages
#     snippet_list = []
#     buffer = []
#     block_starts = re.compile(r"^\s*(def |class |if |elif |else:|try:|except |for |while |public |private |function |var )")
#
#     for line in code.split("\n"):
#         if block_starts.match(line) and buffer:
#             snippet_list.append("\n".join(buffer))
#             buffer = []
#
#         buffer.append(line)
#
#     if buffer:
#         snippet_list.append("\n".join(buffer))
#
#     return snippet_list
#############################################
@lru_cache(maxsize=1000)
def cached_prediction(snippet):
    """Generate a suggestion with caching for repeated snippets."""
    inputs = tokenizer(snippet, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def process_snippets_multithreaded(snippets):
    """Parallel processing of code snippets."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(cached_prediction, snippets))




# Ensure API Key is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure API Key is loaded
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.models.list()
    print("✅ API Key is working. Available models:", response)
except Exception as e:
    print("❌ Invalid API Key or Quota Issue:", str(e))



# def fetch_github_files(repo_url):
#     """
#     Fetches code files from a GitHub repository.
#     """
#     try:
#         repo_owner, repo_name = repo_url.replace("https://github.com/", "").split("/")[:2]
#         api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
#
#         headers = {"Authorization": f"token {GITHUB_ACCESS_TOKEN}"}
#         response = requests.get(api_url, headers=headers)
#
#         if response.status_code != 200:
#             return None, f"Failed to fetch repository: {response.json().get('message', 'Unknown error')}"
#
#         files = response.json()
#         code_files = []
#
#         allowed_extensions = {".py", ".js", ".java", ".cpp", ".cs", ".php"}
#         for file in files:
#             if any(file["name"].endswith(ext) for ext in allowed_extensions):
#                 file_response = requests.get(file["download_url"], headers=headers)
#                 if file_response.status_code == 200:
#                     code_files.append({"name": file["name"], "code": file_response.text})
#
#         return code_files, None
#
#     except Exception as e:
#         return None, f"Error fetching GitHub repository: {str(e)}"



def fetch_github_files(repo_url):
    """
    Fetches code files from a GitHub repository (supports both public & private).
    """
    try:
        # Extract repo details and branch
        repo_parts = repo_url.replace("https://github.com/", "").split("/")
        repo_owner, repo_name = repo_parts[0], repo_parts[1]
        branch = repo_parts[3] if len(repo_parts) > 3 and repo_parts[2] == "tree" else "main"

        # Fetch file list from the branch
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
        headers = {"Authorization": f"token {GITHUB_ACCESS_TOKEN}"}

        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            return None, f"Failed to fetch repository: {response.json().get('message', 'Unknown error')}"

        files = response.json().get("tree", [])
        code_files = []
        allowed_extensions = {".py", ".js", ".java", ".cpp", ".cs", ".php"}

        for file in files:
            if file["type"] == "blob" and any(file["path"].endswith(ext) for ext in allowed_extensions):
                file_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file['path']}?ref={branch}"
                file_response = requests.get(file_api_url, headers=headers)

                if file_response.status_code == 200:
                    file_content = file_response.json().get("content", "")
                    decoded_content = base64.b64decode(file_content).decode("utf-8")  # Decode base64 content

                    code_files.append({"name": file["path"], "code": decoded_content})

        return code_files, None

    except Exception as e:
        return None, f"Error fetching GitHub repository: {str(e)}"



def ai_code_analysis(snippet):
    """
    Uses OpenAI's GPT-4o to analyze a given code snippet and provide structured suggestions.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4o" for better results/ gpt-3.5-turbo
            messages=[
                {"role": "system",
                 "content": "You are an expert AI code reviewer. Follow this structure:\n"
                            "Issue Identified: Describe the issue concisely.\n"
                            "Why It's a Problem: Explain the consequences very shortly.\n"
                            "Recommended Fix: Provide a solution."},
                {"role": "user",
                 "content": f"Analyze the following Python code and provide structured feedback:\n{snippet}"}
            ],
            max_tokens=300,
            temperature=0.2  # Control randomness
        )

        # Extract AI response
        ai_response = response.choices[0].message.content

        # Format the response for HTML
        formatted_response = ai_response.replace("Issue Identified:", "<b>Issue Identified:</b>") \
            .replace("Why It's a Problem:", "<b>Why It's a Problem:</b>") \
            .replace("Recommended Fix:", "<b>Recommended Fix:</b>")

        return formatted_response  # ✅ Now formatted for HTML rendering
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"



def analyze_code_complexity(code):
    """Analyze code complexity using various metrics."""
    print("🔍 Entering analyze_code_complexity function...")  # ✅ Debug: Check if function is called
    guidelines = load_guidelines()

    loc, eloc = count_lines_of_code(code)
    print(f"📊 Lines of Code: {loc}, Effective LOC: {eloc}")  # ✅ Debug: Log LOC values

    num_functions, avg_function_length = count_functions_and_length(code)
    print(f"📊 Functions: {num_functions}, Avg Length: {avg_function_length}")  # ✅ Debug

    duplicate_percentage = count_duplicate_code_percentage(code)
    comment_density = calculate_comment_density(code)
    readability_score = calculate_readability_score(code)
    complexity_score = calculate_complexity_score(loc, num_functions, duplicate_percentage)

    result = {
        "lines_of_code": loc,
        "effective_lines_of_code": eloc,
        "num_functions": num_functions,
        "avg_function_length": avg_function_length,
        "duplicate_code_percentage": duplicate_percentage,
        "comment_density": comment_density,
        "readability_score": readability_score,
        "complexity_score": complexity_score,
        "rating": {
            "lines_of_code": categorize_value(loc, guidelines["lines_of_code"]),
            "comment_density": categorize_value(comment_density, guidelines["code_density"]),
            "function_length": categorize_value(avg_function_length, guidelines["function_length"]),
            "duplicate_code": categorize_value(duplicate_percentage, guidelines["duplicate_code"]),
            "num_functions": categorize_value(num_functions, guidelines["num_functions"]),
            "complexity_score": categorize_value(complexity_score, guidelines["complexity_score"]),
        }
    }

    print(f"✅ Complexity Analysis Output: {result}")  # ✅ Debug: Print output
    return result


def ai_generate_guideline(summary):
    """
    Uses OpenAI to generate a final coding guideline for the developer based on the summary report.
    This includes best practices, security recommendations, maintainability tips, and efficiency improvements.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Upgrade for better contextual understanding
            messages=[
                {"role": "system",
                 "content": "You are an expert AI code reviewer. Based on the analysis summary, provide a concise final"
                            "guideline for the developer to follow. Keep it within 3-5 bullet points focusing on:\n"
                            "Use this format:\n\n"
                            "🚀 **Final Coding Guideline** 🚀\n\n"
                            "1️⃣ **Security Improvements:**\n"
                            "2️⃣ **Code Readability & Maintainability:**\n"
                            "3️⃣ **Performance Optimization:**\n"
                            "4️⃣  **Reference Links / Guidelines:**\n"
                 },
                {"role": "user",
                 "content": f"Here is the code analysis summary:\n{summary}\n\nGenerate a detailed final guideline for the developer to improve the overall code quality."}
            ],
            max_tokens=300,  # Increase token limit to provide detailed output
            temperature=0.2  # Keep responses structured and predictable
        )

        # Extract AI response
        guideline_response = response.choices[0].message.content

        # Format for HTML rendering
        formatted_guideline = guideline_response.replace("🚀 **Final Coding Guideline** 🚀", "<h3>🚀 Final Coding Guideline 🚀</h3>") \
            .replace("1️⃣ **Security Improvements:**", "<h4>🔒 Security Improvements</h4><ul>") \
            .replace("2️⃣ **Code Readability & Maintainability:**", "</ul><h4>📖 Code Readability & Maintainability</h4><ul>") \
            .replace("3️⃣ **Performance Optimization:**", "</ul><h4>⚡ Performance Optimization</h4><ul>") \
            .replace("4️⃣ **Reference Links / Guidelines:**", "</ul><h4>📚 Reference Guideline</h4><ul>")

        return formatted_guideline  # ✅ Now formatted for better HTML display
    except Exception as e:
        return f"Error generating final guideline: {str(e)}"

# def ai_generate_guideline(summary):
#     """
#     Uses OpenAI to generate a final coding guideline for the developer based on the summary report.
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # Upgrade to "gpt-4o" for better results
#             messages=[
#                 {"role": "system",
#                  "content": "You are an expert AI coding assistant. Based on the analysis summary, provide a final "
#                             "guideline for the developer to follow. Focus on best practices, security, maintainability, "
#                             "and efficiency. Ensure the guideline is actionable and concise."},
#                 {"role": "user",
#                  "content": f"Given this analysis summary:\n{summary}\n\nProvide a final coding guideline to improve the overall code quality."}
#             ],
#             max_tokens=200,
#             temperature=0.3  # Keep it concise and structured
#         )
#
#         # Extract AI response
#         guideline_response = response.choices[0].message.content
#
#         # Format for HTML rendering
#         formatted_guideline = guideline_response.replace("Best Practice:", "<b>Best Practice:</b>") \
#             .replace("Security Tip:", "<b>Security Tip:</b>") \
#             .replace("Performance Tip:", "<b>Performance Tip:</b>") \
#             .replace("Maintainability Tip:", "<b>Maintainability Tip:</b>")
#
#         return formatted_guideline  # ✅ Now formatted for HTML rendering
#     except Exception as e:
#         return f"Error generating final guideline: {str(e)}"


@api_view(['GET', 'POST'])
def analyze_code_view(request):
    """
    Handles code analysis: Takes input, processes it, stores results in PostgreSQL, and retrieves past analysis if available.
    """
    suggestions = []
    summary = {
        "total_snippets": 0,
        "total_suggestions": 0,
        "total_lines": 0,
        "categories": {},
        "severity": {"Critical": 0, "Medium": 0, "Low": 0},
        "complexity_metrics": {},
    }
    code_snippet = ""
    final_guideline = ""

    if request.method == 'POST':
        # ✅ Handle GitHub repository submission
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            github_repo_url = data.get("github_url", "").strip()

            if github_repo_url:
                files, error = fetch_github_files(github_repo_url)
                if error:
                    return JsonResponse({"error": error})

                if files:
                    return JsonResponse({"files": files})  # ✅ Return fetched GitHub files

        # ✅ Handle normal code submission
        code_snippet = request.POST.get('code', '').strip()
        project_name = request.POST.get('project_name', '').strip()

        # ✅ Auto-generate project name if none is provided
        if not project_name:
            project_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"🆕 Auto-generated project name: {project_name}")

        # ✅ Check if project already exists, else create it
        project, _ = Project.objects.get_or_create(name=project_name)

        if not code_snippet:
            return render(request, 'analyze_code.html', {
                "code": "", "suggestions": [], "summary": summary, "final_guideline": "",
                "error": "No code provided for analysis."
            })

        try:
            print(f"📥 Received Code Snippet: {code_snippet[:100]}...")  # ✅ DEBUG: Verify input
            snippets = split_code_snippets(code_snippet)
            summary["total_snippets"] = len(snippets)
            summary["total_lines"] = code_snippet.count('\n') + 1

            for line_num, snippet in enumerate(snippets, start=1):
                # ✅ **Check if this snippet has already been analyzed**
                existing_snippet = CodeSnippet.objects.filter(snippet=snippet).first()

                if existing_snippet:
                    print(f"✅ Found existing analysis for snippet: {snippet[:50]}...")

                    # ✅ **Ensure categories & severity are updated from DB suggestions**
                    category = categorize_suggestion(existing_snippet.model_suggestion)
                    severity = determine_severity(existing_snippet.model_suggestion)

                    summary["total_suggestions"] += 1
                    summary["categories"].setdefault(category, 0)
                    summary["categories"][category] += 1
                    summary["severity"][severity] += 1

                    suggestions.append({
                        "code": existing_snippet.snippet,
                        "suggestion": existing_snippet.model_suggestion,
                        "category": category,
                        "severity": severity,
                        "line": line_num
                    })
                    continue  # Skip AI & Model processing for existing snippets

                try:
                    print(f"🚀 No previous analysis found. Running AI and Model analysis for snippet {line_num}")

                    # ✅ **Perform AI-based T5 Model Analysis**
                    # inputs = tokenizer(
                    #     snippet, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
                    # )
                    # inputs = {key: value.to(device) for key, value in inputs.items()}
                    # model.eval()

                    # with torch.no_grad():
                    #     outputs = model.generate(
                    #         inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True
                    #     )
                    #     t5_suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"🚀 Calling Flask model for snippet {line_num}...")
                    t5_suggestion = call_flask_model(snippet)  # Calls Flask API

                    # ✅ **Generate AI-based GPT analysis**
                    gpt_suggestion = ai_code_analysis(snippet)

                    # ✅ **Combine AI-generated suggestions**
                    final_suggestion = f"Suggestion:\n{t5_suggestion}\n\nDetailed Analysis:\n{gpt_suggestion}"

                    # ✅ **Categorize and determine severity**
                    category = categorize_suggestion(final_suggestion)
                    severity = determine_severity(final_suggestion)

                    # ✅ **Update summary**
                    summary["total_suggestions"] += 1
                    summary["categories"].setdefault(category, 0)
                    summary["categories"][category] += 1
                    summary["severity"][severity] += 1

                    # ✅ **Store the snippet analysis in PostgreSQL**
                    CodeSnippet.objects.create(
                        project=project,
                        snippet=snippet,
                        ai_suggestion=gpt_suggestion,
                        model_suggestion=t5_suggestion,
                    )

                    print(f"📌 Stored analysis for new snippet {line_num} in DB.")

                    # ✅ **Store the suggestion in results**
                    suggestions.append({
                        "code": snippet,
                        "category": category,
                        "suggestion": final_suggestion,
                        "severity": severity,
                        "line": line_num
                    })

                except Exception as snippet_error:
                    print(f"❌ Error analyzing snippet {line_num}: {str(snippet_error)}")
                    suggestions.append({
                        "code": snippet,
                        "suggestion": f"Error: {str(snippet_error)}",
                        "severity": "Low",
                        "line": line_num
                    })

            # ✅ **Perform Complexity Analysis**
            print("🔍 Performing Complexity Analysis...")
            summary["complexity_metrics"] = analyze_code_complexity(code_snippet)
            print(f"✅ Complexity Results: {summary['complexity_metrics']}")

            # ✅ **Generate Developer Guideline**
            final_guideline = ai_generate_guideline(summary)

        except Exception as e:
            print(f"❌ Critical Error in analysis: {str(e)}")
            suggestions.append({
                "code": "",
                "suggestion": f"Error analyzing code: {str(e)}",
                "severity": "Critical",
                "line": 0
            })

    return render(
        request,
        'analyze_code.html',
        {'code': code_snippet, 'suggestions': suggestions, 'summary': summary, 'final_guideline': final_guideline}
    )










# @api_view(['GET', 'POST'])
# def analyze_code_view(request):
#     suggestions = []
#     summary = {
#         "total_snippets": 0,
#         "total_suggestions": 0,
#         "total_lines": 0,
#         "categories": {},
#         "severity": {"Critical": 0, "Medium": 0, "Low": 0},
#         "complexity_metrics": {}  # ✅ Ensure complexity metrics are initialized
#     }
#     code_snippet = ""
#
#     if request.method == 'POST':
#         # ✅ Handle GitHub repository submission
#         if request.content_type == 'application/json':
#             data = json.loads(request.body)
#             github_repo_url = data.get("github_url", "").strip()
#
#             if github_repo_url:
#                 files, error = fetch_github_files(github_repo_url)
#                 if error:
#                     return JsonResponse({"error": error})
#
#                 if files:
#                     return JsonResponse({"files": files})  # ✅ Return fetched GitHub files
#
#         # ✅ Handle normal code submission
#         code_snippet = request.POST.get('code', '').strip()
#         if not code_snippet:
#             return render(request, 'analyze_code.html', {
#                 "code": "", "suggestions": [], "summary": summary,
#                 "error": "No code provided for analysis."
#             })
#
#         try:
#             print(f"📥 Received Code Snippet: {code_snippet[:100]}...")  # ✅ DEBUG: Verify input
#             snippets = split_code_snippets(code_snippet)
#             summary["total_snippets"] = len(snippets)
#             summary["total_lines"] = code_snippet.count('\n') + 1
#
#             for line_num, snippet in enumerate(snippets, start=1):
#                 try:
#                     # ✅ **Perform AI-based T5 Model Analysis**
#                     inputs = tokenizer(
#                         snippet,
#                         truncation=True,
#                         padding="max_length",
#                         max_length=512,
#                         return_tensors="pt"
#                     )
#                     inputs = {key: value.to(device) for key, value in inputs.items()}
#                     model.eval()
#
#                     with torch.no_grad():
#                         outputs = model.generate(
#                             inputs["input_ids"],
#                             max_length=256,
#                             num_beams=5,
#                             early_stopping=True
#                         )
#                         t5_suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#                     # ✅ **Generate AI-based GPT analysis**
#                     gpt_suggestion = ai_code_analysis(snippet)
#
#                     # ✅ **Combine AI-generated suggestions**
#                     final_suggestion = f"Suggestion:\n{t5_suggestion}\n\nDetailed Analysis:\n{gpt_suggestion}"
#
#                     # ✅ **Categorize and determine severity**
#                     category = categorize_suggestion(final_suggestion)
#                     severity = determine_severity(final_suggestion)
#
#                     # ✅ **Update summary**
#                     summary["total_suggestions"] += 1
#                     summary["categories"].setdefault(category, 0)
#                     summary["categories"][category] += 1
#                     summary["severity"][severity] += 1
#
#                     # ✅ **Store the suggestion in results**
#                     suggestions.append({
#                         "code": snippet,
#                         "category": category,
#                         "suggestion": final_suggestion,
#                         "severity": severity,
#                         "line": line_num
#                     })
#
#                 except Exception as snippet_error:
#                     suggestions.append({
#                         "code": snippet,
#                         "suggestion": f"Error: {str(snippet_error)}",
#                         "severity": "Low",
#                         "line": line_num
#                     })
#
#             # ✅ **Perform Complexity Analysis**
#             print("🔍 Performing Complexity Analysis...")  # ✅ DEBUG LOG
#             summary["complexity_metrics"] = analyze_code_complexity(code_snippet)
#             print(f"✅ Complexity Results: {summary['complexity_metrics']}")  # ✅ DEBUG OUTPUT
#
#         except Exception as e:
#             suggestions.append({
#                 "code": "",
#                 "suggestion": f"Error analyzing code: {str(e)}",
#                 "severity": "Critical",
#                 "line": 0
#             })
#
#     return render(
#         request,
#         'analyze_code.html',
#         {'code': code_snippet, 'suggestions': suggestions, 'summary': summary}
#     )



def categorize_suggestion(suggestion):
    """Categorize a suggestion based on its content using meaningful categories."""
    suggestion_lower = suggestion.lower()

    # Security-related issues
    if any(term in suggestion_lower for term in [
        "sql injection", "command injection", "hardcoded secret",
        "authentication", "authorization", "xss", "csrf", "security",
        "data exposure", "encryption", "password hashing", "secret key"
    ]):
        return "Security Vulnerabilities"

    # Performance-related improvements
    elif any(term in suggestion_lower for term in [
        "optimize", "performance", "efficiency", "redundant code",
        "slow database query", "memory leak", "loop optimization",
        "batch processing", "unnecessary computation"
    ]):
        return "Performance Enhancements"

    # Code Quality & Maintainability
    elif any(term in suggestion_lower for term in [
        "refactor", "code quality", "readability", "maintainability",
        "magic numbers", "naming conventions", "global variables",
        "long function", "long class", "duplicate code", "function too complex",
        "dead code", "improve clarity"
    ]):
        return "Code Readability & Maintainability"

    # Resource & Memory Management Issues
    elif any(term in suggestion_lower for term in [
        "resource leak", "memory management", "insecure file handling",
        "open file without closing", "socket not closed",
        "unclosed connection", "improper resource release"
    ]):
        return "Resource & Memory Management"

    # Exception Handling Issues
    elif any(term in suggestion_lower for term in [
        "empty exception handling", "broad exception handling",
        "unhandled exception", "try except pass", "error handling",
        "catch generic exception", "silent failure"
    ]):
        return "Exception Handling Issues"

    # Deprecated & Compatibility Issues
    elif any(term in suggestion_lower for term in [
        "deprecated function", "deprecated library", "legacy code",
        "backward compatibility", "unsupported method",
        "modern alternatives"
    ]):
        return "Deprecated & Compatibility Issues"

    # Object-Oriented Programming (OOP) Best Practices
    elif any(term in suggestion_lower for term in [
        "inheritance misuse", "misuse of polymorphism", "encapsulation violation",
        "tight coupling", "improper abstraction", "large class",
        "single responsibility principle violation"
    ]):
        return "Object-Oriented Design Issues"

    # API & Cloud Security Issues
    elif any(term in suggestion_lower for term in [
        "exposed api key", "weak api authentication",
        "rate limiting missing", "caching missing", "api security",
        "insecure endpoint", "unvalidated input"
    ]):
        return "API & Cloud Security Issues"

    # Business Logic Vulnerabilities
    elif any(term in suggestion_lower for term in [
        "flawed business logic", "logic flaw", "invalid transaction",
        "unauthorized access", "insufficient validation", "race condition"
    ]):
        return "Business Logic Vulnerabilities"

    # Compliance & Standard Violations
    elif any(term in suggestion_lower for term in [
        "violates coding standards", "non-compliant", "does not follow",
        "missing documentation", "inconsistent style", "linting issues",
        "coding standard violation"
    ]):
        return "Compliance & Standard Violations"

    # API Design Issues
    elif any(term in suggestion_lower for term in [
        "bad api design", "poor rest implementation", "missing status codes",
        "improper json response", "error response missing", "misuse of http verbs"
    ]):
        return "API Design & Best Practices"

    # Multithreading & Concurrency Issues
    elif any(term in suggestion_lower for term in [
        "race condition", "deadlock", "thread safety", "improper synchronization",
        "mutex missing", "concurrent modification"
    ]):
        return "Multithreading & Concurrency Issues"

    return "General Issues"


def determine_severity(suggestion):
    """Determine the severity level of a suggestion."""
    suggestion_lower = suggestion.lower()

    # Critical Severity
    if any(term in suggestion_lower for term in [
        "sql injection", "command injection", "hardcoded secret", "secret key",
        "weak cryptography", "directory traversal", "resource leak"
    ]):
        return "Critical"

    # Medium Severity
    if any(term in suggestion_lower for term in [
        "should fix", "deep nesting", "global variables",
        "deprecated libraries", "variable shadowing", "magic numbers",
        "unreachable code", "empty exception handling"
    ]):
        return "Medium"

    # Low Severity
    if any(term in suggestion_lower for term in [
        "consistent naming", "consistent return types",
        "unused imports", "excessive comments", "consistent whitespace"
    ]):
        return "Low"

    # Default to Low if no matches
    return "Low"


# def split_code_snippets(code_snippet):
#     """
#     Split the input code snippet into individual Python functions or top-level blocks.
#     """
#     try:
#         # Parse the input code into an Abstract Syntax Tree (AST)
#         tree = ast.parse(code_snippet)
#         snippets = []
#
#         for node in tree.body:  # Iterate over top-level statements
#             # Get the source code for each node
#             snippet = ast.unparse(node) if hasattr(ast, 'unparse') else ast.dump(node)
#             snippets.append(snippet)
#
#         return snippets
#     except Exception as e:
#         print(f"Error parsing code snippets: {e}")
#         return []  # Return an empty list if parsing fails
#
#
#
# @api_view(['GET', 'POST'])
# def analyze_code_view(request):
#     """
#     Analyze the given code snippet(s) for vulnerabilities using the trained model.
#     """
#     suggestions = []
#     code_snippet = ""
#
#     if request.method == 'POST':
#         # Get the code snippet(s) from the form
#         code_snippet = request.POST.get('code', '').strip()
#         if not code_snippet:
#             suggestions.append({"code": "", "suggestion": "Please provide valid code snippet(s)."})
#         else:
#             try:
#                 # Debug: Log the received code snippet(s)
#                 print(f"Received code snippet(s):\n{code_snippet}")
#
#                 # Split code into individual snippets
#                 snippets = split_code_snippets(code_snippet)
#                 if not snippets:
#                     suggestions.append({"code": "", "suggestion": "Error: Unable to parse code snippets."})
#                 else:
#                     # Debug: Log parsed snippets
#                     print("Parsed code snippets:", snippets)
#
#                     # Load the trained model and tokenizer
#                     model_path = "./models/custom_seq2seq_model"  # Adjust path if needed
#                     tokenizer = AutoTokenizer.from_pretrained(model_path)
#                     model = T5ForConditionalGeneration.from_pretrained(model_path)
#                     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#                     model.to(device)
#
#                     # Process each snippet separately
#                     for snippet in snippets:
#                         # Tokenize the individual snippet
#                         inputs = tokenizer(
#                             snippet,
#                             truncation=True,
#                             padding="max_length",
#                             max_length=512,
#                             return_tensors="pt"
#                         )
#                         inputs = {key: value.to(device) for key, value in inputs.items()}
#
#                         # Generate suggestion for the snippet
#                         model.eval()
#                         with torch.no_grad():
#                             outputs = model.generate(
#                                 inputs["input_ids"],
#                                 max_length=256,  # Increased max_length for longer outputs
#                                 num_beams=5,     # Using beam search for better suggestions
#                                 early_stopping=True
#                             )
#
#                             # Decode the suggestion
#                             suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#                         # Append the result as a dictionary
#                         suggestions.append({"code": snippet, "suggestion": suggestion})
#
#             except Exception as e:
#                 # Handle errors during model inference
#                 suggestions.append({"code": "", "suggestion": f"Error analyzing the code: {str(e)}"})
#
#     return render(
#         request,
#         'analyze_code.html',
#         {'code': code_snippet, 'suggestions': suggestions}
#     )


def detect_defects_view(request):
    model_path = "./models/defect_detection_model"
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if request.method == "POST":
        try:
            # Handle pasted code
            code_snippet = request.POST.get("code_snippet", "").strip()
            if not code_snippet:
                return JsonResponse({"error": "No code snippet provided."}, status=400)

            inputs = tokenizer(
                code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=512
            )
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()

            return JsonResponse({"defect_detected": bool(prediction)})

        except Exception as e:
            return JsonResponse({"error": f"Detection failed: {str(e)}"}, status=500)

    return render(request, "detect_defects.html")


# @api_view(['GET', 'POST'])
# def java_code_analysis(request):
#     if request.method == 'POST':
#         try:
#             code = request.POST.get('code', '')
#             if not code:
#                 return JsonResponse({'error': 'No code provided'}, status=400)
#
#             analyser = JavaCodeAnalyzer(code)
#             recommendations = analyser.generate_recommendations()
#             return JsonResponse({'recommendations': recommendations})
#
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#
#     return render(request, 'java_code_analysis.html')


# @api_view(['GET', 'POST'])
# def upload_python_files(request):
#     if request.method == 'POST':
#         files = request.FILES.getlist('files')
#         if not files:
#             return JsonResponse({'error': 'No files uploaded'}, status=400)
#
#         results = {}
#         for file in files:
#             try:
#                 content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
#                 analyzer = PythonCodeAnalyzer(content)
#                 recommendations = analyzer.generate_recommendations()
#                 results[file.name] = recommendations
#             except Exception as e:
#                 results[file.name] = f"Error analyzing file: {str(e)}"
#
#         return JsonResponse({'results': results})
#
#     return render(request, 'upload_python.html')


# @api_view(['GET', 'POST'])
# def upload_java_files(request):
#     if request.method == 'POST':
#         files = request.FILES.getlist('files')
#         if not files:
#             return JsonResponse({'error': 'No files uploaded'}, status=400)
#
#         results = {}
#         for file in files:
#             try:
#                 content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
#                 analyzer = JavaCodeAnalyzer(content)
#                 recommendations = analyzer.generate_recommendations()
#                 results[file.name] = recommendations
#             except Exception as e:
#                 results[file.name] = f"Error analyzing file: {str(e)}"
#
#         return JsonResponse({'results': results})
#
#     return render(request, 'upload_java.html')

@api_view(['GET', 'POST'])
def drink_list(request, format=None):
    if request.method == 'GET':
        drinks = Drink.objects.all()
        serializer = DrinkSerializer(drinks, many=True)
        return Response(serializer.data)
    if request.method == 'POST':
        serializer = DrinkSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PUT', 'DELETE'])
def drink_detail(request, id, format=None):
    Drink.objects.get(pk=id)

    try:
        drink = Drink.objects.get(pk=id)
    except Drink.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = DrinkSerializer(drink)
        return Response(serializer.data)
    elif request.method == 'PUT':
        serializer = DrinkSerializer(drink, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        drink.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# @api_view(['POST', 'GET'])
# def calculate_complexity_line_by_line(request):
#     if request.method == 'POST':
#         code = request.data.get('code')
#         if not code:
#             return Response({'error': 'Code is required'}, status=status.HTTP_400_BAD_REQUEST)
#
#         complexity = calculate_code_complexity_line_by_line(code)
#         return Response(complexity, status=status.HTTP_200_OK)



@api_view(['GET', 'POST'])
def calculate_complexity_line_by_line(request):
    if request.method == 'POST':
        code = request.data.get('code')
        if not code:
            return Response({'error': 'Code is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Call your function to calculate complexity
        result = calculate_code_complexity_line_by_line(code)
        complexity = result['line_complexities']
        cbo = result['cbo']

        # Render the result in the template
        return render(request, 'complexityA_table.html', {'complexities': complexity, 'cbo': cbo})

    # If GET request, just show the form
    return render(request, 'complexityA_form.html')


def get_thresholds():
    thresholds_file = os.path.join(settings.BASE_DIR, 'media', 'thresholds.json')
    if os.path.exists(thresholds_file):
        with open(thresholds_file, 'r') as json_file:
            thresholds = json.load(json_file)
        return thresholds
    # Default values if the file doesn't exist
    return {'threshold_low': 10, 'threshold_medium': 20}


@api_view(['GET', 'POST'])
def calculate_complexity_multiple_java_files(request):
    if request.method == 'POST':
        # Expecting multiple Java files in the request
        files = request.FILES.getlist('files')
        if not files:
            return Response({'error': 'No files uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file_contents = {}
        for file in files:
            # Read the content of each file
            content = file.read().decode('utf-8')  # Assuming the files are UTF-8 encoded
            file_contents[file.name] = content

        # Load thresholds from the JSON file
        thresholds = get_thresholds()
        threshold_low = thresholds.get('threshold_low', 10)
        threshold_medium = thresholds.get('threshold_medium', 20)

        # Call your function to calculate complexity for multiple files
        result = calculate_code_complexity_multiple_files(file_contents)

        # Prepare a list to store complexities for each file
        complexities = []

        # Create a table for the response
        results_table = PrettyTable()
        results_table.field_names = ["Filename", "Line Number", "Line", "Size", "Tokens",
                                     "Control Structure Complexity", "Nesting Weight",
                                     "Inheritance Weight", "Compound Condition Weight",
                                     "Try-Catch Weight", "Thread Weight", "Total Complexity"]

        # Prepare another table for displaying MPC and CBO values
        mp_cbo_table = PrettyTable()
        mp_cbo_table.field_names = ["Filename", "CBO", "MPC"]

        # Collect complexities for each file
        for filename, file_data in result.items():
            complexity_data = file_data['complexity_data']
            cbo = file_data['cbo']
            mpc = file_data['mpc']
            method_complexities = file_data.get('method_complexities', [])
            print("method_complexities", method_complexities)
            recommendations = file_data['recommendation']
            pie_chart_path = file_data['pie_chart_path']

            for line_data in complexity_data:
                results_table.add_row([filename] + line_data)  # Now line_data has 9 values

            # Categorize each method based on total complexity
            categorized_methods = []
            for method_name, method_data in method_complexities.items():
                if isinstance(method_data, dict):
                    total_complexity = method_data.get('total_complexity', 0)

                    # Determine the category based on thresholds
                    if total_complexity <= threshold_low:
                        category = 'Low'
                    elif total_complexity <= threshold_medium:
                        category = 'Medium'
                    else:
                        category = 'High'

                    # Append category to the method data
                    categorized_method = method_data.copy()
                    categorized_method['category'] = category
                    categorized_method['method_name'] = method_name
                    categorized_methods.append(categorized_method)
                else:
                    print(f"Unexpected format in method_data: {method_data}")

            complexities.append({
                'filename': filename,
                'complexity_data': complexity_data,
                'cbo': cbo,
                'mpc': mpc,
                'method_complexities': categorized_methods,
                'recommendations': recommendations,
                'pie_chart_path': pie_chart_path
            })

        # Log the result table for debugging or reference
        # print(results_table)

        # Instead of returning a JSON response, render the template and pass complexities
        return render(request, 'complexity_table.html', {'complexities': complexities})

    # If GET request, just show the form
    return render(request, 'complexity_form.html')


@api_view(['GET', 'POST'])
def calculate_complexity_line_by_line_csharp(request):
    if request.method == 'POST':
        code = request.data.get('code')
        if not code:
            return Response({'error': 'Code is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Call the function to calculate complexity line-by-line for a C# file
        result = calculate_code_complexity_line_by_line(code)
        complexity = result['line_complexities']
        cbo = result['cbo']

        # Render the result in the template
        return render(request, 'complexityB_table.html', {'complexities': complexity, 'cbo': cbo})

    # If GET request, just show the form
    return render(request, 'complexityB_form.html')


@api_view(['GET', 'POST'])
def calculate_complexity_multiple_csharp_files(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        if not files:
            return Response({'error': 'No files uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file_contents = {}
        for file in files:
            content = file.read().decode('utf-8')
            file_contents[file.name] = content

        # Call the function to calculate complexity for multiple files
        result = calculate_code_complexity_multiple_files(file_contents)

        # Prepare data structures for response
        complexities = []

        results_table = PrettyTable()
        results_table.field_names = [
            "Filename", "Line Number", "Line", "Size", "Tokens",
            "Control Structure Complexity", "Nesting Weight", "Inheritance Weight",
            "Compound Condition Weight", "Try-Catch Weight", "Thread Weight", "Total Complexity"
        ]

        mp_cbo_table = PrettyTable()
        mp_cbo_table.field_names = ["Filename", "CBO", "MPC"]

        for filename, file_data in result.items():
            complexity_data = file_data['complexity_data']
            cbo = file_data['cbo']
            mpc = file_data['mpc']

            # Check the structure of line_data and adjust if necessary
            for line_data in complexity_data:
                # Ensure line_data is a dictionary and has the expected keys
                if isinstance(line_data, dict):
                    total_complexity = sum([
                        line_data.get('size', 0), line_data.get('control_structure_complexity', 0),
                        line_data.get('nesting_level', 0), line_data.get('inheritance_level', 0),
                        line_data.get('compound_condition_weight', 0), line_data.get('try_catch_weight', 0),
                        line_data.get('thread_weight', 0)
                    ])
                    results_table.add_row([
                        filename, line_data.get('line_number', ''), line_data.get('line_content', ''),
                        line_data.get('size', 0), ', '.join(line_data.get('tokens', [])),
                        line_data.get('control_structure_complexity', 0), line_data.get('nesting_level', 0),
                        line_data.get('inheritance_level', 0), line_data.get('compound_condition_weight', 0),
                        line_data.get('try_catch_weight', 0), line_data.get('thread_weight', 0), total_complexity
                    ])

            complexities.append({
                'filename': filename,
                'complexity_data': complexity_data,
                'cbo': cbo,
                'mpc': mpc
            })
            mp_cbo_table.add_row([filename, cbo, mpc])

        return render(request, 'complexityC_table.html', {'complexities': complexities})

    return render(request, 'complexityC_form.html')


@api_view(['GET', 'POST'])
def calculate_complexity(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']  # Assuming a form is used for file upload

        # Check if the file is a CSV
        if uploaded_file.name.endswith('.csv'):
            # Define the media directory path
            media_dir = os.path.join(settings.BASE_DIR, 'media')

            # Create the media directory if it doesn't exist
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Save the uploaded file in the media directory
            file_path = os.path.join(media_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Load the CSV into a Pandas DataFrame
            data = pd.read_csv(file_path)

            # Columns to calculate correlations with WCC
            target_columns = ['Complexity', 'Maintainability', 'Readability']

            # Initialize dictionaries to store correlation results
            pearson_results = {}
            spearman_results = {}

            # Perform Pearson and Spearman correlations
            for column in target_columns:
                # Pearson Correlation
                pearson_corr, pearson_p_value = pearsonr(data['WCC'], data[column])
                pearson_results[column] = {
                    'correlation': round(pearson_corr, 4),
                    'p_value': round(pearson_p_value, 4)
                }

                # Spearman Correlation
                spearman_corr, spearman_p_value = spearmanr(data['WCC'], data[column])
                spearman_results[column] = {
                    'correlation': round(spearman_corr, 4),
                    'p_value': round(spearman_p_value, 4)
                }

            # Generate correlation matrix
            correlation_matrix = data[['WCC', 'Complexity', 'Maintainability', 'Readability']].corr()

            # Save the heatmap as an image
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            heatmap_path = os.path.join(media_dir, 'heatmap.png')
            plt.title('Correlation Matrix between WCC and Metrics')
            plt.savefig(heatmap_path)
            plt.close()  # Close the plot after saving to avoid memory leaks

            # Scatter plot for each target column
            scatter_plots = {}
            for column in target_columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(data['WCC'], data[column], alpha=0.6, c='green')
                plt.title(f'WCC vs {column}')
                plt.xlabel('WCC Scores')
                plt.ylabel(column)
                scatter_plot_path = os.path.join(media_dir, f'scatter_{column}.png')
                plt.savefig(scatter_plot_path)
                plt.close()
                scatter_plots[column] = scatter_plot_path

            # Prepare the results
            context = {
                'pearson_results': pearson_results,
                'spearman_results': spearman_results,
                'heatmap_path': heatmap_path,
                'scatter_plots': scatter_plots
            }

            # Render the result page with the context
            return render(request, 'analysis_result.html', context)

        else:
            return HttpResponse("Invalid file format. Please upload a CSV file.")

    # If it's a GET request, render the upload form page
    return render(request, 'upload.html')
