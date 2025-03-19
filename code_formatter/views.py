from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .complexity_calculator import calculate_loc, calculate_readability
import openai
import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from .models import Guideline, DesignPatternResource
from .forms import GuidelineForm, DesignPatternResourceForm
from .models import CodeRefactoringRecord
from .utils import analyze_code, refactor_code
import re
import requests
import base64
from django.contrib import messages
import difflib

# Create OpenAI Client with API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GitHub API Token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

GUIDELINE_PROMPTS = {
    "Factory": "What are the best practices for using the Factory Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Strategy": "What are the best practices for implementing the Strategy Pattern in code? Please provide your answer in a numbered list without any markdown formatting.",
    "Observer": "What are the best practices for using the Observer Pattern effectively? Please provide your answer in a numbered list without any markdown formatting.",
    "AbstractFactory": "What are the best practices for using the Abstract Factory Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Builder": "What are the best practices for using the Builder Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Prototype": "What are the best practices for using the Prototype Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Singleton": "What are the best practices for using the Singleton Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Adapter": "What are the best practices for using the Adapter Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Bridge": "What are the best practices for using the Bridge Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Composite": "What are the best practices for using the Composite Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Decorator": "What are the best practices for using the Decorator Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Facade": "What are the best practices for using the Facade Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Flyweight": "What are the best practices for using the Flyweight Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Proxy": "What are the best practices for using the Proxy Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "ChainOfResponsibility": "What are the best practices for using the Chain of Responsibility Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Command": "What are the best practices for using the Command Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Interpreter": "What are the best practices for using the Interpreter Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Iterator": "What are the best practices for using the Iterator Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Mediator": "What are the best practices for using the Mediator Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Memento": "What are the best practices for using the Memento Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "State": "What are the best practices for using the State Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "TemplateMethod": "What are the best practices for using the Template Method Pattern in software design? Please provide your answer in a numbered list without any markdown formatting.",
    "Visitor": "What are the best practices for using the Visitor Pattern in software design? Please provide your answer in a numbered list without any markdown formatting."
}


# Function to display the refactor page
def refactor_view(request):
    return render(request, 'code_formatter/refactor.html')


# Function to handle file upload and extract code
def upload_code(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read().decode('utf-8')
        return JsonResponse({'code': file_content})

    return JsonResponse({'error': 'Invalid request'}, status=400)


# Function to handle code refactoring using OpenAI API
@csrf_exempt
def refactor_code(request):
    """Handles code refactoring with optional guideline usage."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            code = data.get('code', '').strip()
            use_guidelines = data.get('use_guidelines', False)  # Check if user enabled guidelines

            if not code:
                return JsonResponse({'error': 'No code provided'}, status=400)

            # ✅ Identify the design pattern used in the code
            detected_pattern = analyze_code_for_pattern(code)

            # ✅ Initialize guidelines_prompt properly
            guidelines_prompt = ""  # Default empty guidelines

            # ✅ Check for guideline restrictions
            if use_guidelines:
                guidelines = Guideline.objects.all()
                pattern_guidelines = {g.pattern: g.rule for g in guidelines}  # Convert to dict

                if detected_pattern in pattern_guidelines:
                    guideline_text = pattern_guidelines[detected_pattern]
                    guidelines_prompt = guideline_text  # ✅ Set guidelines prompt correctly
                    return JsonResponse({
                        'message': f"Refactoring blocked. {detected_pattern} pattern is restricted.",
                        'guideline': guideline_text
                    }, status=400)  # 🚨 Stop refactoring if the pattern is disallowed

            # ✅ If no guideline restricts it, proceed with refactoring
            # Calculate LOC and Readability for the original code
            original_loc = calculate_loc(code)
            original_readability = calculate_readability(code)

            # AI Code Refactoring Request
            prompt = f"""
            Refactor the following code to improve structure, readability, and efficiency. 
            Ensure that you:
            1. Improve maintainability and efficiency.
            2. Follow best practices while keeping the functionality intact.

            ### Original Code:
            {code}

            ### Guidelines (if applicable):
            {guidelines_prompt if guidelines_prompt else "No specific guidelines provided."}

            After refactoring, clearly provide a "Changes Made" section listing the improvements in bullet points.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional code refactoring assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            # ✅ Ensure response_text is properly extracted
            response_text = str(response.choices[0].message.content).strip()

            # ✅ Debugging log to check AI response
            print("AI Response:", response_text)

            # ✅ Extract the refactored code using regex
            code_match = re.search(r"```java([\s\S]*?)```", response_text)
            refactored_code = code_match.group(1).strip() if code_match else response_text  # Fallback to full response

            # ✅ Extract "Changes Made" section
            changes_match = re.search(r"Changes Made:\n([\s\S]*)", response_text)
            changes_made = changes_match.group(1).strip().split("\n") if changes_match else []

            # Calculate LOC and Readability for the refactored code
            refactored_loc = calculate_loc(refactored_code)
            refactored_readability = calculate_readability(refactored_code)

            # Save to DB
            record = CodeRefactoringRecord.objects.create(
                original_code=code,
                refactored_code=refactored_code,
                original_complexity=original_loc,
                refactored_complexity=refactored_loc,
                original_readability=original_readability,
                refactored_readability=refactored_readability,
            )

            return JsonResponse({
                'refactored_code': refactored_code,
                'changes_made': changes_made,  # ✅ Return extracted changes
                'original_loc': original_loc,
                'refactored_loc': refactored_loc,
                'original_readability': original_readability,
                'refactored_readability': refactored_readability,
                'id': record.id
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


# API Endpoint for AI to Fetch Guidelines
def get_guidelines(request):
    guidelines = list(Guideline.objects.values('pattern', 'rule'))
    return JsonResponse({'guidelines': guidelines})


def define_guidelines(request):
    if request.method == "POST":
        form = GuidelineForm(request.POST, request.FILES)  # Include request.FILES to handle file uploads
        if form.is_valid():
            guideline = form.save(commit=False)
            if 'company_logo' in request.FILES:
                guideline.company_logo = request.FILES['company_logo']
            guideline.save()
            return redirect('define_guidelines')

    guidelines = Guideline.objects.all()
    return render(request, "code_formatter/define_guidelines.html", {"guidelines": guidelines})


@csrf_exempt
def edit_guideline(request, id):
    if request.method == "POST":
        try:
            guideline = get_object_or_404(Guideline, id=id)

            if "company_name" in request.POST:
                guideline.company_name = request.POST["company_name"]

            if "pattern" in request.POST:
                guideline.pattern = request.POST["pattern"]

            if "rule" in request.POST:
                guideline.rule = request.POST["rule"]

            if "company_logo" in request.FILES:  # Handle logo update
                guideline.company_logo = request.FILES["company_logo"]

            guideline.save()

            return JsonResponse({
                "success": True,
                "company_name": guideline.company_name,
                "pattern": guideline.pattern,
                "rule": guideline.rule,
                "company_logo": guideline.company_logo.url if guideline.company_logo else None
            })
        except Guideline.DoesNotExist:
            return JsonResponse({"success": False, "error": "Guideline not found"})
    return JsonResponse({"success": False, "error": "Invalid request"})


def delete_guideline(request, guideline_id):
    if request.method == "POST":
        guideline = get_object_or_404(Guideline, id=guideline_id)
        guideline.delete()
        return JsonResponse({"success": True})  # ✅ Return a JSON response

    return JsonResponse({"success": False, "error": "Invalid request"}, status=400)


@csrf_exempt
def refactor_and_compare(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            original_code = data.get("code", "")

            print("Received code:", original_code)  # Debugging log

            # Analyze Original Code Complexity
            original_complexity, original_readability = analyze_code(original_code)

            # Refactor Code
            refactored_code = refactor_code(original_code)

            # Analyze Refactored Code Complexity
            refactored_complexity, refactored_readability = analyze_code(refactored_code)

            # Save to Database
            history = CodeRefactoringRecord.objects.create(
                original_code=original_code,
                refactored_code=refactored_code,
                original_complexity=original_complexity,
                refactored_complexity=refactored_complexity,
                original_readability=original_readability,
                refactored_readability=refactored_readability
            )

            print("Saved to database:", history.id)  # Debugging log

            return JsonResponse({
                "success": True,
                "before_code": original_code,
                "after_code": refactored_code,
                "original_complexity": original_complexity,
                "refactored_complexity": refactored_complexity,
                "original_readability": original_readability,
                "refactored_readability": refactored_readability,
                "id": history.id
            })
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Invalid request"})


def refactor_code_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            original_code = data.get('code', '')

            # Simulate AI-based refactoring (Replace this with actual AI call)
            refactored_code = "Refactored: " + original_code  # Placeholder result
            original_complexity = 78  # Replace with actual analysis
            refactored_complexity = 92  # Replace with actual analysis
            original_readability = 117.33  # Replace with actual analysis
            refactored_readability = 116.73  # Replace with actual analysis

            # Save to database
            history = CodeRefactoringRecord.objects.create(
                original_code=original_code,
                refactored_code=refactored_code,
                original_complexity=original_complexity,
                refactored_complexity=refactored_complexity,
                original_readability=original_readability,
                refactored_readability=refactored_readability,
            )

            return JsonResponse({
                "refactored_code": refactored_code,
                "original_loc": original_complexity,
                "refactored_loc": refactored_complexity,
                "original_readability": original_readability,
                "refactored_readability": refactored_readability,
                "id": history.id
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)


# ✅ AI-Powered Guideline Generation
@csrf_exempt
def generate_guideline(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            pattern = data.get("pattern", "").strip()

            if not pattern or pattern not in GUIDELINE_PROMPTS:
                return JsonResponse({"error": "Invalid pattern selected"}, status=400)

            prompt = GUIDELINE_PROMPTS[pattern]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional software architect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            # ✅ Correct way to extract the response
            guideline_text = response.choices[0].message.content.strip()

            return JsonResponse({"suggestion": guideline_text})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def fetch_github_file(request):
    """Fetches a file from a GitHub repository based on user input."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            repo_url = data.get('repo_url', '').strip()
            file_path = data.get('file_path', '').strip()

            if not repo_url or not file_path:
                return JsonResponse({'error': 'Repository URL and file path are required'}, status=400)

            # Extract owner and repo name from the GitHub URL
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 2:
                return JsonResponse({'error': 'Invalid GitHub repository URL'}, status=400)

            owner, repo = parts[-2], parts[-1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"

            headers = {'Authorization': f'token {GITHUB_TOKEN}'}
            response = requests.get(api_url, headers=headers)

            if response.status_code == 200:
                file_content = response.json().get('content', '')
                decoded_content = base64.b64decode(file_content).decode('utf-8')
                return JsonResponse({'code': decoded_content})
            else:
                return JsonResponse({'error': 'Failed to fetch file from GitHub', 'details': response.json()},
                                    status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def create_github_pr(request):
    """Creates a Pull Request (PR) on GitHub with refactored code."""
    if request.method == 'POST':
        try:
            # Ensure GitHub Token is available
            if not GITHUB_TOKEN:
                return JsonResponse({'error': 'GitHub token is missing'}, status=400)

            data = json.loads(request.body)
            repo_url = data.get('repo_url', '').strip()
            file_path = data.get('file_path', '').strip()
            refactored_code = data.get('refactored_code', '').strip()

            if not repo_url or not file_path or not refactored_code:
                return JsonResponse({'error': 'Missing required fields'}, status=400)

            # Extract owner and repo name
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 2:
                return JsonResponse({'error': 'Invalid GitHub repository URL'}, status=400)

            owner, repo = parts[-2], parts[-1]

            # Define API URLs
            repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"
            branch_api_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
            file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
            pr_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

            headers = {
                'Authorization': f'token {GITHUB_TOKEN}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # ✅ Get the default branch dynamically
            repo_response = requests.get(repo_api_url, headers=headers)
            if repo_response.status_code != 200:
                return JsonResponse({'error': 'Repository not found or access denied'}, status=404)

            default_branch = repo_response.json().get('default_branch', 'main')

            # ✅ Get latest commit SHA of the default branch
            branch_response = requests.get(f"{branch_api_url}/heads/{default_branch}", headers=headers)
            if branch_response.status_code != 200:
                return JsonResponse({'error': 'Failed to fetch latest commit SHA'}, status=500)

            latest_commit_sha = branch_response.json().get('object', {}).get('sha')

            # ✅ Check if the branch already exists
            branch_name = "refactored-code-update"
            check_branch_response = requests.get(f"{branch_api_url}/heads/{branch_name}", headers=headers)
            if check_branch_response.status_code == 200:
                return JsonResponse({'error': 'Branch already exists, cannot create duplicate PR'}, status=400)

            # ✅ Create a new branch
            branch_data = {"ref": f"refs/heads/{branch_name}", "sha": latest_commit_sha}
            create_branch_response = requests.post(branch_api_url, headers=headers, json=branch_data)
            if create_branch_response.status_code != 201:
                return JsonResponse({'error': 'Failed to create branch'}, status=500)

            # ✅ Get file SHA for existing file
            file_response = requests.get(file_api_url, headers=headers)
            if file_response.status_code != 200:
                return JsonResponse({'error': 'File not found in repository'}, status=404)

            file_sha = file_response.json().get('sha')

            # ✅ Update file with refactored code
            update_file_data = {
                "message": "Refactored code update",
                "content": base64.b64encode(refactored_code.encode()).decode(),
                "branch": branch_name,
                "sha": file_sha
            }
            update_file_response = requests.put(file_api_url, headers=headers, json=update_file_data)
            if update_file_response.status_code not in [200, 201]:
                return JsonResponse({'error': 'Failed to update file with refactored code'}, status=500)

            # ✅ Create a pull request
            pr_data = {
                "title": "Refactored Code Suggestion",
                "body": "This PR contains refactored code improvements.",
                "head": branch_name,
                "base": default_branch
            }
            pr_response = requests.post(pr_api_url, headers=headers, json=pr_data)

            if pr_response.status_code not in [200, 201]:
                return JsonResponse({'error': 'Failed to create Pull Request'}, status=500)

            pr_url = pr_response.json().get("html_url")

            return JsonResponse({'message': 'Pull Request Created Successfully!', 'pr_url': pr_url})

        except requests.RequestException as req_error:
            return JsonResponse({'error': f'GitHub API request failed: {str(req_error)}'}, status=500)

        except Exception as e:
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


def github_import_modal(request):
    """ Renders the GitHub Import Modal. """
    return render(request, "code_formatter/github_import_modal.html")


def get_github_token(request):
    """Returns the GitHub Access Token securely"""
    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    return JsonResponse({"token": github_token})


def add_resource(request):
    if request.method == "POST":
        form = DesignPatternResourceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('list_resources')  # Redirect to resource list after saving

    else:
        form = DesignPatternResourceForm()

    return render(request, 'code_formatter/add_resource.html', {'form': form})


def list_resources(request):
    resources = DesignPatternResource.objects.all().order_by('-added_on')
    return render(request, 'code_formatter/list_resources.html', {'resources': resources})


@csrf_exempt
def fetch_snippet_diff(request):
    """Return the code difference between original and refactored code."""
    record_id = request.GET.get("record_id")

    if not record_id:
        return JsonResponse({"success": False, "error": "Missing record_id"})

    try:
        record = get_object_or_404(CodeRefactoringRecord, id=record_id)
        original_code = record.original_code  # Keep as a string
        refactored_code = record.refactored_code  # Keep as a string

        # Generate diff using difflib
        original_lines = original_code.splitlines()
        refactored_lines = refactored_code.splitlines()

        diff = difflib.unified_diff(
            original_lines,
            refactored_lines,
            fromfile="Original Code",
            tofile="Refactored Code",
            lineterm=""
        )

        diff_result = "\n".join(diff)

        return JsonResponse({
            "success": True,
            "original_code": original_code,  # Add original code
            "refactored_code": refactored_code,  # Add refactored code
            "diff": diff_result
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
def get_pattern(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            code = data.get("code", "")

            if not code.strip():
                return JsonResponse({"error": "Empty code provided"}, status=400)

            # Call the updated AI-powered pattern analysis function
            suggested_pattern = analyze_code_for_pattern(code)

            return JsonResponse({"pattern": suggested_pattern})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


def analyze_code_for_pattern(code):
    """
    Uses AI and keyword-based detection to find the best-suited design pattern.
    """
    # Dictionary mapping design patterns to common keywords and structure
    patterns = {
        "Factory": ["create", "newInstance", "factory", "instantiate"],
        "Strategy": ["interface", "strategy", "context", "algorithm"],
        "Observer": ["notify", "subscriber", "event", "listener"],
        "Abstract Factory": ["abstract factory", "create", "product", "factory method"],
        "Builder": ["builder", "construct", "step-by-step"],
        "Prototype": ["clone", "prototype", "copy"],
        "Singleton": ["private static", "getInstance", "singleton"],
        "Adapter": ["adapter", "convert", "interface"],
        "Bridge": ["bridge", "decouple", "abstraction", "implementation"],
        "Composite": ["component", "composite", "tree structure"],
        "Decorator": ["extends", "wrap", "component", "dynamic behavior"],
        "Facade": ["facade", "simplify", "unified interface"],
        "Flyweight": ["flyweight", "shared objects", "minimize memory"],
        "Proxy": ["proxy", "control access", "delegate"],
        "Chain of Responsibility": ["chain", "handle", "successor"],
        "Command": ["execute", "command", "invoker"],
        "Interpreter": ["grammar", "interpreter", "parse"],
        "Iterator": ["iterator", "next", "traverse"],
        "Mediator": ["mediator", "communication", "colleague"],
        "Memento": ["memento", "save state", "restore"],
        "State": ["state", "context", "transition"],
        "Template Method": ["template method", "base class", "override"],
        "Visitor": ["visitor", "accept", "operation"],
    }

    # **Step 1: Keyword-Based Detection**
    for pattern, keywords in patterns.items():
        if any(keyword.lower() in code.lower() for keyword in keywords):
            return pattern  # Return if direct match found

    # **Step 2: AI-Based Suggestion for More Complex Detections**
    return get_ai_suggested_pattern(code, patterns)


def get_ai_suggested_pattern(code, patterns):
    """
    Uses OpenAI GPT model to analyze the code and suggest a design pattern.
    """
    try:
        prompt = f"""
        Analyze the following Java code and determine the most appropriate design pattern to apply.

        **Rules for detection:**
        - If multiple instances of a class are created but should only be **one**, suggest `Singleton`.
        - If an interface is used to change behavior at runtime, suggest `Strategy`.
        - If a separate **factory method** is used for object creation, suggest `Factory`.
        - If objects notify observers when they change, suggest `Observer`.
        - If a method builds a complex object step by step, suggest `Builder`.
        - If an object copies itself to create new instances, suggest `Prototype`.
        - If there is a need to convert one interface into another, suggest `Adapter`.
        - If two independent parts of an application need to work together without being tightly coupled, suggest `Bridge`.
        - If a tree structure is used to represent part-whole hierarchies, suggest `Composite`.
        - If additional functionality is added dynamically to an object, suggest `Decorator`.
        - If an object provides a simplified interface to a larger body of code, suggest `Facade`.
        - If there are too many instances of the same object and memory needs to be optimized, suggest `Flyweight`.
        - If an object acts as a placeholder for another object to control access, suggest `Proxy`.
        - If multiple handlers process a request sequentially, suggest `Chain of Responsibility`.
        - If a request is wrapped as an object to parameterize clients with different requests, suggest `Command`.
        - If a language interpreter is implemented with a grammar, suggest `Interpreter`.
        - If an object is needed to iterate over elements of a collection, suggest `Iterator`.
        - If a central object manages communication between multiple objects, suggest `Mediator`.
        - If an object saves and restores its state later, suggest `Memento`.
        - If an object changes its behavior based on its state, suggest `State`.
        - If a base class defines a template for its subclasses to override specific steps, suggest `Template Method`.
        - If new operations need to be added without modifying existing objects, suggest `Visitor`.

        **Example Code:**
        {code}

        **Only return the design pattern name.**
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a software architecture expert specializing in design patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        # Extract AI response
        suggested_pattern = response.choices[0].message.content.strip()

        # Validate the pattern exists in our known list
        if suggested_pattern in patterns.keys():
            return suggested_pattern

        return "General Refactoring Opportunity"

    except Exception as e:
        return f"Error fetching AI suggestion: {str(e)}"


def get_metrics(request):
    """Fetch code metrics for a specific refactoring record using record_id."""
    record_id = request.GET.get("record_id")

    if not record_id:
        return JsonResponse({"success": False, "error": "Missing record_id"}, status=400)

    try:
        record = get_object_or_404(CodeRefactoringRecord, id=record_id)

        metrics = {
            "before": {
                "loc": record.original_complexity,
                "complexity": record.original_complexity,
                "maintainability": record.original_readability,
                "readability": record.original_readability
            },
            "after": {
                "loc": record.refactored_complexity,
                "complexity": record.refactored_complexity,
                "maintainability": record.refactored_readability,
                "readability": record.refactored_readability
            },
            "improvement": {
                "reduction": round((1 - (record.refactored_complexity / record.original_complexity)) * 100, 2),
                "readability": round((record.refactored_readability - record.original_readability), 2),
                "maintainability": round((record.refactored_readability - record.original_readability), 2),
            }
        }

        return JsonResponse({"success": True, "metrics": metrics})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@csrf_exempt
def generate_refactoring_explanation(request):
    """Generates an AI-powered explanation for refactoring improvements."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            record_id = data.get('record_id')

            if not record_id:
                return JsonResponse({'error': 'Missing record_id'}, status=400)

            # Retrieve the code refactoring record
            record = get_object_or_404(CodeRefactoringRecord, id=record_id)

            # Extract metrics
            original_loc = record.original_complexity
            refactored_loc = record.refactored_complexity
            original_readability = record.original_readability
            refactored_readability = record.refactored_readability

            # Construct AI Prompt
            prompt = f"""
            Analyze the code refactoring improvements based on the given metrics.

            **Before Refactoring:**
            - Lines of Code (LOC): {original_loc}
            - Readability Score: {original_readability}

            **After Refactoring:**
            - Lines of Code (LOC): {refactored_loc}
            - Readability Score: {refactored_readability}

            **Code Comparison:**
            - **Original Code:**
            {record.original_code}

            - **Refactored Code:**
            {record.refactored_code}

            **Explain the changes:**
            - How has the structure improved?
            - Even if LOC increased, why does maintainability improve?
            - How do readability and complexity impact the new structure?

            Provide a clear, professional explanation that highlights improvements without using unnecessary jargon.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional software engineer and code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            explanation = response.choices[0].message.content.strip()

            return JsonResponse({"success": True, "explanation": explanation})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)