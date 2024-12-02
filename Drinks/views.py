import json
import os
import torch
import logging

import pandas as pd
from django.conf import settings
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import analysis
from Drinks.models import Drink
from analysis.code_analyzer import CodeAnalyzer
from analysis.java_code_analyser import JavaCodeAnalyzer
from analysis.javascript_code_analyser import JavaScriptCodeAnalyzer
from analysis.php_code_analyser import PHPCodeAnalyzer
from analysis.python_code_analyser import PythonCodeAnalyzer
from .serializers import DrinkSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .complexity_calculator import calculate_code_complexity_multiple_files, ai_recommend_refactoring, \
    calculate_code_complexity_by_method
from .complexity_calculator import calculate_code_complexity_line_by_line
from .complexity_calculator_csharp import calculate_code_complexity_line_by_line_csharp
from django.shortcuts import render
from prettytable import PrettyTable
import statsmodels.api as sm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from rest_framework.decorators import api_view
from django.shortcuts import render
import torch


# @api_view(['GET', 'POST'])
# def python_code_analysis(request):
#     if request.method == 'POST':
#         recommendations = {}
#
#
#         # Handle pasted code
#         code = request.POST.get('code', '').strip()
#         if code:
#             try:
#                 analyzer = PythonCodeAnalyzer(code)
#                 recommendations = analyzer.generate_recommendations()
#             except Exception as e:
#                 return JsonResponse({'error': f"Error analyzing pasted code: {str(e)}"}, status=500)
#
#         # Handle uploaded files
#         files = request.FILES.getlist('files')
#         if files:
#             file_results = {}
#             for file in files:
#                 try:
#                     content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
#                     analyzer = PythonCodeAnalyzer(content)
#                     file_results[file.name] = analyzer.generate_recommendations()
#                 except Exception as e:
#                     file_results[file.name] = f"Error analyzing file: {str(e)}"
#             recommendations['files'] = file_results
#
#         # Group recommendations by line
#         grouped_recommendations = {}
#         if isinstance(recommendations, list):  # For pasted code
#             for rec in recommendations:
#                 line = rec.get('line', 'unknown')
#                 grouped_recommendations.setdefault(line, []).append({
#                     'rule': rec.get('rule'),
#                     'message': rec.get('message'),
#                 })
#         elif isinstance(recommendations, dict):  # For files
#             grouped_recommendations = recommendations
#
#         # return JsonResponse({'recommendations': grouped_recommendations})
#         return render(request, 'python_code_analysis.html', {'recommendations': grouped_recommendations})
#     return render(request, 'python_code_analysis.html')

@api_view(['GET', 'POST'])
def python_code_analysis(request):
    recommendations = {}
    code = ""  # Initialize code to maintain state

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = PythonCodeAnalyzer(code)
                recommendations = analyzer.generate_recommendations()
            except Exception as e:
                recommendations = [{'error': f"Error analyzing pasted code: {str(e)}"}]

        # Handle uploaded files
        files = request.FILES.getlist('files')
        if files:
            file_results = {}
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = PythonCodeAnalyzer(content)
                    file_results[file.name] = analyzer.generate_recommendations()
                except Exception as e:
                    file_results[file.name] = [{'error': f"Error analyzing file: {str(e)}"}]
            recommendations['files'] = file_results

        # Group recommendations by line or filename
        grouped_recommendations = {}
        if isinstance(recommendations, list):  # For pasted code
            for rec in recommendations:
                line = rec.get('line', 'unknown')
                grouped_recommendations.setdefault(line, []).append({
                    'rule': rec.get('rule'),
                    'message': rec.get('message'),
                })
        elif isinstance(recommendations, dict):  # For files
            grouped_recommendations = recommendations

        return render(
            request,
            'python_code_analysis.html',
            {'recommendations': grouped_recommendations, 'code': code}
        )

    return render(request, 'python_code_analysis.html', {'code': code})




# @api_view(['GET', 'POST'])
# def java_code_analysis(request):
#     if request.method == 'POST':
#         recommendations = {}
#
#         # Handle pasted code
#         code = request.POST.get('code', '').strip()
#         if code:
#             try:
#                 analyzer = JavaCodeAnalyzer(code)
#                 recommendations = analyzer.generate_recommendations()
#             except Exception as e:
#                 return JsonResponse({'error': f"Error analyzing pasted code: {str(e)}"}, status=500)
#
#         # Handle uploaded files
#         files = request.FILES.getlist('files')
#         if files:
#             file_results = {}
#             for file in files:
#                 try:
#                     content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
#                     analyzer = JavaCodeAnalyzer(content)
#                     file_results[file.name] = analyzer.generate_recommendations()
#                 except Exception as e:
#                     file_results[file.name] = f"Error analyzing file: {str(e)}"
#             recommendations['files'] = file_results
#
#         # Group recommendations by line
#         grouped_recommendations = {}
#         if isinstance(recommendations, list):  # For pasted code
#             for rec in recommendations:
#                 line = rec.get('line', 'unknown')
#                 grouped_recommendations.setdefault(line, []).append({
#                     'rule': rec.get('rule'),
#                     'message': rec.get('message'),
#                 })
#         elif isinstance(recommendations, dict):  # For files
#             grouped_recommendations = recommendations
#
#         # return JsonResponse({'recommendations': grouped_recommendations})
#         return render(request, 'java_code_analysis.html', {'recommendations': grouped_recommendations})
#     return render(request, 'java_code_analysis.html')


@api_view(['GET', 'POST'])
def java_code_analysis(request):
    recommendations = {}
    code = ""

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()  # Keep the submitted code
        if code:
            try:
                analyzer = JavaCodeAnalyzer(code)
                recommendations = analyzer.generate_recommendations()
            except Exception as e:
                recommendations = [{'error': f"Error analyzing pasted code: {str(e)}"}]

        # Handle uploaded files
        files = request.FILES.getlist('files')
        if files:
            file_results = {}
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = JavaCodeAnalyzer(content)
                    file_results[file.name] = analyzer.generate_recommendations()
                except Exception as e:
                    file_results[file.name] = [{'error': f"Error analyzing file: {str(e)}"}]
            recommendations['files'] = file_results

        # Group recommendations
        grouped_recommendations = {}
        if isinstance(recommendations, list):  # For pasted code
            for rec in recommendations:
                line = rec.get('line', 'unknown')
                grouped_recommendations.setdefault(line, []).append({
                    'rule': rec.get('rule'),
                    'message': rec.get('message'),
                })
        elif isinstance(recommendations, dict):  # For files
            for filename, recs in recommendations['files'].items():
                grouped_recommendations[filename] = []
                for rec in recs:
                    grouped_recommendations[filename].append({
                        'rule': rec.get('rule'),
                        'message': rec.get('message'),
                        'line': rec.get('line', 'unknown'),
                    })

        return render(
            request,
            'java_code_analysis.html',
            {'recommendations': grouped_recommendations, 'code': code}
        )

    return render(request, 'java_code_analysis.html', {'code': code})





# @api_view(['GET', 'POST'])
# def js_code_analyser(request):
#     if request.method == 'POST':
#         recommendations = {}
#
#         # Handle pasted code
#         code = request.POST.get('code', '').strip()
#         if code:
#             try:
#                 analyzer = JavaScriptCodeAnalyzer(code)
#                 recommendations = analyzer.generate_recommendations()
#             except Exception as e:
#                 return JsonResponse({'error': f"Error analyzing pasted code: {str(e)}"}, status=500)
#
#         # Handle uploaded files
#         files = request.FILES.getlist('files')
#         if files:
#             file_results = {}
#             for file in files:
#                 try:
#                     content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
#                     analyzer = JavaScriptCodeAnalyzer(content)
#                     file_results[file.name] = analyzer.generate_recommendations()
#                 except Exception as e:
#                     file_results[file.name] = f"Error analyzing file: {str(e)}"
#             recommendations['files'] = file_results
#
#         # Group recommendations by line
#         grouped_recommendations = {}
#         if isinstance(recommendations, list):  # For pasted code
#             for rec in recommendations:
#                 line = rec.get('line', 'unknown')
#                 grouped_recommendations.setdefault(line, []).append({
#                     'rule': rec.get('rule'),
#                     'message': rec.get('message'),
#                 })
#         elif isinstance(recommendations, dict):  # For files
#             grouped_recommendations = recommendations
#
#         return JsonResponse({'recommendations': grouped_recommendations})
#
#     return render(request, 'js_code_analyser.html')

@api_view(['GET', 'POST'])
def js_code_analyser(request):
    recommendations = {}
    code = ""  # Initialize code to maintain state

    if request.method == 'POST':
        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = JavaScriptCodeAnalyzer(code)
                recommendations = analyzer.generate_recommendations()
            except Exception as e:
                recommendations = [{'error': f"Error analyzing pasted code: {str(e)}"}]

        # Handle uploaded files
        files = request.FILES.getlist('files')
        if files:
            file_results = {}
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = JavaScriptCodeAnalyzer(content)
                    file_results[file.name] = analyzer.generate_recommendations()
                except Exception as e:
                    file_results[file.name] = [{'error': f"Error analyzing file: {str(e)}"}]
            recommendations['files'] = file_results

        # Group recommendations by line or filename
        grouped_recommendations = {}
        if isinstance(recommendations, list):  # For pasted code
            for rec in recommendations:
                line = rec.get('line', 'unknown')
                grouped_recommendations.setdefault(line, []).append({
                    'rule': rec.get('rule'),
                    'message': rec.get('message'),
                })
        elif isinstance(recommendations, dict):  # For files
            grouped_recommendations = recommendations

        return render(
            request,
            'js_code_analyser.html',
            {'recommendations': grouped_recommendations, 'code': code}
        )

    return render(request, 'js_code_analyser.html', {'code': code})

@api_view(['GET', 'POST'])
def php_code_analyser(request):
    if request.method == 'POST':
        recommendations = {}

        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = PHPCodeAnalyzer(code)
                recommendations = analyzer.generate_recommendations()
            except Exception as e:
                return JsonResponse({'error': f"Error analyzing pasted code: {str(e)}"}, status=500)

        # Handle uploaded files
        files = request.FILES.getlist('files')
        if files:
            file_results = {}
            for file in files:
                try:
                    content = file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    analyzer = PHPCodeAnalyzer(content)
                    file_results[file.name] = analyzer.generate_recommendations()
                except Exception as e:
                    file_results[file.name] = f"Error analyzing file: {str(e)}"
            recommendations['files'] = file_results

        # Group recommendations by line
        grouped_recommendations = {}
        if isinstance(recommendations, list):  # For pasted code
            for rec in recommendations:
                line = rec.get('line', 'unknown')
                grouped_recommendations.setdefault(line, []).append({
                    'rule': rec.get('rule'),
                    'message': rec.get('message'),
                })
        elif isinstance(recommendations, dict):  # For files
            grouped_recommendations = recommendations

        return JsonResponse({'recommendations': grouped_recommendations})

    return render(request, 'php_code_analyser.html')


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


@api_view(['GET', 'POST'])
def analyze_code_view(request):
    """
    Analyze the given code snippet(s) for vulnerabilities using the trained model.
    """
    suggestions = []
    code_snippet = ""

    if request.method == 'POST':
        # Get the code snippet(s) from the form
        code_snippet = request.POST.get('code', '').strip()
        print("=== Received Code ===")
        print(code_snippet)

        if not code_snippet:
            message = "No code provided for analysis."
            print(f"ERROR: {message}")
            suggestions.append({"code": "", "suggestion": message})
        else:
            try:
                # Normalize and split code into individual snippets
                try:
                    print("\n=== Normalizing and Splitting Code ===")
                    snippets = split_code_snippets(code_snippet)
                    print(f"Extracted {len(snippets)} snippet(s):")
                    for i, snippet in enumerate(snippets, 1):
                        print(f"Snippet {i}:\n{snippet}\n")
                except Exception as parse_error:
                    error_message = f"Error during code parsing: {str(parse_error)}. Check for syntax or indentation issues."
                    print(f"ERROR: {error_message}")
                    suggestions.append({"code": "", "suggestion": error_message})
                    snippets = []

                if not snippets:
                    message = "No valid code snippets were extracted. Ensure the code is properly formatted."
                    print(f"WARNING: {message}")
                    suggestions.append({"code": "", "suggestion": message})
                else:
                    # Load model and tokenizer
                    print("\n=== Loading Model and Tokenizer ===")
                    model_path = "./models/custom_seq2seq_model"  # Adjust path if needed
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = T5ForConditionalGeneration.from_pretrained(model_path)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    print("Model and tokenizer loaded successfully.")

                    # Process each snippet
                    for snippet in snippets:
                        try:
                            print(f"\n=== Processing Snippet ===\n{snippet}")
                            # Tokenize the individual snippet
                            inputs = tokenizer(
                                snippet,
                                truncation=True,
                                padding="max_length",
                                max_length=512,
                                return_tensors="pt"
                            )
                            inputs = {key: value.to(device) for key, value in inputs.items()}
                            print(f"Tokenized Inputs: {inputs}")

                            # Generate suggestion for the snippet
                            model.eval()
                            with torch.no_grad():
                                outputs = model.generate(
                                    inputs["input_ids"],
                                    max_length=256,
                                    num_beams=5,
                                    early_stopping=True
                                )
                                print(f"Raw Model Output: {outputs}")

                                # Decode the suggestion
                                suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                print(f"Suggestion: {suggestion}")

                            # Append the result as a dictionary
                            suggestions.append({"code": snippet, "suggestion": suggestion})
                        except Exception as snippet_error:
                            error_message = f"Error processing snippet: {str(snippet_error)}"
                            print(f"ERROR: {error_message}")
                            suggestions.append({"code": snippet, "suggestion": error_message})
            except Exception as e:
                error_message = f"Error analyzing the code: {str(e)}"
                print(f"ERROR: {error_message}")
                suggestions.append({"code": "", "suggestion": error_message})

    print("\n=== Final Suggestions ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"Suggestion {i}: {suggestion}")

    return render(
        request,
        'analyze_code.html',
        {'code': code_snippet, 'suggestions': suggestions}
    )


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

@api_view(['GET', 'PUT' ,'DELETE'])
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
        result= calculate_code_complexity_line_by_line(code)
        complexity = result['line_complexities']
        cbo = result['cbo']

        # Render the result in the template
        return render(request, 'complexityA_table.html', {'complexities': complexity, 'cbo':cbo})

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


@api_view(['GET','POST'])
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
            for method_name,method_data in method_complexities.items():
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



# @api_view(['GET', 'POST'])
# def calculate_complexity(request):
#     if request.method == 'POST':
#         uploaded_file = request.FILES['file']  # Assuming a form is used for file upload
#
#         # Check if the file is a CSV
#         if uploaded_file.name.endswith('.csv'):
#             # Define the media directory path
#             media_dir = os.path.join(settings.BASE_DIR, 'media')
#
#             # Create the media directory if it doesn't exist
#             if not os.path.exists(media_dir):
#                 os.makedirs(media_dir)
#
#             # Save the uploaded file in the media directory
#             file_path = os.path.join(media_dir, uploaded_file.name)
#             with open(file_path, 'wb+') as destination:
#                 for chunk in uploaded_file.chunks():
#                     destination.write(chunk)
#
#             # Load the CSV into a Pandas DataFrame
#             data = pd.read_csv(file_path)
#
#             # Perform the analysis
#             pearson_corr, pearson_p_value = pearsonr(data['WCC'], data['Complexity'])
#             spearman_corr, spearman_p_value = spearmanr(data['WCC'], data['Complexity'])
#
#             # Generate correlation matrix
#             correlation_matrix = data[['WCC', 'Maintainability', 'Readability', 'Complexity']].corr()
#
#             # Save the heatmap as an image
#             sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#             heatmap_path = os.path.join(media_dir, 'heatmap.png')
#             plt.title('Correlation Matrix between WCC and Developer Ratings')
#             plt.savefig(heatmap_path)
#             plt.close()  # Close the plot after saving to avoid memory leaks
#
#             # Establish thresholds for complexity based on WCC scores
#             thresholds = data['WCC'].quantile([0.33, 0.66])
#
#             # Save thresholds to a JSON file
#             thresholds_file = os.path.join(media_dir, 'thresholds.json')
#             with open(thresholds_file, 'w') as json_file:
#                 json.dump({
#                     'threshold_low': thresholds[0.33],
#                     'threshold_medium': thresholds[0.66]
#                 }, json_file)
#
#             # Scatter plot of WCC vs Developer Complexity Ratings
#             plt.figure(figsize=(10, 6))
#             plt.scatter(data['WCC'], data['Complexity'], alpha=0.5, c='blue')
#             plt.title('WCC vs Developer Complexity Ratings')
#             plt.xlabel('WCC Scores')
#             plt.ylabel('Developer Complexity Ratings')
#             scatter_plot_path = os.path.join(media_dir, 'scatter_plot.png')
#             plt.savefig(scatter_plot_path)
#             plt.close()
#
#             # Perform linear regression
#             X = data['WCC']
#             y = data['Complexity']
#             X = sm.add_constant(X)
#             model = sm.OLS(y, X).fit()
#
#             # Prepare the results
#             context = {
#                 'pearson_corr': pearson_corr,
#                 'spearman_corr': spearman_corr,
#                 'heatmap_path': heatmap_path,
#                 'scatter_plot_path': scatter_plot_path,
#                 'regression_summary': model.summary().as_text(),  # Regression results as text
#                 'threshold_low': thresholds[0.33],
#                 'threshold_medium': thresholds[0.66]
#             }
#
#             # Render the result page with the context
#             return render(request, 'analysis_result.html', context)
#
#         else:
#             return HttpResponse("Invalid file format. Please upload a CSV file.")
#
#     # If it's a GET request, render the upload form page
#     return render(request, 'upload.html')

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
