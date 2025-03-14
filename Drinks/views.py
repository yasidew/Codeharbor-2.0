import json
import os

import numpy as np
import torch

import pandas as pd
from django.conf import settings
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from javalang.parser import JavaSyntaxError
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.cluster import KMeans
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import analysis
from Drinks.models import Drink, MethodComplexity, JavaFile, CSharpFile, CSharpMethodComplexity
from analysis.code_analyzer import CodeAnalyzer
from analysis.java_code_analyser import JavaCodeAnalyzer
from analysis.python_code_analyser import PythonCodeAnalyzer
from .serializers import DrinkSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .complexity_calculator import calculate_code_complexity_multiple_files, ai_recommend_refactoring, \
    calculate_code_complexity_by_method
from .complexity_calculator import calculate_code_complexity_line_by_line
from .complexity_calculator_csharp import calculate_code_complexity_multiple_files_csharp
from django.shortcuts import render
from prettytable import PrettyTable
import statsmodels.api as sm



# @api_view(['GET', 'POST'])
# def python_code_analysis(request):
#     if request.method == 'POST':
#         try:
#             code = request.POST.get('code', '')
#             if not code:
#                 return JsonResponse({'error': 'No code provided'}, status=400)
#
#             analyser = PythonCodeAnalyzer(code)
#             recommendations = analyser.generate_recommendations()
#
#             # Group recommendations by line number for better readability
#             grouped_recommendations = {}
#             for rec in recommendations:
#                 line = rec.get('line', 'unknown')
#                 grouped_recommendations.setdefault(line, []).append({
#                     'rule': rec.get('rule'),
#                     'message': rec.get('message'),
#                 })
#
#             return JsonResponse({'recommendations': grouped_recommendations})
#
#         except Exception as e:
#             # Log the exception message
#             print(f"Exception: {e}")
#             return JsonResponse({'error': str(e)}, status=500)
#
#     return render(request, 'python_code_analysis.html')
def home(request):
    return render(request, 'home.html')

@api_view(['GET', 'POST'])
def python_code_analysis(request):
    if request.method == 'POST':
        recommendations = {}

        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = PythonCodeAnalyzer(code)
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
                    analyzer = PythonCodeAnalyzer(content)
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

    return render(request, 'python_code_analysis.html')



@api_view(['GET', 'POST'])
def java_code_analysis(request):
    if request.method == 'POST':
        recommendations = {}

        # Handle pasted code
        code = request.POST.get('code', '').strip()
        if code:
            try:
                analyzer = JavaCodeAnalyzer(code)
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
                    analyzer = JavaCodeAnalyzer(content)
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

    return render(request, 'java_code_analysis.html')


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
    thresholds_file = os.path.join(settings.BASE_DIR, 'media', 'threshold4.json')
    if os.path.exists(thresholds_file):
        with open(thresholds_file, 'r') as json_file:
            thresholds = json.load(json_file)
        return thresholds
    # Default values if the file doesn't exist
    return {'threshold_low': 10, 'threshold_medium': 20}


@api_view(['GET', 'POST'])
def calculate_complexity_multiple_java_files(request):
    if request.method == 'POST':
        try:
            # Expecting multiple Java files in the request
            files = request.FILES.getlist('files')
            if not files:
                return Response({'error': 'No files uploaded'}, status=status.HTTP_400_BAD_REQUEST)

            file_contents = {file.name: file.read().decode('utf-8') for file in files}

            # Load thresholds from the JSON file
            thresholds = get_thresholds()
            print("thresholds<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>", thresholds)
            threshold_low = thresholds.get('threshold_low', 10)
            # threshold_medium = thresholds.get('threshold_medium', 20)
            threshold_high = thresholds.get('threshold_high', 50)

            # Calculate complexity for multiple Java files
            result, cbo_predictions = calculate_code_complexity_multiple_files(file_contents)

            complexities, cbo_summary = [], []
            results_table = PrettyTable()
            results_table.field_names = ["Filename", "Line Number", "Line", "Size", "Tokens",
                                         "Control Structure Complexity", "Nesting Weight",
                                         "Inheritance Weight", "Compound Condition Weight",
                                         "Try-Catch Weight", "Thread Weight", "CBO", "Total Complexity"]

            mp_cbo_table = PrettyTable()
            mp_cbo_table.field_names = ["Filename", "CBO", "MPC"]
            # saved_files = []

            # Process complexity results
            for filename, file_data in result.items():
                complexity_data = file_data.get('complexity_data', [])
                cbo, mpc = file_data.get('cbo', 0), file_data.get('mpc', 0)
                method_complexities = file_data.get('method_complexities', {})
                recommendations = file_data.get('recommendation', [])
                pie_chart_path = file_data.get('pie_chart_path', '')
                total_wcc = file_data.get('total_wcc', 0)
                bar_charts = file_data.get('bar_charts', {})

                # java_file, created = JavaFile.objects.update_or_create(
                #     filename=filename,
                #     defaults={'total_wcc': file_data.get('total_wcc', 0)}
                # )
                # saved_files.append({'filename': filename, 'total_wcc': java_file.total_wcc})

                java_code = file_contents.get(filename, "")

                save_complexity_to_db(filename, java_code, total_wcc, method_complexities)

                for line_data in complexity_data:
                    if len(line_data) == 12:
                        results_table.add_row([filename] + line_data)
                    else:
                        print(f"Skipping malformed data for {filename}: {line_data}")

                # Categorize methods based on complexity
                categorized_methods = []
                for method_name, method_data in method_complexities.items():
                    if isinstance(method_data, dict):
                        total_complexity = method_data.get('total_complexity', 0)
                        if total_complexity <= threshold_low:
                            category = 'Low'
                        elif threshold_low <= total_complexity <= threshold_high:
                            category = 'Medium'
                        else:
                            category = 'High'

                        categorized_methods.append({
                            **method_data,
                            'category': category,
                            'method_name': method_name,
                            'bar_chart': bar_charts.get(method_name, '')
                        })
                    else:
                        print(f"Unexpected format in method_data: {method_data}")

                complexities.append({
                    'filename': filename,
                    'complexity_data': complexity_data,
                    'cbo': cbo,
                    'mpc': mpc,
                    'method_complexities': categorized_methods,
                    'recommendations': recommendations,
                    'pie_chart_path': pie_chart_path,
                    'total_wcc': total_wcc
                })

            # Extract CBO Predictions & Recommendations
            for filename, prediction_data in cbo_predictions.items():
                cbo_summary.append({
                    'filename': filename,
                    'prediction': prediction_data.get('prediction', 'Unknown'),
                    'recommendations': prediction_data.get('recommendations', [])
                })

            # Return JSON if requested
            if request.headers.get('Accept') == 'application/json':
                return Response({
                    'complexities': complexities,
                    'cbo_predictions': cbo_summary
                }, status=status.HTTP_200_OK)

            # Render template for web-based UI
            return render(request, 'complexity_table.html',
                          {'complexities': complexities, 'cbo_predictions': cbo_summary})


        except JavaSyntaxError as e:
            return Response({
                'error': 'Java Syntax Error detected. Please correct your Java code.',
                'details': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({
                'error': 'An unexpected error occurred.',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    complexities = JavaFile.objects.prefetch_related("methods").all()

    # Structure data for rendering in the template
    complexity_data = []
    for java_file in complexities:
        file_data = {
            "filename": java_file.filename,
            "total_wcc": java_file.total_wcc,
            "method_complexities": [
                {
                    "method_name": method.method_name,
                    "total_complexity": method.total_complexity,
                    "size": method.size,
                    "control_structure_complexity": method.control_structure_complexity,
                    "nesting_weight": method.nesting_weight,
                    "inheritance_weight": method.inheritance_weight,
                    "compound_condition_weight": method.compound_condition_weight,
                    "try_catch_weight": method.try_catch_weight,
                    "thread_weight": method.thread_weight,
                    "cbo_weight": method.cbo_weight,
                    "category": method.category,
                }
                for method in java_file.methods.all()
            ],
        }
        complexity_data.append(file_data)

    # Render form for GET requests
    return render(request, 'complexity_form.html', {"complexities": complexity_data})


def save_complexity_to_db(filename, java_code, total_wcc, method_complexities):
    try:
        # Create or update JavaFile entry
        java_file, created = JavaFile.objects.update_or_create(
            filename=filename,
            defaults={'java_code': java_code, 'total_wcc': total_wcc}
        )

        # Remove old method complexities (if updating)
        if not created:
            java_file.methods.all().delete()

        # Store new method complexities
        for method_name, method_data in method_complexities.items():
            MethodComplexity.objects.create(
                java_file=java_file,
                method_name=method_name,
                total_complexity=method_data.get('total_complexity', 0),
                category=method_data.get('category', 'Low'),

                # Store additional complexity metrics
                size=method_data.get('size', 0),
                control_structure_complexity=method_data.get('control_structure_complexity', 0),
                nesting_weight=method_data.get('nesting_level', 0),
                inheritance_weight=method_data.get('inheritance_level', 0),
                compound_condition_weight=method_data.get('compound_condition_weight', 0),
                try_catch_weight=method_data.get('try_catch_weight', 0),
                thread_weight=method_data.get('thread_weight', 0),
                cbo_weight=method_data.get('cbo_weights', 0),
            )

    except Exception as e:
        print(f"Error saving complexity data for {filename}: {e}")


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
        print("thresholds.....]]]]]]]]]]]]]]]]]]]", thresholds)
        threshold_low = thresholds.get('threshold_low', 10)
        threshold_medium = thresholds.get('threshold_medium', 20)
        threshold_high = thresholds.get('threshold_high', 50)

        # Call your function to calculate complexity for multiple files
        result, cbo_predictions = calculate_code_complexity_multiple_files_csharp(file_contents)

        # Prepare a list to store complexities for each file
        complexities = []
        cbo_summary = []

        # Create a table for the response
        results_table = PrettyTable()
        results_table.field_names = ["Filename", "Line Number", "Line", "Size", "Tokens",
                                     "Control Structure Complexity", "Nesting Weight",
                                     "Inheritance Weight", "Compound Condition Weight",
                                     "Try-Catch Weight", "Thread Weight", "CBO", "Total Complexity"]

        # Prepare another table for displaying MPC and CBO values
        mp_cbo_table = PrettyTable()
        mp_cbo_table.field_names = ["Filename", "CBO", "MPC"]

        # Collect complexities for each file
        for filename, file_data in result.items():
            complexity_data = file_data['complexity_data']
            cbo = file_data['cbo']
            mpc = file_data['mpc']
            method_complexities = file_data['method_complexities']
            recommendations = file_data['recommendation']
            pie_chart_path = file_data['pie_chart_path']
            total_wcc = file_data['total_wcc']
            bar_charts = file_data.get('bar_charts', {})

            save_complexity_to_db_csharp(filename, file_contents[filename], total_wcc, method_complexities)

            for line_data in complexity_data:
                results_table.add_row([filename] + line_data)  # Now line_data has 9 values

            # # Categorize each method based on total complexity
            categorized_methods = []
            for method_name, method_data in method_complexities.items():
                if isinstance(method_data, dict):
                    total_complexity = method_data.get('total_complexity', 0)

                    if total_complexity <= threshold_low:
                        category = 'Low'
                    elif threshold_low <= total_complexity <= threshold_high:
                        category = 'Medium'
                    else:
                        category = 'High'

                    # Append category to the method data
                    # categorized_method = method_data.copy()
                    # categorized_method['category'] = category
                    # categorized_method['method_name'] = method_name
                    # categorized_methods.append(categorized_method)

                    categorized_methods.append({
                        **method_data,
                        'category': category,
                        'method_name': method_name,
                        'bar_chart': bar_charts.get(method_name, '')
                    })
                else:
                    print(f"Unexpected format in method_data: {method_data}")

                # print("categorized_method", categorized_method)
            complexities.append({
                'filename': filename,
                'complexity_data': complexity_data,
                'cbo': cbo,
                'mpc': mpc,
                'method_complexities': categorized_methods,
                'recommendations': recommendations,
                'pie_chart_path': pie_chart_path,
                'total_wcc': total_wcc
            })

        # Log the result table for debugging or reference
        # print(results_table)
        for filename, prediction_data in cbo_predictions.items():
            cbo_summary.append({
                'filename': filename,
                'prediction': prediction_data['prediction'],
                'recommendations': prediction_data['recommendations']
            })
        if request.headers.get('Accept') == 'application/json':
            return Response({
                'complexities': complexities,
                'cbo_predictions': cbo_summary
            }, status=status.HTTP_200_OK)

        # Instead of returning a JSON response, render the template and pass complexities
        return render(request, 'complexityC_table.html', {'complexities': complexities, 'cbo_predictions': cbo_summary})

    csharp_files = CSharpFile.objects.all()
    complexities = []

    for file in csharp_files:
        methods = CSharpMethodComplexity.objects.filter(csharp_file=file)
        method_list = []
        for method in methods:
            method_list.append({
                'method_name': method.method_name,
                'total_complexity': method.total_complexity,
                'size': method.size,
                'control_structure_complexity': method.control_structure_complexity,
                'nesting_weight': method.nesting_weight,
                'inheritance_weight': method.inheritance_weight,
                'compound_condition_weight': method.compound_condition_weight,
                'try_catch_weight': method.try_catch_weight,
                'thread_weight': method.thread_weight,
                'cbo_weight': method.cbo_weight,
                'category': method.category,
            })

        complexities.append({
            'filename': file.filename,
            'total_wcc': file.total_wcc,
            'method_complexities': method_list
        })

    # If GET request, just show the form
    return render(request, 'complexityC_form.html', {'complexities': complexities})

def save_complexity_to_db_csharp(filename, csharp_code, total_wcc, method_complexities):
    """Save C# complexity results to the database while preventing duplicate method names."""
    try:
        # Update or create the CSharpFile record
        csharp_file, created = CSharpFile.objects.update_or_create(
            filename=filename,
            defaults={'csharp_code': csharp_code, 'total_wcc': total_wcc}
        )

        # Delete old methods before inserting new ones (to avoid duplicates)
        CSharpMethodComplexity.objects.filter(csharp_file=csharp_file).delete()

        for method_name, method_data in method_complexities.items():
            if isinstance(method_data, dict):
                # Ensure valid data before saving
                CSharpMethodComplexity.objects.update_or_create(
                    csharp_file=csharp_file,
                    method_name=method_name,  # Ensures uniqueness per file
                    defaults={
                        'total_complexity': method_data.get('total_complexity', 0),
                        'size': method_data.get('size', 0),
                        'control_structure_complexity': method_data.get('control_structure_complexity', 0),
                        'nesting_weight': method_data.get('nesting_level', 0),
                        'inheritance_weight': method_data.get('inheritance_level', 0),
                        'compound_condition_weight': method_data.get('compound_condition_weight', 0),
                        'try_catch_weight': method_data.get('try_catch_weight', 0),
                        'thread_weight': method_data.get('thread_weight', 0),
                        'cbo_weight': method_data.get('cbo_weights', 0),
                        'category': method_data.get('category', 'Low'),
                    }
                )
    except Exception as e:
        print(f"Error saving complexity data for {filename}: {e}")

@api_view(['GET', 'POST'])
def calculate_complexity(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']

        if uploaded_file.name.endswith('.csv'):
            # Define the media directory path
            media_dir = os.path.join(settings.BASE_DIR, 'media')
            os.makedirs(media_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Save the uploaded file
            file_path = os.path.join(media_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Load the CSV into a DataFrame
            data = pd.read_csv(file_path)

            # Columns to analyze
            target_columns = ['Complexity', 'Maintainability', 'Readability']

            # Initialize correlation results
            pearson_results, spearman_results = {}, {}

            # Perform correlation analysis
            for column in target_columns:
                pearson_corr, pearson_p_value = pearsonr(data['WCC'], data[column])
                spearman_corr, spearman_p_value = spearmanr(data['WCC'], data[column])

                pearson_results[column] = {
                    'correlation': round(pearson_corr, 4),
                    'p_value': round(pearson_p_value, 4)
                }
                spearman_results[column] = {
                    'correlation': round(spearman_corr, 4),
                    'p_value': round(spearman_p_value, 4)
                }

            # Generate Correlation Matrix
            correlation_matrix = data[['WCC', 'Complexity', 'Maintainability', 'Readability']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            heatmap_path = os.path.join(media_dir, 'heatmap.png')
            plt.title('Correlation Matrix')
            plt.savefig(heatmap_path)
            plt.close()

            # Clustering for WCC thresholds
            wcc_values = np.array([101, 191, 205, 246, 291, 293, 298, 344, 346, 382,
                                   170, 61, 108, 183, 204, 153, 305, 270, 137, 201,
                                   345, 73, 114, 153]).reshape(-1, 1)

            plt.figure(figsize=(10, 6))
            sns.histplot(wcc_values, bins=10, kde=True, color='blue')

            # Labels and title
            plt.xlabel("WCC Values")
            plt.ylabel("Frequency")
            plt.title("WCC Distribution Histogram")
            plt.grid(True)
            histogram_path = os.path.join(media_dir, 'wcc_distribution.png')
            plt.savefig(histogram_path)
            plt.close()

            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(wcc_values)
            data['Cluster_Level'] = kmeans.predict(data[['WCC']])

            # Map clusters to categories based on cluster centers
            cluster_centers = sorted(kmeans.cluster_centers_.flatten())
            low_center, high_center = cluster_centers[0], cluster_centers[1]
            data['Cluster_Level'] = data['Cluster_Level'].map({
                0: 'Low Complexity',
                1: 'High Complexity'
            })

            # Save thresholds to JSON
            thresholds = {
                'threshold_low': round(low_center, 2),
                # 'threshold_medium': round(medium_center, 2),
                'threshold_high': round(high_center, 2)
            }

            # Path to save the thresholds
            threshold_file_path = os.path.join(media_dir, 'threshold4.json')
            with open(threshold_file_path, 'w') as json_file:
                json.dump(thresholds, json_file)

            # Scatter plots for each metric
            scatter_plots = {}
            for column in target_columns:
                plt.figure(figsize=(10, 6))
                sns.regplot(x=data['WCC'], y=data[column], scatter=True, ci=None, line_kws={'color': 'red'})
                plt.title(f'WCC vs {column}')
                plt.xlabel('WCC')
                plt.ylabel(column)
                scatter_plot_path = os.path.join(media_dir, f'scatter_{column}.png')
                plt.savefig(scatter_plot_path)
                plt.close()
                scatter_plots[column] = scatter_plot_path

                # Boxplot to show WCC thresholds with KMeans cluster thresholds
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Cluster_Level', y='WCC', data=data)

                # Add horizontal lines for KMeans thresholds
                plt.axhline(y=low_center, color='red', linestyle='--', label=f'Low Threshold ({round(low_center, 2)})')
                # plt.axhline(y=medium_center, color='orange', linestyle='--',
                #             label=f'Medium Threshold ({round(medium_center, 2)})')
                plt.axhline(y=high_center, color='green', linestyle='--',
                            label=f'High Threshold ({round(high_center, 2)})')

                plt.title('WCC by Complexity Level with KMeans Thresholds')
                plt.ylabel('WCC')
                plt.xlabel('Cluster Level')
                plt.legend()

                # Save the updated boxplot
                boxplot_path = os.path.join(media_dir, 'boxplot_with_thresholds.png')
                plt.savefig(boxplot_path)
                plt.close()

            # Prepare context for the result page
            context = {
                'pearson_results': pearson_results,
                'spearman_results': spearman_results,
                'scatter_plots': scatter_plots,
                'low_center': round(low_center, 2),
                # 'medium_center': round(medium_center, 2),
                'high_center': round(high_center, 2),
                'kmeans_centers': cluster_centers,
                'plot_path': 'scatter_Complexity.png',
                'heatmap_path': 'heatmap.png',
            }

            # Render the result page
            return render(request, 'analysis_result.html', context)

        else:
            return HttpResponse("Invalid file format. Please upload a CSV file.")

    return render(request, 'upload.html')


def guidelines_view(request):
    guidelines = [
        {
            "title": "Weight Due to Type of Control Structures (Wc)",
            "description": "Control structures influence code complexity by introducing decision paths.",
            "table": [
                {"structure": "Sequential Statements", "weight": 0},
                {"structure": "Branching (if, else, else if)", "weight": 1},
                {"structure": "Loops (for, while, do-while)", "weight": 2},
                {"structure": "Switch statement with n cases", "weight": "n"},
            ],
            "examples": {
                "Java": """
                    public class ControlExample {
                        public static void main(String[] args) {
                            int num = 10;

                            // Branching (if-else) - Weight: 1
                            if (num > 0) {
                                System.out.println("Positive number");
                            } else {
                                System.out.println("Negative number");
                            }

                            // Loop (for) - Weight: 2
                            for (int i = 0; i < 5; i++) {
                                System.out.println(i);
                            }

                            // Switch statement (3 cases) - Weight: 3
                            switch (num) {
                                case 5: System.out.println("Five"); break;
                                case 10: System.out.println("Ten"); break;
                                default: System.out.println("Other"); break;
                            }
                        }
                    }
                    """,
                "Csharp": """
                    using System;

                    class ControlExample {
                        static void Main() {
                            int num = 10;

                            // Branching (if-else) - Weight: 1
                            if (num > 0)
                                Console.WriteLine("Positive number");
                            else
                                Console.WriteLine("Negative number");

                            // Loop (while) - Weight: 2
                            int i = 0;
                            while (i < 5) {
                                Console.WriteLine(i);
                                i++;
                            }

                            // Switch statement (3 cases) - Weight: 3
                            switch (num) {
                                case 5: Console.WriteLine("Five"); break;
                                case 10: Console.WriteLine("Ten"); break;
                                default: Console.WriteLine("Other"); break;
                            }
                        }
                    }
                    """
            }
        },
        {
            "title": "Weight Due to Nesting Level of Control Structures (Wn)",
            "description": "Nesting increases complexity as deeply nested statements are harder to understand.",
            "table": [
                {"structure": "Outermost statements", "weight": 1},
                {"structure": "Second level", "weight": 2},
                {"structure": "nth level", "weight": "n"},
            ],
            "examples": {
                "Java": """
    public class NestingExample {
        public static void main(String[] args) {
            int num = 5;

            if (num > 0) {  // Level 1 - Weight: 1
                for (int i = 0; i < 3; i++) {  // Level 2 - Weight: 2
                    while (num > 0) {  // Level 3 - Weight: 3
                        System.out.println(num);
                        num--;
                    }
                }
            }
        }
    }
                """,
                "Csharp": """
    using System;

    class NestingExample {
        static void Main() {
            int num = 5;

            if (num > 0) { // Level 1 - Weight: 1
                for (int i = 0; i < 3; i++) { // Level 2 - Weight: 2
                    while (num > 0) { // Level 3 - Weight: 3
                        Console.WriteLine(num);
                        num--;
                    }
                }
            }
        }
    }
                """
            }
        },
        {
            "title": "Weight Due to Inheritance Level of Statements (Wi)",
            "description": "Statements in derived classes increase complexity as they depend on parent classes.",
            "table": [
                {"structure": "Base class", "weight": 1},
                {"structure": "First-level derived class", "weight": 2},
                {"structure": "nth level derived class", "weight": "n"},
            ],
            "examples": {
                "Java": """
    class Animal {  // Base class - Weight: 1
        void makeSound() {
            System.out.println("Animal sound"); 
        }
    }

    class Dog extends Animal { // First-level derived - Weight: 2
        @Override
        void makeSound() { 
            System.out.println("Bark");
        }
    }

    class Puppy extends Dog { // Second-level derived - Weight: 3
        @Override
        void makeSound() { 
            System.out.println("Small Bark");
        }
    }
                """,
                "Csharp": """
    using System;

    class Animal {  // Base class - Weight: 1
        public virtual void MakeSound() {
            Console.WriteLine("Animal sound");
        }
    }

    class Dog : Animal { // First-level derived - Weight: 2
        public override void MakeSound() { 
            Console.WriteLine("Bark");
        }
    }

    class Puppy : Dog { // Second-level derived - Weight: 3
        public override void MakeSound() { 
            Console.WriteLine("Small Bark");
        }
    }
                """
            }
        },
        {
            "title": "Coupling Between Object Classes (Wcbo)",
            "description": "CBO measures the degree of dependency between classes. High coupling increases complexity and reduces maintainability.",
            "table": [
                {"structure": "Loose coupling (e.g., dependency injection via constructor/setter)", "weight": 1},
                {
                    "structure": "Tight coupling (e.g., static method call, static variable usage, direct object instantiation)",
                    "weight": 3},
            ],
            "examples": {
                "Java": """
class Engine {
    public int rpm;
    static void start() { 
        System.out.println("Engine started");
    }
}

class Car {
    private Engine engine;

    public void setEngine(Engine engine) { // Setter Injection
        this.engine = engine; // Loose Coupling - Weight: 1
    }

    void drive() {
        engine.start();
        Engine.rpm = 1000; // Static variable usage (Tight Coupling - Weight: 3)
        System.out.println("Car is driving");
    }
}

class CarTight {
    void drive() {
        Engine.start(); // Static method call (Tight Coupling - Weight: 3)
        System.out.println("Car is using an engine of type: " + Engine.type); // Static variable usage (Tight Coupling - Weight: 3)
    }
}

                    """,
                "Csharp": """
            class Engine {

    public static void Start() { // Static method call
        Console.WriteLine("Engine started");
    }
}

class Car {
    private Engine _engine;

    public void SetEngine(Engine engine) { // Setter Injection
        _engine = engine; // Loose Coupling - Weight: 1
    }

    public void Drive() {
        _engine.Start();
        Console.WriteLine("Car is driving");
    }
}

class CarTight {
    public void Drive() {
        Engine.Start(); // Static method call (Tight Coupling - Weight: 3)
        Console.WriteLine($"Car is using an engine of type: {Engine.Type}"); // Static variable usage (Tight Coupling - Weight: 3)
    }
}
                    """
            }
        },
        {
            "title": "Weight Due to Try-Catch-Finally Blocks (Wtc)",
            "description": "Try-Catch-Finally structures influence code complexity by adding exception-handling paths. The weight is determined based on the nesting depth and control structure type.",
            "table": [
                {"structure": "try-catch",
                 "guideline": "Assigned weight based on nesting depth. Deeper nesting increases complexity.",
                 "weight": "1 (Level 1), 2 (Level 2), 3 (Level 3), 4 (Level 4+)"},
                {"structure": "finally",
                 "guideline": "Always executes, adding a mandatory execution path. Assigned a fixed weight.",
                 "weight": 1}
            ],
            "examples": {
                "Java": """
            public class ExceptionHandlingExample {
                public static void main(String[] args) {
                    try {
                        int result = 10 / 0; 
                    } catch (ArithmeticException e) { // Catch at Level 1 (Weight: 1)
                        System.out.println("Division by zero!"); 
                    }

                    try {
                        try {
                            int[] arr = {1, 2, 3};
                            System.out.println(arr[5]); 
                        } catch (ArrayIndexOutOfBoundsException e) { // Catch at Level 2 (Weight: 2)
                            System.out.println("Index out of bounds!"); 
                        }
                    } catch (Exception e) { // Catch at Level 1 (Weight: 1)
                        System.out.println("Generic exception!"); 
                    } finally { // Finally block (Weight: 1)
                        System.out.println("Execution finished."); 
                    }
                }
            }
                    """,
                "Csharp": """
            using System;

            class ExceptionHandlingExample {
                static void Main() {
                    try {
                        int result = 10 / 0; 
                    } catch (DivideByZeroException e) {  // Catch at Level 1 (Weight: 1)
                        Console.WriteLine("Division by zero!");
                    }

                    try {
                        try {
                            int[] arr = {1, 2, 3};
                            Console.WriteLine(arr[5]); 
                        } catch (IndexOutOfRangeException e) { // Catch at Level 2 (Weight: 2)
                            Console.WriteLine("Index out of bounds!"); 
                        }
                    } catch (Exception e) { // Catch at Level 1 (Weight: 1)
                        Console.WriteLine("Generic exception!"); 
                    } finally { // Finally block (Weight: 1)
                        Console.WriteLine("Execution finished."); 
                    }
                }
            }
                    """
            }
        },
        {
            "title": "Weight Due to Compound Conditional Statements(Wcc)",
            "description": "Logical operators increase complexity.",
            "table": [
                {"structure": "Simple condition", "weight": 1},
                {"structure": "Compound condition with n logical operators", "weight": "n+1"},
            ],
            "examples": {
                "Java": """
    if (age > 18) {  // Weight: 1
        System.out.println("Adult");
    }

    if (age > 18 && country.equals("USA")) {  // Weight: 2
        System.out.println("Eligible voter in the USA");
    }
                """,
                "Csharp": """
    if (age > 18) { // Weight: 1
        Console.WriteLine("Adult");
    }

    if (age > 18 && country == "USA") { // Weight: 2
        Console.WriteLine("Eligible voter in the USA");
    }
                """
            }
        },
        {
            "title": "Weight Due to Threads (Wth)",
            "description": "Multi-threading increases complexity. Thread creation and synchronization mechanisms contribute to concurrency overhead.",
            "table": [
                {
                    "structure": "Simple thread creation",
                    "guideline": "Creating a new thread increases complexity.\n\nJava:\n```java\nThread t1 = new Thread(() -> System.out.println(\"Thread running\"));\nt1.start();\n```\n\nC#:\n```csharp\nThread t1 = new Thread(() => Console.WriteLine(\"Thread running\"));\nt1.Start();\n```",
                    "weight": 2
                },
                {
                    "structure": "Basic synchronized block",
                    "guideline": "Using synchronized blocks to protect shared resources.\n\nJava:\n```java\nsynchronized (this) {\n    System.out.println(\"Synchronized block\");\n}\n```\n\nC#:\n```csharp\nobject lockObj = new object();\nlock (lockObj) {\n    Console.WriteLine(\"Synchronized block\");\n}\n```",
                    "weight": 3
                },
                {
                    "structure": "Nested synchronized block",
                    "guideline": "Synchronization inside another synchronized block increases complexity.\n\nJava:\n```java\nsynchronized (this) {\n    synchronized (this) {\n        System.out.println(\"Nested synchronized block\");\n    }\n}\n```\n\nC#:\n```csharp\nlock (lockObj) {\n    lock (lockObj) {\n        Console.WriteLine(\"Nested synchronized block\");\n    }\n}\n```",
                    "weight": 4
                },
                {
                    "structure": "Method-level synchronization",
                    "guideline": "Declaring a method as synchronized increases complexity significantly.\n\nJava:\n```java\npublic synchronized void syncMethod() {\n    System.out.println(\"Synchronized method\");\n}\n```\n\nC#:\n```csharp\nprivate static readonly object _lock = new object();\npublic void SyncMethod() {\n    lock (_lock) {\n        Console.WriteLine(\"Synchronized method\");\n    }\n}\n```",
                    "weight": 5
                }
            ],
            "examples": {
                "Java": "\nclass ThreadExample {\n    public static void main(String[] args) {\n        // Simple Thread Creation - Weight: 2\n        Thread t1 = new Thread(() -> System.out.println(\"Thread 1 running\"));\n        t1.start();\n\n        // Basic Synchronized Block - Weight: 3\n        synchronized (this) {\n            System.out.println(\"Synchronized block\");\n        }\n\n        // Nested Synchronized Block - Weight: 4\n        synchronized (this) {\n            synchronized (this) {\n                System.out.println(\"Nested synchronized block\");\n            }\n        }\n\n             new ThreadExample().syncMethod();\n    }\n\n// Method-Level Synchronization - Weight: 5\n public synchronized void syncMethod() {\n        System.out.println(\"Synchronized method\");\n    }\n}\n",
                "Csharp": "\nusing System;\nusing System.Threading;\n\nclass ThreadExample {\n    public static void Main() {\n        // Simple Thread Creation - Weight: 2\n        Thread t1 = new Thread(() => Console.WriteLine(\"Thread 1 running\"));\n        t1.Start();\n\n        // Basic Synchronized Block - Weight: 3\n        object lockObj = new object();\n        lock (lockObj) {\n            Console.WriteLine(\"Synchronized block\");\n        }\n\n        // Nested Synchronized Block - Weight: 4\n        lock (lockObj) {\n            lock (lockObj) {\n                Console.WriteLine(\"Nested synchronized block\");\n            }\n        }\n\n        // Method-Level Synchronization - Weight: 5\n        new ThreadExample().SyncMethod();\n    }\n\n    private static readonly object _lock = new object();\n    public void SyncMethod() {\n        lock (_lock) {\n            Console.WriteLine(\"Synchronized method\");\n        }\n    }\n}\n"
            }
        }

    ]

    return render(request, 'guidelines.html', {"guidelines": guidelines})