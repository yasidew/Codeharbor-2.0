import json
import os

import pandas as pd
from datetime import datetime

from django.conf import settings
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from Drinks.models import Drink
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
from challenges.models import Challenges

def home(request):
    return render(request, 'home.html')

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


