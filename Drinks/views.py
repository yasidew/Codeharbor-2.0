import json
import os

import pandas as pd
import seaborn as sns
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

from Drinks.models import Drink
from .complexity_calculator import calculate_code_complexity_line_by_line
from .complexity_calculator import calculate_code_complexity_multiple_files
from .complexity_calculator_csharp import calculate_code_complexity_multiple_files_csharp
from .serializers import DrinkSerializer


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
    thresholds_file = os.path.join(settings.BASE_DIR, 'media', 'threshold.json')
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
        print("thresholds.....]]]]]]]]]]]]]]]]]]]", thresholds)
        threshold_low = thresholds.get('threshold_low', 10)
        threshold_medium = thresholds.get('threshold_medium', 20)
        threshold_high = thresholds.get('threshold_high', 50)

        # Call your function to calculate complexity for multiple files
        result, cbo_predictions = calculate_code_complexity_multiple_files(file_contents)

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
            print("Complexity data@@@@@@@@@@@@@@@@@@@@@22", complexity_data)
            cbo = file_data['cbo']
            mpc = file_data['mpc']
            method_complexities = file_data.get('method_complexities', [])
            print("method_complexities", method_complexities)
            recommendations = file_data['recommendation']
            pie_chart_path = file_data['pie_chart_path']
            total_wcc = file_data['total_wcc']
            bar_charts = file_data.get('bar_charts', {})

            for line_data in complexity_data:
                print("line_data", line_data)
                if len(line_data) == 12:  # Ensure the row matches the expected number of columns
                    results_table.add_row([filename] + line_data)
                else:
                    print(f"Skipping malformed data for {filename}: {line_data}")

            # Categorize each method based on total complexity
            categorized_methods = []
            for method_name, method_data in method_complexities.items():
                if isinstance(method_data, dict):
                    total_complexity = method_data.get('total_complexity', 0)

                    if threshold_low <= total_complexity <= threshold_medium:
                        category = 'Low'
                    elif threshold_medium < total_complexity <= threshold_high:
                        category = 'Medium'
                    elif total_complexity > threshold_high:
                        category = 'High'
                    elif total_complexity < threshold_low:
                        category = 'Low'
                    else:
                        category = 'Unknown'

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
            # Extract CBO Predictions & Recommendations
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
        return render(request, 'complexity_table.html', {'complexities': complexities, 'cbo_predictions': cbo_summary})

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

            print("method_complexities.....]]]]]]]]]]", method_complexities)

            for line_data in complexity_data:
                results_table.add_row([filename] + line_data)  # Now line_data has 9 values

            # # Categorize each method based on total complexity
            categorized_methods = []
            for method_name, method_data in method_complexities.items():
                if isinstance(method_data, dict):
                    total_complexity = method_data.get('total_complexity', 0)

                    if threshold_low <= total_complexity <= threshold_medium:
                        category = 'Low'
                    elif threshold_medium < total_complexity <= threshold_high:
                        category = 'Medium'
                    elif total_complexity > threshold_high:
                        category = 'High'
                    elif total_complexity < threshold_low:
                        category = 'Low'
                    else:
                        category = 'Unknown'

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

    # If GET request, just show the form
    return render(request, 'complexityC_form.html')

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

            # Calculate WCC thresholds using percentiles
            # low_threshold = data['WCC'].quantile(0.33)
            # high_threshold = data['WCC'].quantile(0.67)
            #
            # # Assign percentile-based categories
            # data['Percentile_Level'] = pd.cut(
            #     data['WCC'],
            #     bins=[-float('inf'), low_threshold, high_threshold, float('inf')],
            #     labels=['Low Complexity', 'Medium Complexity', 'High Complexity']
            # )

            # Clustering for WCC thresholds
            wcc_values = data['WCC'].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=0).fit(wcc_values)
            data['Cluster_Level'] = kmeans.predict(wcc_values)

            # Map clusters to categories based on cluster centers
            cluster_centers = sorted(kmeans.cluster_centers_.flatten())
            low_center, medium_center, high_center = cluster_centers[0], cluster_centers[1], cluster_centers[2]
            data['Cluster_Level'] = data['Cluster_Level'].map({
                0: 'Low Complexity',
                1: 'Medium Complexity',
                2: 'High Complexity'
            })

            # Save thresholds to JSON
            thresholds = {
                'threshold_low': round(low_center, 2),
                'threshold_medium': round(medium_center, 2),
                'threshold_high': round(high_center, 2)
            }

            # Path to save the thresholds
            threshold_file_path = os.path.join(media_dir, 'threshold.json')
            with open(threshold_file_path, 'w') as json_file:
                json.dump(thresholds, json_file)

            # Scatter plots for each metric
            scatter_plots = {}
            for column in target_columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(data['WCC'], data[column], alpha=0.6, c='blue')
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
                plt.axhline(y=medium_center, color='orange', linestyle='--',
                            label=f'Medium Threshold ({round(medium_center, 2)})')
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
                'medium_center': round(medium_center, 2),
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
    return render(request, 'guidelines.html', {})
