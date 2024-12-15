# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "datetime",
#   "requests"
# ]
# ///

from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, zscore
import sys
from datetime import datetime
import requests
import json




def load_data(file_path):

    """Load the dataset from 'file_path'."""


    try:
        # read the dataset using pandas
        data = pd.read_csv(file_path,encoding='latin1',encoding_errors='ignore')
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        # handle exceptions if any
        print(f"Error loading dataset: {e}")
        sys.exit(1)
 


def perform_chi_square_tests(data, categorical_cols):
    """
    Performs Chi-Square tests for categorical column pairs to assess independence.
    
    Args:
        data (DataFrame): Input DataFrame with categorical columns.
        categorical_cols (Index): List of categorical columns.

    Returns:
        str: A string summarizing results of the Chi-Square statistical tests.
    """
    chi_square_results = [] # List to store Chi-Square test results
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2: # ingnore self-comparisons
                contingency_table = pd.crosstab(data[col1], data[col2])
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

                # Record the test results
                chi_square_results.append({
                    "col1": col1,
                    "col2": col2,
                    "chi2_stat": chi2_stat,
                    "p_value": p_value,
                    "dof": dof
                })
    
    # Format Chi-Square findings
    result_str = "\nChi-Square Test Results:\n"
    for result in chi_square_results:
        result_str += (f"Chi2 Statistic between {result['col1']} and {result['col2']}: {result['chi2_stat']:.2f}, "
                       f"p-value: {result['p_value']:.4f}\n")
    
    return result_str


def perform_outlier_detection(data, numerical_cols):
    """
    Detect multivariate outliers using z-scores or IQR filtering.

    Args:
        data (DataFrame): Input numerical DataFrame.
        numerical_cols (Index): List of numerical columns to analyze.

    Returns:
        dict: Outlier counts for each feature.
    """
    outliers = {} # Dictionary to store outlier counts, key by feature and value by count
    for col in numerical_cols:
        z_scores = zscore(data[col].dropna())
        outlier_count = (abs(z_scores) > 3).sum() 
        outliers[col] = outlier_count
    return outliers


def general_statistics(data):
    """Generate general statistics with advanced statistical insights."""
    
    # Select columns for analysis
    numerical_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    # Base summary statistics
    stats_summary = f"Dataset Overview:\n"
    stats_summary += f"Shape: {data.shape[0]} rows, {data.shape[1]} columns\n"
    stats_summary += f"Columns: {', '.join(data.columns)}\n"
    stats_summary += f"Total missing values: {data.isna().sum().sum()}\n"
    stats_summary += f"Unique values across categorical features:\n{data[categorical_cols].nunique().to_string()}\n"

    # Correlation summary
    numerical_cols = data.select_dtypes(include='number').columns

    numerical_data = data.loc[:,numerical_cols]
    
    stats_summary += "\nCorrelation Summary:\n"
    for num_col1 in numerical_cols:
        for num_col2 in numerical_cols:
            if num_col1 != num_col2: # Checking for self-comparisons
                stats_summary += f"- Correlation between {num_col1} and {num_col2}: {numerical_data[num_col1].corr(numerical_data[num_col2])}\n"

    # Detect multivariate outliers
    outliers_detected = perform_outlier_detection(data, numerical_cols)

    # Outlier detection summary
    stats_summary += "\nOutlier Detection (z-score > 3):\n"
    for key, value in outliers_detected.items():
        stats_summary += f"{key}: {value} outliers\n"
    
    # Perform Chi-square tests for categorical relationships
    chi_square_summary = perform_chi_square_tests(data, categorical_cols)
    
    
    # Final statistical insights
    stats_summary += chi_square_summary
    
    return stats_summary


def plot_visualizations(data):
    """
    Generate and save visualizations for numerical features using heatmaps, histograms, and pairplots.
    
    Parameters:
    data (DataFrame): Input data containing numerical and categorical features.
    
    Returns:
    List[str]: A list of saved visualization image filenames.
    """
    images = []

    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    numerical_data = data.loc[:, numerical_cols]
    categorical_data = data.loc[:, categorical_cols]

    # Generate a heatmap if multiple numerical columns exist
    if len(numerical_cols) > 1:
        # Correlation Heatmap
        corr_mat = numerical_data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.tight_layout()
        heatmap_file = "heatmap.png"
        plt.savefig(heatmap_file)
        plt.close()
        images.append(heatmap_file)

        # Histograms for numerical features
        plt.figure(figsize=(10, 8))
        numerical_data.hist(bins=20, color='skyblue', edgecolor='black')
        plt.suptitle('Histograms for Numerical Features')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle
        hist_file = "histograms.png"
        plt.savefig(hist_file)
        plt.close()
        images.append(hist_file)

        # Pairplot for numerical feature relationships
        try:
            plt.figure(figsize=(10, 8))
            sns.pairplot(numerical_data, diag_kind='kde')
            plt.suptitle('Pairplot of Numerical Features', y=1.02)  # Offset the title above
            pairplot_file = "pairplot.png"
            plt.savefig(pairplot_file)
            plt.close()
            images.append(pairplot_file)
        except Exception as e:
            print(f"Pairplot could not be generated: {e}")

    else:
        print("Not enough numerical features for heatmap or pairplot visualization.")

    return images

def write_report(stats):

    """Write the analysis results to README.md."""

    with open('README.md', 'w') as file:
        file.write("# Data Analysis Report\n")
        file.write(f"Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write("## General Statistics\n")
        file.write(stats)
    return 

def analyze_data(data):

    """Main function to load data, analyze it."""

    # Generate general statistics
    stats = general_statistics(data)

    # Write the analysis results to README.md
    write_report(stats)

    analysis = ""

    #Read the analysis results
    with open('README.md', 'r') as file:
        analysis = file.read()
        file.close()
    return analysis


def generate_story(data_analysis, df):
    """
    Generates a narrative story using LLM based on data analysis insights,
    integrates visualizations, and writes a structured markdown report into README.md.

    Args:
        data_analysis (str): A summary of the findings from data analysis.
        df (DataFrame): Input DataFrame used for visualization generation.

    Returns:
        str: The narrative story generated by the LLM.
    """
    # Load environment variables
    load_dotenv()
    AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')

    if AIPROXY_TOKEN is None:
        print("Error: AIPROXY_TOKEN not found in the .env file.")
        sys.exit(1)

    # Set up the request endpoint and headers
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Optimized function description with concise instructions
    function_descriptions = [
        {
            "name": "generate_story",
            "description": "Generates a professional narrative based on findings and data analysis insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_analysis": {
                        "type": "string",
                        "description": "Summary of key insights from the data analysis findings."
                    }
                }
            }
        }
    ]

    # Prepare a concise and structured LLM payload
    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a data analysis assistant that crafts narratives emphasizing key insights from analysis findings and send it json format."
            },
            {
                "role": "user",
                "content": f"""Generate a narrative story with subheadings and should be in professional tone that emphasizes key insights from data analysis 
                findings and send it in json format."""
            }
        ],
        "max_tokens": 400,  # Optimized token usage
        "functions": function_descriptions,
        "function_call": {"name": "generate_story", "arguments": {"data_analysis": data_analysis}}
    }

    # Send the request
    try:
        response = requests.post(url, headers=headers, json=request_payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to connect to LLM endpoint. Details: {e}")
        sys.exit(1)

    # Handle response robustly
    try:
        raw_story = response.json()['choices'][0]['message']['function_call']['arguments']
        print(raw_story)
        story = json.loads(raw_story)['data_analysis']
    except Exception as e:
        print(f"Error parsing the response from LLM: {e}")
        sys.exit(1)

    # Generate visualizations
    images = plot_visualizations(df)

    # Write the markdown report to README.md
    try:
        with open('README.md', 'w') as file:
            file.write("# Data Analysis Report\n")
            file.write(f"\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Embed visualizations into the README
            file.write("# Visualizations\n")
            for image in images:
                file.write(f"![{image}]({image})\n")

            # Write the story with improved logical structure
            file.write("\n# Narrative Story\n")
            file.write(f"{story}\n")
        
        print("Story and visualizations have been successfully generated. Check README.md for details.")
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        sys.exit(1)

    return story



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)

    # Get the file path from the command line
    dataset_path = sys.argv[1]
    print(dataset_path)
    # Load the dataset
    data = load_data(dataset_path)
    # Analyze the data
    data_analysis = analyze_data(data)
    
    #  Generate the story
    story = generate_story(data_analysis,data)
