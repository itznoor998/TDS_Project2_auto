#/// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv",
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "seaborn",
#   "os",
#   "sys",
#   "warnings",
#   "datetime",
#   "requests",
#   "json",
# ]
# ///

from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
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
    

def general_statistics(data):

    """Generate general statistics and return as a string."""

    numerical_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    numerical_data = data.loc[:,numerical_cols]
    categorical_data = data.loc[:,categorical_cols]

    stats = f"Dataset Summary:\n"
    stats += f"Shape: {data.shape[0]} rows, {data.shape[1]} columns\n"
    stats += f"Columns: {', '.join(data.columns)}\n"
    stats += f"Total no of NaN values in the dataset:{data.isna().sum().sum()}\n"
    stats += f"Unique values in categorical columns:\n{categorical_data.nunique()}\n"


    for col in categorical_cols:
        top_values = data[col].value_counts().head(5).to_dict()
        stats += f"Top five values in {col}:\n{top_values}\n"

    stats += f"\nSummary Statistics:\n{data.describe()}\n"

    outliers = {}
    
    for col in numerical_cols:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        threshold = 3
        outliers[col] = len(data[z_scores.abs() > threshold][col].tolist())
    stats += f"Number of outliers in each column:\n{outliers}\n"

    for num_col in numerical_cols:
        for Num_col in numerical_cols:
            if num_col != Num_col:
                stats += f"- Correlation between {num_col} and {Num_col}: {numerical_data[num_col].corr(numerical_data[Num_col])}\n"
    
    
    return stats

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
    file.close()

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

    # Define the function description for structured LLM prompting
    function_descriptions = [
        {
            "name": "generate_story",
            "description": "Generates an engaging, structured narrative based on data analysis insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_analysis": {
                        "type": "string",
                        "description": "Summary of the analysis findings."
                    }
                }
            }
        }
    ]

    # Prepare the request payload
    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates structured narratives from data analysis findings."
            },
            {
                "role": "user",
                "content": """Based on the following data analysis findings, generate a professional, engaging narrative 
                    with clear, insightful subheadings highlighting key trends and insights. Structure the story well."""
            }
        ],
        "max_tokens": 500,
        "functions": function_descriptions,
        "function_call": {"name": "generate_story", "arguments": {"data_analysis": data_analysis}}
    }

    # Make the request to the LLM endpoint
    try:
        response = requests.post(url, headers=headers, json=request_payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to connect to LLM endpoint. Details: {e}")
        sys.exit(1)

    # Parse the response
    try:
        raw_story = response.json()['choices'][0]['message']['function_call']['arguments']
        story = json.loads(raw_story)['data_analysis']
    except (KeyError, json.JSONDecodeError) as e:
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
            file.write("## Visualizations\n")
            for image in images:
                file.write(f"![{image}]({image})\n")
            
            # Write the story
            file.write("\n## Narrative Story\n")
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
