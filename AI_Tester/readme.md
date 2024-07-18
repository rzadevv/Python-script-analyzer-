# AI Assignment 

## Overview
This Python script analyzes a dataset containing questions, human answers, and machine answers. The goal is to perform various analyses, including generating word clouds, plotting histograms of answer lengths, and calculating cosine similarity scores between questions and answers.

## Code Explanation
- `import` statements: Import necessary libraries and modules.
- Load Data: Load the dataset from the Excel file into a pandas DataFrame.
- TF-IDF Vectorization: Use TF-IDF vectorization to convert text data into numerical representations.
- Word Clouds: Generate word clouds to visualize the most frequent words in questions and answers.
- Histograms: Plot histograms of answer lengths to understand the distribution of answer lengths.
- Cosine Similarity Heatmap: Calculate cosine similarity scores between questions and human answers, then plot them as a heatmap.

## Graph Interpretation
1. Word Clouds:
    - Questions Word Cloud: Visualizes the most common words in questions.
    - Human Answers Word Cloud: Visualizes the most common words in human answers.
    - Machine Answers Word Cloud: Visualizes the most common words in machine answers.
2. Histograms:
    - Our code shows the frequency distribution of answer lengths for human and machine answers.
3. Cosine Similarity Heatmap:
    - Heatmap of Question to Human Answer Cosine Similarities: Represents the cosine similarity scores between questions and human answers as a heatmap.

## Instructions
1. Ensure the `soru_cevap.xlsx` file containing the dataset is in the same directory as the script.
2. Run the script to perform the analyses and generate visualizations.

It can take  a little bit more time to run due to large data size,TF-IDF Vectorization,Cosine Similarity Calculations,Batch Processing in Loops :)
