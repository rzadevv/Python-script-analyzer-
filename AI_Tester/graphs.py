import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the data from the Excel file
data = pd.read_excel('soru_cevap.xlsx')

# Prepare the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Combine all texts (questions, human answers, machine answers) for TF-IDF training
all_texts = data['soru'].tolist() + data['insan cevabı'].tolist() + data['makine cevabı'].tolist()

# Fit the TF-IDF vectorizer on all texts
tfidf_vectorizer.fit(all_texts)

# Transform questions and answers to TF-IDF vector representations
question_tfidf = tfidf_vectorizer.transform(data['soru'])
human_answer_tfidf = tfidf_vectorizer.transform(data['insan cevabı'])
machine_answer_tfidf = tfidf_vectorizer.transform(data['makine cevabı'])

# Function to plot word clouds
def plot_word_cloud(text_series, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_series))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate word clouds for questions, human answers, and machine answers
plot_word_cloud(data['soru'], 'Questions Word Cloud')
plot_word_cloud(data['insan cevabı'], 'Human Answers Word Cloud')
plot_word_cloud(data['makine cevabı'], 'Machine Answers Word Cloud')

# Plot histogram of answer lengths
plt.figure(figsize=(10, 5))
plt.hist(data['insan cevabı'].str.len(), bins=30, alpha=0.7, label='Human Answers', color='red')
plt.hist(data['makine cevabı'].str.len(), bins=30, alpha=0.7, label='Machine Answers', color='purple')
plt.title('Distribution of Answer Lengths')
plt.xlabel('Length of Answers')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate and plot a heatmap of the cosine similarity scores
similarity_matrix = cosine_similarity(question_tfidf, human_answer_tfidf)
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Heatmap of Question to Human Answer Cosine Similarities')
plt.xlabel('Human Answers')
plt.ylabel('Questions')
plt.show()
