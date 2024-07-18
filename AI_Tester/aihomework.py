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

# Function to calculate top N similar items
def get_top_n_similarities(tfidf_matrix, query_vector, n=5):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[-n:][::-1]
    return top_n_indices

# Sample 1000 random indices for questions
sample_indices = np.random.choice(question_tfidf.shape[0], size=1000, replace=False)

# Initialize success rate counters
human_success_top1 = 0
human_success_top5 = 0
machine_success_top1 = 0
machine_success_top5 = 0

# Calculate top N similarities in batches
batch_size = 100  # Batch size for cosine similarity computation
for start_index in range(0, len(sample_indices), batch_size):
    end_index = start_index + batch_size
    batch_indices = sample_indices[start_index:end_index]
    
    # Get the vectors for the current batch
    batch_question_vectors = question_tfidf[batch_indices]
    
    # Calculate cosine similarity in a batch for human and machine answers
    batch_human_similarities = cosine_similarity(batch_question_vectors, human_answer_tfidf)
    batch_machine_similarities = cosine_similarity(batch_question_vectors, machine_answer_tfidf)
    
    # Calculate success rates
    for i, question_index in enumerate(batch_indices):
        top_human_indices = np.argsort(-batch_human_similarities[i])[:5]
        top_machine_indices = np.argsort(-batch_machine_similarities[i])[:5]
        
        # For human answers
        if question_index in top_human_indices:
            human_success_top5 += 1
            if question_index == top_human_indices[0]:
                human_success_top1 += 1

        # For machine answers
        if question_index in top_machine_indices:
            machine_success_top5 += 1
            if question_index == top_machine_indices[0]:
                machine_success_top1 += 1

# Calculate success rates
human_success_top1_rate = (human_success_top1 / 1000) * 100
human_success_top5_rate = (human_success_top5 / 1000) * 100
machine_success_top1_rate = (machine_success_top1 / 1000) * 100
machine_success_top5_rate = (machine_success_top5 / 1000) * 100

# Plot success rates
labels = ['Top 1', 'Top 5']
human_rates = [human_success_top1_rate, human_success_top5_rate]
machine_rates = [machine_success_top1_rate, machine_success_top5_rate]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, human_rates, width, label='Human Answers', color='red')
rects2 = ax.bar(x + width/2, machine_rates, width, label='Machine Answers', color='purple')

# Add labels and legend
ax.set_ylabel('Success Rates (%)')
ax.set_title('Top 1 and Top 5 Success Rates for Human vs. Machine Answers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

# Additional analysis and visualization can be added here, such as word clouds, histograms, and heatmaps
