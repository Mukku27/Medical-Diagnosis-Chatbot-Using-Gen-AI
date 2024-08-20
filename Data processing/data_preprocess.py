# -*- coding: utf-8 -*-
"""data_preprocess.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/148mYtSdPOidLBfk3X5H6N4t3OsKpuS5D
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import spacy
import gc

# Ensure necessary NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Load data
clinical_text_df = pd.read_csv("/Dataset/mtsamples.csv")

# Check columns and head of the dataframe
print(clinical_text_df.columns)
print(clinical_text_df.head(5))

# Remove rows with missing 'transcription'
clinical_text_df = clinical_text_df[clinical_text_df['transcription'].notna()]

# Function to get sentence and word count
def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
    for text in text_list:
        sentences = sent_tokenize(str(text).lower())
        sent_count += len(sentences)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    word_count = len(vocab)
    return sent_count, word_count

# Get sentence and word count
sent_count, word_count = get_sentence_word_count(clinical_text_df['transcription'].tolist())
print("Number of sentences in transcriptions column: " + str(sent_count))
print("Number of unique words in transcriptions column: " + str(word_count))

# Group by medical specialty and filter categories with more than 50 samples
data_categories = clinical_text_df.groupby('medical_specialty')
filtered_data_categories = data_categories.filter(lambda x: x.shape[0] > 50)
final_data_categories = filtered_data_categories.groupby('medical_specialty')

# Print category distribution
i = 1
print('===========Original Categories =======================')
for catName, dataCategory in data_categories:
    print('Cat:' + str(i) + ' ' + catName + ' : ' + str(len(dataCategory)))
    i += 1
print('==================================')

print('============Reduced Categories ======================')
i = 1
for catName, dataCategory in final_data_categories:
    print('Cat:' + str(i) + ' ' + catName + ' : ' + str(len(dataCategory)))
    i += 1

# Plot category distribution
plt.figure(figsize=(10, 10))
sns.countplot(y='medical_specialty', data=filtered_data_categories)
plt.show()

# Sample transcriptions
data = filtered_data_categories[['transcription', 'medical_specialty']]
data = data.drop(data[data['transcription'].isna()].index)
print('Sample Transcription 1:' + data.iloc[5]['transcription'] + '\n')
print('Sample Transcription 2:' + data.iloc[125]['transcription'] + '\n')
print('Sample Transcription 3:' + data.iloc[1000]['transcription'])

# Text preprocessing functions
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([w for w in text if not w.isdigit()])
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    text = REPLACE_BY_SPACE_RE.sub('', text.lower())
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Apply preprocessing
data['transcription'] = data['transcription'].apply(lambda x: clean_text(lemmatize_text(x)))
print('Sample Transcription 1:' + data.iloc[5]['transcription'] + '\n')
print('Sample Transcription 2:' + data.iloc[125]['transcription'] + '\n')
print('Sample Transcription 3:' + data.iloc[1000]['transcription'])

import spacy

# Load an alternative spaCy model
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    wordlist = []
    doc = nlp(text)
    for ent in doc.ents:
        wordlist.append(ent.text)
    return ' '.join(wordlist)

# Apply named entity recognition
data['transcription'] = data['transcription'].apply(process_text)
data['transcription'] = data['transcription'].apply(lambda x: clean_text(lemmatize_text(x)))
print('Sample Transcription 1:' + data.iloc[5]['transcription'] + '\n')
print('Sample Transcription 2:' + data.iloc[125]['transcription'] + '\n')
print('Sample Transcription 3:' + data.iloc[1000]['transcription'])

# Vectorize text data
vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 3),
    max_df=0.75,
    min_df=5,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    max_features=1000
)

tfIdfMat = vectorizer.fit_transform(data['transcription'])

# Use get_feature_names_out() instead of get_feature_names()
feature_names = sorted(vectorizer.get_feature_names_out())
print(feature_names)


# Dimensionality reduction
pca = PCA(n_components=0.95)
tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())

# t-SNE visualization
tsne_results = TSNE(n_components=2, init='random', random_state=0, perplexity=40).fit_transform(tfIdfMat_reduced)
plt.figure(figsize=(20, 10))
palette = sns.hls_palette(12, l=.3, s=.9)
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=data['medical_specialty'],
    palette=palette,
    legend="full",
    alpha=0.3
)
plt.show()

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, data['medical_specialty'], stratify=data['medical_specialty'], random_state=1)
print('Train_Set_Size:' + str(X_train.shape))
print('Test_Set_Size:' + str(X_test.shape))

clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=1)
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

# Evaluation
labels = data['medical_specialty'].unique()
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(cm, annot=True, cmap="Greens", ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()
print(classification_report(y_test, y_test_pred, labels=labels))

# SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(tfIdfMat_reduced, data['medical_specialty'])

# Split and train on resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=1)
print('Train_Set_Size:' + str(X_train.shape))
print('Test_Set_Size:' + str(X_test.shape))
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

# Evaluation on resampled data
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(cm, annot=True, cmap="Greens", ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()
print(classification_report(y_test, y_test_pred, labels=labels))

# Additional data exploration
mask = filtered_data_categories['medical_specialty'] == 'Radiology'
radiologyData = filtered_data_categories[mask]
print(radiologyData.head())
'''
mask = clinical_text_df['medical_specialty'] == 'Pediatrics - Neonatal'
pediaData = clinical_text_df[mask]
print(pediaData['transcription'].tolist()[1])
'''

# Filter for 'Pediatrics - Neonatal' medical specialty
mask = clinical_text_df['medical_specialty'] == 'Pediatrics - Neonatal'
pediaData = clinical_text_df[mask]

# Check if the DataFrame is empty before accessing elements
if not pediaData.empty:
    # Ensure there are enough rows in the DataFrame
    if len(pediaData['transcription'].tolist()) > 1:
        print(pediaData['transcription'].tolist()[1])
    else:
        print("The 'Pediatrics - Neonatal' specialty has less than 2 transcriptions.")
else:
    print("No data found for the 'Pediatrics - Neonatal' specialty.")


# Filter for 'Hematology - Oncology' medical specialty
mask = clinical_text_df['medical_specialty'] == 'Hematology - Oncology'
oncoData = clinical_text_df[mask]

# Check if the DataFrame is empty before accessing elements
if not oncoData.empty:
    # Ensure there are enough rows in the DataFrame
    if len(oncoData['transcription'].tolist()) > 1:
        print(oncoData['transcription'].tolist()[1])
    else:
        print("The 'Hematology - Oncology' specialty has less than 2 transcriptions.")
else:
    print("No data found for the 'Hematology - Oncology' specialty.")

# Cleanup
gc.collect()