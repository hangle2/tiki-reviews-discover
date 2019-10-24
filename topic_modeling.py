# https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

import sys

from data_helper import get_reviews_from_database
from pre_processing import pre_process, get_stop_words

product_id = 4536405

if len(sys.argv) > 1:
    product_id = sys.argv[1]

papers = get_reviews_from_database(product_id)

# Importing modules
import pandas as pd

papers = pd.DataFrame(papers)
papers.head()

data = pd.DataFrame({'text': papers['content']})

data['text'] = data['text'].apply(lambda x: pre_process(x))

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

stop_words = get_stop_words("data/stopwords.txt")

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words=stop_words)
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(data['text'])
# Visualise the 10 most common words
# plot_10_most_common_words(count_data, count_vectorizer)

import warnings

warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# Tweak the two parameters below
number_topics = 5
number_words = 20
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
