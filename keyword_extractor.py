import pandas as pd

df_idf = pd.read_json("/Users/quannk/tiki-reviews/data/review_all_clean.json")
# get the first 100k reviews only
df_idf = df_idf[:][:100000]

data = pd.DataFrame({"id": df_idf["id"], "product_id": df_idf['product_id'], 'text': df_idf['text']})

dict = {}

for index, row in df_idf.iterrows():
    if dict[row['product_id']] is not None:
        dict[row['product_id']] += " " + row['text']

import re
from pyvi import ViTokenizer


def pre_process(text):
    text = text.lower()
    text = re.sub("&lt;/?.*&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = ViTokenizer.tokenize(text)
    return text


df_idf['text'] = df_idf['title'] + " " + df_idf['content']
df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(x))

from sklearn.feature_extraction.text import CountVectorizer


def get_stop_words(stopword_file_path):
    """load stop words"""
    with open(stopword_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


docs = df_idf['text'].tolist()

stop_words = get_stop_words("/Users/quannk/tiki-reviews/data/stopwords.txt")

cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=50000, ngram_range=(1, 1))
word_count_vector = cv.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

feature_names = cv.get_feature_names()


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=20):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def magic(i):
    doc = docs[i]
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    print(doc)
    for k in keywords:
        print(k, keywords[k])


for i in range(100):
    magic(i)
    input("Press Enter to continue...")
