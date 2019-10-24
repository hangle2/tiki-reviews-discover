import re
from pyvi import ViTokenizer


def get_stop_words(stopword_file_path):
    """load stop words"""
    with open(stopword_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


stop_words = get_stop_words("data/stopwords.txt")


def pre_process(text):
    text = text.lower()
    text = re.sub("&lt;/?.*&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = ViTokenizer.tokenize(text)
    tokens = [w for w in text.split(' ') if not w in stop_words]
    return ' '.join(tokens)
