from data_helper import get_reviews_from_database
from pre_processing import pre_process, get_stop_words

sentences = get_reviews_from_database(4536405)
formated_sentences = []

for sentence in sentences:
    formated_sentences.append(pre_process(sentence['content']))

from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default

stop_words = get_stop_words("data/stopwords.txt")
r = Rake(stopwords=stop_words, language="vietnamese")

# Extraction given the text.
r.extract_keywords_from_sentences(formated_sentences)

phrases = r.get_ranked_phrases()
for phrase in phrases:
    print(phrase)
