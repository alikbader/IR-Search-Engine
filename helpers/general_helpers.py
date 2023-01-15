import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import csv

nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize_removeStopWord(query):
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens_list = []
    no_dupl = list(dict.fromkeys(tokens))
    for t in no_dupl:
        if t not in all_stopwords:
            tokens_list.append(t)
    return tokens_list


def read_csv_to_dict(path):
    output = {}

    with open(path, 'r') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            output[row[0]] = row[1]

    return output


def get_doc_id_title_list(doc_id_any, doc_id_title):
    doc_id_title_res = []

    for doc_id, _any in doc_id_any:
        doc_id_title_res.append((doc_id, doc_id_title[str(doc_id)]))

    return doc_id_title_res


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
