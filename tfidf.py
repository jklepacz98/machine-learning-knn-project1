import math
from collections import Counter

def compute_tf(doc):
    term_count = Counter(doc)
    total_terms = len(doc)
    return {term: count / total_terms for term, count in term_count.items()}

def compute_idf(processed_docs):
    total_docs = len(processed_docs)
    all_terms = set(term for doc in processed_docs for term in doc)
    return {term: math.log(total_docs / (1 + sum(term in doc for doc in processed_docs)))
            for term in all_terms}

def compute_tfidf(tf, idf):
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in idf}