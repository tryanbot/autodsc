from sklearn import pipeline
from autodsc.utils.sklearn import feature_extraction 

class TextEncoder:
  def __call__(self, corpus, encoder, num_features=None, vocabulary=None):
    if encoder == "hash": # Convert a collection of text documents to a matrix of token occurrences.
      vectorizer = feature_extraction.text.HashingVectorizer(n_features=num_features)
      encoded_corpus = vectorizer.fit_transform(corpus)
    
    elif encoder == "count": # Transforms text into a sparse matrix of n-gram counts.
      vectorizer = feature_extraction.text.CountVectorizer()
      encoded_corpus = vectorizer.fit_transform(corpus)
    
    elif encoder == "tfidf": # Convert a collection of raw documents to a matrix of TF-IDF features.
      vectorizer = feature_extraction.text.TfidfVectorizer()
      encoded_corpus = vectorizer.fit_transform(corpus)
    
    elif encoder == "count-tfidf": # Transform a count matrix to a normalized tf or tf-idf representation.
      pipe = pipeline.Pipeline([('count', feature_extraction.text.CountVectorizer(vocabulary=vocabulary)),
                       ('tfid', feature_extraction.text.TfidfTransformer())]).fit(corpus)
      encoded_corpus = pipe['count'].transform(corpus)            
    
    return encoded_corpus
      