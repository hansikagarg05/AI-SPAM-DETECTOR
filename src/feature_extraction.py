from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features, vectorizer