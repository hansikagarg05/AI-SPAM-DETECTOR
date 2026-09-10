from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, X_test, y_test