def predict_message(message, model, vectorizer, clean_text):
    cleaned_message = clean_text(message)
    features = vectorizer.transform([cleaned_message])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    return prediction, probability