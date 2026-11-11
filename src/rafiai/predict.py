def predict_sentiment(model, vectorizer, text):
    """
    Predicts the sentiment of a text.
    """

    text_matrix = vectorizer.transform([text])

    prediction = model.predict(text_matrix)

    return prediction[0]
