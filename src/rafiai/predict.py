from rafiai.preprocessing import transform_texts

def predict_sentiment(model, vectorizer, text):
    """
    Predicts the sentiment of a text.
    """

    text_matrix = transform_texts(vectorizer, [text])

    if text_matrix.nnz == 0:
        return "unknown"

    prediction = model.predict(text_matrix)

    return prediction[0]