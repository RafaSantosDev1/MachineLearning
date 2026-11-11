
from rafiai.preprocessing import transform_texts


def predict_sentiment(model, vectorizer, text):
    """
    Predicts the sentiment and its estimated probabilities.
    """

    text_matrix = transform_texts(vectorizer, [text])

    if text_matrix.nnz == 0:   #nnz significa Number of Non-Zero elements.
        return None, None

    prediction = model.predict(text_matrix)[0]

    probabilities = model.predict_proba(text_matrix)[0]

    results = dict(zip(model.classes_, probabilities))

    return prediction, results
