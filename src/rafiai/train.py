
from sklearn.naive_bayes import MultinomialNB


def create_model():
    """
    Creates the Naive Bayes classification model.
    """
    return MultinomialNB()


def train_model(model, training_matrix, training_labels):
    """
    Trains the model using numerical text representations
    and their corresponding labels.
    """
    model.fit(training_matrix, training_labels)

    return model
