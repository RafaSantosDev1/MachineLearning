from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def create_model():
    """
    Creates the Naive Bayes classification model.
    """
    return MultinomialNB(alpha=0.005)


def create_logistic_regression_model():
    """
    Creates the Logistic Regression classification model.
    """
    return LogisticRegression(max_iter=1000)


def train_model(model, training_matrix, training_labels):
    """
    Trains the model using numerical text representations
    and their corresponding labels.
    """
    model.fit(training_matrix, training_labels)

    return model