
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from rafiai.preprocessing import clean_texts


def evaluate_model(model, testing_matrix, testing_labels):
    """
    Evaluates the trained model using the testing dataset.
    """

    predictions = model.predict(testing_matrix)

    accuracy = accuracy_score(testing_labels, predictions)

    print("\n================================")
    print("       RafiAI - Evaluation")
    print("================================")

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification report:")

    print(
        classification_report(
            testing_labels,
            predictions,
            labels=model.classes_,
            target_names=model.classes_,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(f"Labels: {list(model.classes_)}")

    print(
        confusion_matrix(
            testing_labels,
            predictions,
            labels=model.classes_
        )
    )


def cross_validate_model(texts, labels):
    """
    Evaluates the model using Stratified K-Fold Cross-Validation.
    """

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if min_class_count < 2:
        print("\nNot enough examples for Cross-Validation.")
        return

    n_splits = min(5, min_class_count)

    pipeline = Pipeline([
        ("cleaning", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        texts,
        labels,
        cv=cross_validation,
        scoring="accuracy"
    )

    print("\n================================")
    print("   RafiAI - Cross-Validation")
    print("================================")

    for index, score in enumerate(scores):
        print(f"Fold {index + 1}: {score:.2%}")

    print(f"\nAverage Accuracy: {scores.mean():.2%}")
    print(f"Standard Deviation: {scores.std():.2%}")

def evaluate_training_performance(model, training_matrix, training_labels):
    """
    Evaluates the model on its training dataset.
    """

    predictions = model.predict(training_matrix)

    accuracy = accuracy_score(training_labels, predictions)

    print("\n================================")
    print("   RafiAI - Training Performance")
    print("================================")

    print(f"\nTraining Accuracy: {accuracy:.2%}")