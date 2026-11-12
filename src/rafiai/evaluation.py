
from collections import Counter

from rafiai.train import (
    create_model,
    create_logistic_regression_model
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
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
    Evaluates the model using Stratified K-Fold Cross-Validation
    and multiple evaluation metrics.
    """

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if min_class_count < 2:
        print("\nNot enough examples for Cross-Validation.")
        return

    n_splits = min(5, min_class_count)

    pipeline = Pipeline([
        ("cleaning", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB())
    ])

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )
    }

    results = cross_validate(
        pipeline,
        texts,
        labels,
        cv=cross_validation,
        scoring=scoring
    )

    print("\n================================")
    print("   RafiAI - Cross-Validation")
    print("================================")

    metrics = {
        "Accuracy": "test_accuracy",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "F1-score": "test_f1"
    }

    for metric_name, result_key in metrics.items():

        scores = results[result_key]

        print(f"\n{metric_name}:")

        for index, score in enumerate(scores, start=1):
            print(f"Fold {index}: {score:.2%}")

        print(f"Average: {scores.mean():.2%}")
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


def compare_models(texts, labels):
    """
    Compares Naive Bayes and Logistic Regression
    using the same Cross-Validation folds.
    """

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if min_class_count < 2:
        print("\nNot enough examples for model comparison.")
        return

    n_splits = min(5, min_class_count)

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    models = {
        "Naive Bayes": create_model(),
        "Logistic Regression": create_logistic_regression_model()
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )
    }

    print("\n================================")
    print("       RafiAI - Model Comparison")
    print("================================")

    for model_name, classifier in models.items():

        pipeline = Pipeline([
            ("cleaning", FunctionTransformer(clean_texts, validate=False)),
            ("tfidf", TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2)
            )),
            ("classifier", classifier)
        ])

        results = cross_validate(
            pipeline,
            texts,
            labels,
            cv=cross_validation,
            scoring=scoring
        )

        print(f"\n{model_name}")

        for metric_name in scoring:
            scores = results[f"test_{metric_name}"]

            print(
                f"{metric_name.capitalize()}: "
                f"{scores.mean():.2%}"
            )


def optimize_naive_bayes(texts, labels):
    """
    Tests different Alpha values for Naive Bayes
    using the same Cross-Validation folds.
    """

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if min_class_count < 2:
        print("\nNot enough examples for optimization.")
        return

    n_splits = min(5, min_class_count)

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    alpha_values = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
    ]
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )
    }

    print("\n================================")
    print("   RafiAI - Alpha Optimization")
    print("================================")

    for alpha in alpha_values:

        pipeline = Pipeline([
            ("cleaning", FunctionTransformer(clean_texts, validate=False)),
            ("tfidf", TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2)
            )),
            ("classifier", MultinomialNB(alpha=alpha))
        ])

        results = cross_validate(
            pipeline,
            texts,
            labels,
            cv=cross_validation,
            scoring=scoring
        )

        print(f"\nAlpha: {alpha}")

        for metric_name in scoring:

            scores = results[f"test_{metric_name}"]

            print(
                f"{metric_name.capitalize()}: "
                f"{scores.mean():.2%}"
            )
