
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

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from rafiai.preprocessing import clean_texts


def create_pipeline(classifier):
    """
    Creates a machine learning pipeline.
    """

    return Pipeline([
        ("cleaning", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])


def create_cross_validation(labels):
    """
    Creates a Stratified K-Fold cross-validation strategy.
    """

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if min_class_count < 2:
        return None

    return StratifiedKFold(
        n_splits=min(5, min_class_count),
        shuffle=True,
        random_state=42
    )


def evaluate_model(model, testing_matrix, testing_labels, testing_texts):
    """
    Evaluates the trained model using the testing dataset.
    """

    predictions = model.predict(testing_matrix)

    accuracy = accuracy_score(testing_labels, predictions)

    precision = precision_score(
        testing_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        testing_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        testing_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    incorrect_count = sum(
        actual != predicted
        for actual, predicted in zip(testing_labels, predictions)
    )

    print("\n================================")
    print("       RafiAI - Evaluation")
    print("================================")

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")

    print(f"\nIncorrect predictions: {incorrect_count}/{len(testing_labels)}")

    return predictions


def cross_validate_model(texts, labels):
    """
    Evaluates the model using Cross-Validation.
    """

    cross_validation = create_cross_validation(labels)

    if cross_validation is None:
        print("\nNot enough examples for Cross-Validation.")
        return

    pipeline = create_pipeline(create_model())

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

    print("\nCross-Validation:")

    print(f"Accuracy:  {results['test_accuracy'].mean():.2%}")
    print(f"Precision: {results['test_precision'].mean():.2%}")
    print(f"Recall:    {results['test_recall'].mean():.2%}")
    print(f"F1-score:  {results['test_f1'].mean():.2%}")


def evaluate_training_performance(model, training_matrix, training_labels):
    """
    Evaluates the model on its training dataset.
    """

    predictions = model.predict(training_matrix)

    accuracy = accuracy_score(training_labels, predictions)

    print(f"\nTraining Accuracy: {accuracy:.2%}")


def compare_models(texts, labels):
    """
    Compares Naive Bayes and Logistic Regression.
    """

    cross_validation = create_cross_validation(labels)

    if cross_validation is None:
        print("\nNot enough examples for model comparison.")
        return

    models = {
        "Naive Bayes": create_model(),
        "Logistic Regression": create_logistic_regression_model()
    }

    print("\nModel Comparison:")

    for model_name, classifier in models.items():

        pipeline = create_pipeline(classifier)

        results = cross_validate(
            pipeline,
            texts,
            labels,
            cv=cross_validation,
            scoring="f1_macro"
        )

        print(
            f"{model_name}: "
            f"F1-score = {results['test_score'].mean():.2%}"
        )


def optimize_naive_bayes(texts, labels):
    """
    Tests different Alpha values for Naive Bayes.
    """

    cross_validation = create_cross_validation(labels)

    if cross_validation is None:
        print("\nNot enough examples for optimization.")
        return

    alpha_values = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05
    ]

    best_alpha = None
    best_score = 0

    for alpha in alpha_values:

        pipeline = create_pipeline(
            MultinomialNB(alpha=alpha)
        )

        results = cross_validate(
            pipeline,
            texts,
            labels,
            cv=cross_validation,
            scoring="f1_macro"
        )

        score = results["test_score"].mean()

        if score > best_score:
            best_score = score
            best_alpha = alpha

    print("\nAlpha Optimization:")
    print(f"Best Alpha: {best_alpha}")
    print(f"Best F1-score: {best_score:.2%}")


def grid_search_naive_bayes(texts, labels):
    """
    Finds the best Naive Bayes Alpha value using GridSearchCV.
    Returns the best trained model and TF-IDF vectorizer.
    """

    cross_validation = create_cross_validation(labels)

    if cross_validation is None:
        print("\nNot enough examples for GridSearchCV.")
        return None

    pipeline = create_pipeline(MultinomialNB())

    parameter_grid = {
        "classifier__alpha": [
            0.001,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05
        ]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True
    )

    search.fit(texts, labels)

    print("\nGrid Search:")
    print(f"Best Alpha: {search.best_params_['classifier__alpha']}")
    print(f"Best F1-score: {search.best_score_:.2%}")

    best_pipeline = search.best_estimator_

    best_model = best_pipeline.named_steps["classifier"]
    best_vectorizer = best_pipeline.named_steps["tfidf"]

    return best_model, best_vectorizer


def analyze_vocabulary(vectorizer, testing_texts, testing_labels, predictions):
    """
    Analyzes vocabulary used in incorrectly classified sentences.
    """

    analyzer = vectorizer.build_analyzer()
    vocabulary = set(vectorizer.get_feature_names_out())

    incorrect_count = 0

    for text, actual, predicted in zip(
        testing_texts,
        testing_labels,
        predictions
    ):
        if actual != predicted:
            incorrect_count += 1

            cleaned_text = clean_texts([text])[0]
            features = analyzer(cleaned_text)

            recognized = [
                word for word in features
                if word in vocabulary
            ]

            missing = [
                word for word in features
                if word not in vocabulary
            ]

            print(f"\nIncorrect: {text}")
            print(f"Unknown features: {missing}")

    if incorrect_count == 0:
        print("\nAll predictions were correct!")


def analyze_sentiment_vocabulary(texts, labels):
    """
    Analyzes the most frequent words in each sentiment class.
    This function can be used when detailed analysis is needed.
    """

    from rafiai.preprocessing import clean_text

    positive_words = Counter()
    negative_words = Counter()

    for text, label in zip(texts, labels):

        words = clean_text(text).split()

        if label == "positive":
            positive_words.update(words)

        elif label == "negative":
            negative_words.update(words)

    return {
        "positive": positive_words.most_common(20),
        "negative": negative_words.most_common(20)
    }
