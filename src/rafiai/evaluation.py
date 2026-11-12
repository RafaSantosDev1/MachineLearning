
from collections import Counter

from rafiai.train import (
    create_model,
    create_logistic_regression_model
)

from rafiai.preprocessing import (
    clean_texts,
    create_vectorizer,
    transform_texts
)

from sklearn.metrics import (
    accuracy_score,
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
from sklearn.naive_bayes import MultinomialNB


def create_pipeline(classifier):
    """
    Creates a machine learning pipeline.
    """

    return Pipeline([
        ("cleaning", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", create_vectorizer()),
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

    print(
        f"\nIncorrect predictions: "
        f"{incorrect_count}/{len(testing_labels)}"
    )

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
    Analyzes incorrectly classified sentences and their vocabulary.
    """

    analyzer = vectorizer.build_analyzer()
    vocabulary = set(vectorizer.get_feature_names_out())

    errors = 0

    print("\n================================")
    print("       RafiAI - Error Analysis")
    print("================================")

    for text, actual, predicted in zip(
        testing_texts,
        testing_labels,
        predictions
    ):

        if actual != predicted:
            errors += 1

            processed_text = clean_texts([text])[0]
            features = analyzer(processed_text)

            recognized = [
                word for word in features
                if word in vocabulary
            ]

            missing = [
                word for word in features
                if word not in vocabulary
            ]

            print(f"\nSentence: {text}")
            print(f"Expected: {actual}")
            print(f"Predicted: {predicted}")
            print(f"Recognized features: {recognized}")
            print(f"Unknown features: {missing}")

    if errors == 0:
        print("\nAll predictions were correct!")

    print(f"\nTotal errors: {errors}/{len(testing_labels)}")


def analyze_sentiment_vocabulary(texts, labels):
    """
    Analyzes the most frequent words in each sentiment class.
    """

    positive_words = Counter()
    negative_words = Counter()

    processed_texts = clean_texts(texts)

    for text, label in zip(processed_texts, labels):

        words = text.split()

        if label == "positive":
            positive_words.update(words)

        elif label == "negative":
            negative_words.update(words)

    return {
        "positive": positive_words.most_common(20),
        "negative": negative_words.most_common(20)
    }


def analyze_confusion_and_confidence(
    model,
    testing_matrix,
    testing_texts,
    testing_labels,
    predictions
):
    """
    Displays the confusion matrix and analyzes prediction confidence.
    """

    probabilities = model.predict_proba(testing_matrix)

    classes = model.classes_

    matrix = confusion_matrix(
        testing_labels,
        predictions,
        labels=classes
    )

    print("\n================================")
    print("       RafiAI - Confusion Matrix")
    print("================================")

    print(f"\nClasses: {list(classes)}")
    print(matrix)

    print("\nRows = Expected | Columns = Predicted")

    print("\n================================")
    print("    RafiAI - Confidence Analysis")
    print("================================")

    error_count = 0

    for text, actual, predicted, probability in zip(
        testing_texts,
        testing_labels,
        predictions,
        probabilities
    ):

        if actual != predicted:

            error_count += 1

            predicted_index = list(classes).index(predicted)
            actual_index = list(classes).index(actual)

            predicted_probability = probability[predicted_index]
            actual_probability = probability[actual_index]

            margin = predicted_probability - actual_probability

            print(f"\nSentence: {text}")
            print(f"Expected: {actual}")
            print(f"Predicted: {predicted}")

            print(
                f"Confidence in prediction: "
                f"{predicted_probability:.2%}"
            )

            print(
                f"Probability of correct class: "
                f"{actual_probability:.2%}"
            )

            print(f"Confidence margin: {margin:.2%}")

    print(f"\nTotal incorrect predictions: {error_count}")


def analyze_feature_contributions(
    model,
    vectorizer,
    testing_texts,
    testing_labels,
    predictions
):
    """
    Analyzes which features contribute most to each incorrect prediction.
    """

    feature_names = vectorizer.get_feature_names_out()
    classes = list(model.classes_)

    negative_index = classes.index("negative")
    positive_index = classes.index("positive")

    feature_log_prob = model.feature_log_prob_

    print("\n================================")
    print("   RafiAI - Feature Contribution")
    print("================================")

    errors = 0

    for text, actual, predicted in zip(
        testing_texts,
        testing_labels,
        predictions
    ):

        if actual == predicted:
            continue

        errors += 1

        processed_text = clean_texts([text])[0]
        text_matrix = vectorizer.transform([processed_text])

        feature_indices = text_matrix.nonzero()[1]
        feature_values = text_matrix.data

        contributions = []

        for index, value in zip(feature_indices, feature_values):

            negative_score = (
                value * feature_log_prob[negative_index][index]
            )

            positive_score = (
                value * feature_log_prob[positive_index][index]
            )

            difference = positive_score - negative_score

            contributions.append({
                "feature": feature_names[index],
                "negative": negative_score,
                "positive": positive_score,
                "difference": difference
            })

        contributions.sort(
            key=lambda item: abs(item["difference"]),
            reverse=True
        )

        print(f"\nSentence: {text}")
        print(f"Expected: {actual}")
        print(f"Predicted: {predicted}")

        print("\nFeature contributions:")

        for item in contributions[:10]:

            direction = (
                "positive"
                if item["difference"] > 0
                else "negative"
            )

            print(
                f"{item['feature']:<25} "
                f"Direction: {direction:<8} "
                f"Difference: {item['difference']:.4f}"
            )

    if errors == 0:
        print("\nNo classification errors found.")

    print(f"\nTotal analyzed errors: {errors}")


def analyze_feature_distribution(
    training_texts,
    training_labels,
    features
):
    """
    Analyzes the distribution of selected features across classes.
    Also displays the training sentences containing each feature.
    """

    analyzer = create_vectorizer().build_analyzer()

    feature_examples = {
        feature: {
            "positive": [],
            "negative": []
        }
        for feature in features
    }

    class_counts = Counter(training_labels)

    processed_texts = clean_texts(training_texts)

    for text, label in zip(processed_texts, training_labels):

        text_features = set(analyzer(text))

        for feature in features:

            if feature in text_features:
                feature_examples[feature][label].append(text)

    print("\n================================")
    print("   RafiAI - Feature Distribution")
    print("================================")

    for feature, examples in feature_examples.items():

        positive = examples["positive"]
        negative = examples["negative"]

        print(f"\nFeature: {feature}")

        print(
            f"Positive: {len(positive)}/"
            f"{class_counts['positive']}"
        )

        print(
            f"Negative: {len(negative)}/"
            f"{class_counts['negative']}"
        )

        for label, sentences in examples.items():

            if sentences:

                print(f"\n{label.upper()} examples:")

                for sentence in sentences:
                    print(f" - {sentence}")


def evaluate_validation(
    model,
    vectorizer,
    validation_texts,
    validation_labels
):
    """
    Evaluates the model using validation data and analyzes
    the feature contributions of incorrect predictions.
    """

    print("\n================================")
    print("       RafiAI - Validation")
    print("================================")

    validation_matrix = transform_texts(
        vectorizer,
        validation_texts
    )

    predictions = model.predict(validation_matrix)

    accuracy = accuracy_score(validation_labels, predictions)

    precision = precision_score(
        validation_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        validation_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        validation_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nValidation Performance:")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")

    print("\n================================")
    print("       Validation Errors")
    print("================================")

    errors = sum(
        expected != predicted
        for expected, predicted in zip(
            validation_labels,
            predictions
        )
    )

    for text, expected, predicted in zip(
        validation_texts,
        validation_labels,
        predictions
    ):

        if expected != predicted:
            print(f"\nSentence: {text}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted}")

    print(
        f"\nTotal validation errors: "
        f"{errors}/{len(validation_texts)}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        validation_labels,
        predictions,
        labels=model.classes_
    ))

    if errors > 0:
        print("\n================================")
        print("   Validation Feature Analysis")
        print("================================")

        analyze_feature_contributions(
            model,
            vectorizer,
            validation_texts,
            validation_labels,
            predictions
        )

    return predictions
