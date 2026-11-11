from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, testing_matrix, testing_labels):
    """
    Evaluates the model using the testing dataset.
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
