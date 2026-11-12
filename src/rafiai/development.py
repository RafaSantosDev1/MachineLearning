
import time

from rafiai.data_loader import load_dataset
from rafiai.preprocessing import transform_texts
from rafiai.model_storage import save_model
from rafiai.evaluation import (
    evaluate_model,
    cross_validate_model,
    evaluate_training_performance,
    compare_models,
    optimize_naive_bayes,
    grid_search_naive_bayes
)

from sklearn.metrics import accuracy_score, f1_score


def run_development():
    """
    Executes the RafiAI development mode.
    """

    start_time = time.perf_counter()

    print("\n================================")
    print("       RafiAI - Development")
    print("================================")

    print("\n[1/5] Loading training data...")

    training_data = load_dataset("training.csv")
    training_texts = training_data["text"].tolist()
    training_labels = training_data["label"].tolist()

    print("[2/5] Running model analysis...")

    cross_validate_model(training_texts, training_labels)

    compare_models(training_texts, training_labels)

    optimize_naive_bayes(training_texts, training_labels)

    print("\n[3/5] Finding best model...")

    best_result = grid_search_naive_bayes(
        training_texts,
        training_labels
    )

    if best_result is None:
        print("\nCould not find a suitable model.")
        return

    model, vectorizer = best_result

    print("[4/5] Saving model...")

    save_model(model, vectorizer)

    training_matrix = transform_texts(
        vectorizer,
        training_texts
    )

    evaluate_training_performance(
        model,
        training_matrix,
        training_labels
    )

    print("\n[5/5] Evaluating model...")

    testing_data = load_dataset("testing.csv")
    testing_texts = testing_data["text"].tolist()
    testing_labels = testing_data["label"].tolist()

    testing_matrix = transform_texts(
        vectorizer,
        testing_texts
    )

    predictions = evaluate_model(
        model,
        testing_matrix,
        testing_labels,
        testing_texts
    )

    training_accuracy = accuracy_score(
        training_labels,
        model.predict(training_matrix)
    )

    testing_accuracy = accuracy_score(
        testing_labels,
        predictions
    )

    testing_f1 = f1_score(
        testing_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    elapsed_time = time.perf_counter() - start_time

    print("\n================================")
    print("         RafiAI - Summary")
    print("================================")

    print(f"\nTraining examples: {len(training_texts)}")
    print(f"Testing examples: {len(testing_texts)}")

    print("\nModel: Multinomial Naive Bayes")
    print(f"Alpha: {model.alpha}")

    print("\nModel Performance:")
    print(f"Training Accuracy: {training_accuracy:.2%}")
    print(f"Testing Accuracy: {testing_accuracy:.2%}")
    print(f"Testing F1-score: {testing_f1:.2%}")

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    print("\nModel saved successfully!")

    print("\n================================")
    print("   Development completed!")
    print("================================")
