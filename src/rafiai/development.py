
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


def run_development():
    """
    Executes the RafiAI development mode.
    """

    print("\n================================")
    print("   RafiAI - Development Mode")
    print("================================")

    # Load training dataset
    training_data = load_dataset("training.csv")

    training_texts = training_data["text"].tolist()
    training_labels = training_data["label"].tolist()

    # Cross-Validation
    cross_validate_model(
        training_texts,
        training_labels
    )

    # Compare models
    compare_models(
        training_texts,
        training_labels
    )

    # Optimize Alpha
    optimize_naive_bayes(
        training_texts,
        training_labels
    )

    # Grid Search
    best_result = grid_search_naive_bayes(
        training_texts,
        training_labels
    )

    if best_result is None:
        print("\nCould not find a suitable model.")
        return

    model, vectorizer = best_result

    # Save best model
    save_model(model, vectorizer)

    print("\nBest model trained and saved successfully!")

    # Training performance
    training_matrix = transform_texts(
        vectorizer,
        training_texts
    )

    evaluate_training_performance(
        model,
        training_matrix,
        training_labels
    )

    # Testing performance
    testing_data = load_dataset("testing.csv")

    testing_texts = testing_data["text"].tolist()
    testing_labels = testing_data["label"].tolist()

    testing_matrix = transform_texts(
        vectorizer,
        testing_texts
    )

    evaluate_model(
        model,
        testing_matrix,
        testing_labels
    )
