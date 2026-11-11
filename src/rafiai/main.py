
from rafiai.data_loader import load_dataset
from rafiai.preprocessing import (
    run_preprocessing,
    transform_texts
)
from rafiai.train import create_model, train_model


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    # Load datasets
    training_data = load_dataset("training.csv")
    testing_data = load_dataset("testing.csv")

    # Extract training data
    training_texts = training_data["text"].tolist()
    training_labels = training_data["label"].tolist()

    # Preprocess training data
    vectorizer, training_matrix = run_preprocessing(training_texts)

    # Create and train model
    print("\nTraining model...")

    model = create_model()

    model = train_model(
        model,
        training_matrix,
        training_labels
    )

    print("Model trained successfully!")

    # Preprocess testing data
    testing_texts = testing_data["text"].tolist()
    testing_labels = testing_data["label"].tolist()

    testing_matrix = transform_texts(
        vectorizer,
        testing_texts
    )

    # Predict testing data
    predictions = model.predict(testing_matrix)

    print("\nTesting results:")

    for text, prediction, expected in zip(
        testing_texts,
        predictions,
        testing_labels
    ):
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print(f"Expected: {expected}")
        print("----------------------------")


if __name__ == "__main__":
    main()
