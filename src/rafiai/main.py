
from rafiai.data_loader import load_dataset
from rafiai.preprocessing import run_preprocessing, transform_texts
from rafiai.train import create_model, train_model
from rafiai.predict import predict_sentiment


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    training_data = load_dataset("training.csv")
    testing_data = load_dataset("testing.csv")

    training_texts = training_data["text"].tolist()
    training_labels = training_data["label"].tolist()

    vectorizer, training_matrix = run_preprocessing(training_texts)

    print("\nTraining model...")

    model = create_model()
    model = train_model(model, training_matrix, training_labels)

    print("Model trained successfully!")

    testing_texts = testing_data["text"].tolist()
    testing_labels = testing_data["label"].tolist()

    testing_matrix = transform_texts(vectorizer, testing_texts)

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

    print("\n================================")
    print("       RafiAI - Prediction")
    print("================================")

    while True:
        user_text = input("\nWrite a sentence (or 'exit' to quit): ")

        if user_text.lower() == "exit":
            print("RafiAI closed.")
            break

        if not user_text.strip():
            print("Please write a sentence.")
            continue

        result = predict_sentiment(model, vectorizer, user_text)

        print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
