
from rafiai.data_loader import load_dataset
from rafiai.preprocessing import run_preprocessing, transform_texts
from rafiai.train import create_model, train_model
from rafiai.predict import predict_sentiment
from rafiai.model_storage import model_exists, save_model, load_model
from rafiai.evaluation import evaluate_model


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    if model_exists():
        print("\nLoading saved model...")

        saved_data = load_model()
        model = saved_data["model"]
        vectorizer = saved_data["vectorizer"]

        print("Model loaded successfully!")

    else:
        print("\nTraining new model...")

        training_data = load_dataset("training.csv")

        training_texts = training_data["text"].tolist()
        training_labels = training_data["label"].tolist()

        vectorizer, training_matrix = run_preprocessing(training_texts)

        model = create_model()
        model = train_model(model, training_matrix, training_labels)

        save_model(model, vectorizer)

        print("Model trained and saved successfully!")

    # Model evaluation
    testing_data = load_dataset("testing.csv")

    testing_texts = testing_data["text"].tolist()
    testing_labels = testing_data["label"].tolist()

    testing_matrix = transform_texts(vectorizer, testing_texts)

    evaluate_model(model, testing_matrix, testing_labels)

    # Interactive predictions
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
