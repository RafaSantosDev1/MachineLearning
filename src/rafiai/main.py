
from rafiai.model_storage import model_exists, load_model
from rafiai.predict import predict_sentiment
from rafiai.development import run_development


def run_usage():
    """
    Executes the RafiAI usage mode.
    """

    if not model_exists():
        print("\nNo trained model found.")
        print("Please run Development Mode first.")
        return

    print("\nLoading saved model...")

    saved_data = load_model()

    model = saved_data["model"]
    vectorizer = saved_data["vectorizer"]

    print("Model loaded successfully!")

    print("\n================================")
    print("       RafiAI - Prediction")
    print("================================")

    while True:
        user_text = input(
            "\nWrite a sentence (or 'exit' to quit): "
        )

        if user_text.lower() == "exit":
            print("RafiAI closed.")
            break

        if not user_text.strip():
            print("Please write a sentence.")
            continue

        result, probabilities = predict_sentiment(
            model,
            vectorizer,
            user_text
        )

        if result is None:
            print(
                "RafiAI does not recognize the words "
                "in this sentence."
            )
            continue

        print(f"\nPrediction: {result}")

        print("\nEstimated probabilities:")

        for sentiment, probability in probabilities.items():
            print(f"{sentiment}: {probability:.2%}")


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    while True:
        print("\nSelect a mode:")
        print("1. Development Mode")
        print("2. Usage Mode")
        print("0. Exit")

        option = input("\nOption: ").strip()

        if option == "1":
            run_development()

        elif option == "2":
            run_usage()

        elif option == "0":
            print("\nRafiAI closed.")
            break

        else:
            print("\nInvalid option. Try again.")


if __name__ == "__main__":
    main()
