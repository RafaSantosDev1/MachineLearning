
from rafiai.model_storage import model_exists, load_model
from rafiai.predict import predict_sentiment
from rafiai.development import run_development
from rafiai.feedback_storage import save_feedback
from rafiai.web.app import run_web


def collect_feedback(user_text, prediction, classes):
    """
    Collects user feedback about a sentiment prediction.
    """

    print("\nWas the prediction correct?")
    print("1. Yes")
    print("2. No")
    print("0. Skip")

    while True:
        option = input("\nOption: ").strip()

        if option == "1":
            correct_label = str(prediction)
            break

        elif option == "2":
            print("\nAvailable sentiments:")

            for sentiment in classes:
                print(f"- {sentiment}")

            correct_label = input(
                "\nEnter the correct sentiment: "
            ).strip().lower()

            valid_labels = [
                str(sentiment).lower()
                for sentiment in classes
            ]

            if correct_label not in valid_labels:
                print("Invalid sentiment. Try again.")
                continue

            break

        elif option == "0":
            print("Feedback skipped.")
            return

        else:
            print("Invalid option. Try again.")

    save_feedback(
        user_text,
        prediction,
        correct_label
    )

    print("Feedback saved successfully!")


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

        collect_feedback(
            user_text,
            result,
            model.classes_
        )


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    while True:
        print("\nSelect a mode:")
        print("1. Development Mode")
        print("2. Usage Mode")
        print("3. Web Interface")
        print("0. Exit")

        option = input("\nOption: ").strip()

        if option == "1":
            run_development()

        elif option == "2":
            run_usage()

        elif option == "3":
            run_web()

        elif option == "0":
            print("\nRafiAI closed.")
            break

        else:
            print("\nInvalid option. Try again.")


if __name__ == "__main__":
    main()
