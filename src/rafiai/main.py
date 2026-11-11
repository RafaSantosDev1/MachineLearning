
from rafiai.data_loader import load_dataset
from rafiai.preprocessing import run_preprocessing


def main():
    print("================================")
    print("          RafiAI")
    print("================================")

    print("\nLoading datasets...")

    training_data = load_dataset("training.csv")
    testing_data = load_dataset("testing.csv")

    print("\nTraining dataset:")
    print(f"Number of examples: {len(training_data)}")

    print("\nTesting dataset:")
    print(f"Number of examples: {len(testing_data)}")

    print("\nTraining preprocessing...")

    texts = training_data["text"].tolist()

    vectorizer, matrix = run_preprocessing(texts)
    print("\n vectors creater in data/preprocessing_results.txt")

    print("\nPreprocessing completed successfully! ")


if __name__ == "__main__":
    main()
