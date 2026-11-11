
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path


def create_vectorizer():
    return CountVectorizer()


def fit_transform_texts(vectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)


def run_preprocessing(texts):
    vectorizer = create_vectorizer()
    matrix = fit_transform_texts(vectorizer, texts)

    output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "preprocessing_results.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Vocabulary:\n")
        file.write(str(vectorizer.get_feature_names_out()))

        file.write("\n\nNumerical matrix:\n")
        file.write(str(matrix.toarray()))

    return vectorizer, matrix
