
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    """
    Cleans and normalizes a text.
    """

    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_vectorizer():
    return CountVectorizer(token_pattern=r"(?u)\b\w+\b")


def fit_transform_texts(vectorizer, texts):
    cleaned_texts = [clean_text(text) for text in texts]

    return vectorizer.fit_transform(cleaned_texts)


def transform_texts(vectorizer, texts):
    cleaned_texts = [clean_text(text) for text in texts]

    return vectorizer.transform(cleaned_texts)


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
