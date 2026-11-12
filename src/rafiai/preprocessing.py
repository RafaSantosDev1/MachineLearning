
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



def clean_text(text):
    """
    Cleans and normalizes a text and identifies negations.
    """

    text = text.lower()

    # Normalize common negation expressions
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\bdoesn't\b", "does not", text)
    text = re.sub(r"\bdidn't\b", "did not", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bnever\b", "not", text)

    # Mark the word following a negation
    text = re.sub(
        r"\bnot\s+(\w+)",
        r"NOT_\1",
        text
    )

    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def clean_texts(texts):
    """
    Cleans a list of texts.
    """

    return [clean_text(text) for text in texts]


def create_vectorizer():
    """
    Creates the TF-IDF vectorizer using unigrams and bigrams.
    """

    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2)
    )


def fit_transform_texts(vectorizer, texts):
    """
    Learns the vocabulary and calculates TF-IDF values.
    """

    cleaned_texts = clean_texts(texts)

    return vectorizer.fit_transform(cleaned_texts)


def transform_texts(vectorizer, texts):
    """
    Transforms new texts using the existing vocabulary.
    """

    cleaned_texts = clean_texts(texts)

    return vectorizer.transform(cleaned_texts)


def run_preprocessing(texts):
    """
    Executes the preprocessing pipeline and saves the results.
    """

    vectorizer = create_vectorizer()
    matrix = fit_transform_texts(vectorizer, texts)

    output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "preprocessing_results.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Vocabulary:\n")
        file.write(str(vectorizer.get_feature_names_out()))

        file.write("\n\nTF-IDF matrix:\n")
        file.write(
            np.array2string(
                matrix.toarray(),
                threshold=np.inf,
                precision=3,
                suppress_small=False
            )
        )

    return vectorizer, matrix
