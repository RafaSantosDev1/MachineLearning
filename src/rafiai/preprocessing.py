
import re
from pathlib import Path

import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer


lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Cleans and normalizes text, including common negation expressions.
    """

    text = text.lower()

    # Normalize contractions
    contractions = {
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not",
        r"\bhasn't\b": "has not",
        r"\bhadn't\b": "had not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bcouldn't\b": "could not",
        r"\bwouldn't\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bmustn't\b": "must not"
    }

    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Normalize cannot
    text = re.sub(r"\bcannot\b", "can not", text)

    # Normalize no longer
    text = re.sub(
        r"\bno longer\s+(\w+)",
        r"NOT_\1",
        text
    )

    # Normalize never
    text = re.sub(r"\bnever\b", "not", text)

    # Identify absence of common problems
    problems_pattern = (
        r"(difficulty|difficulties|problem|problems|"
        r"issue|issues|error|errors)"
    )

    text = re.sub(
        rf"\bno\s+{problems_pattern}\b",
        r"NOT_\1",
        text
    )

    text = re.sub(
        rf"\bwithout\s+(?:any\s+)?{problems_pattern}\b",
        r"NOT_\1",
        text
    )

    # Mark the word following a negation
    text = re.sub(
        r"\bnot\s+(?:(?:a|an|the|any|ever)\s+)?(?!only\b)(\w+)",
        r"NOT_\1",
        text
    )

    # Remove unnecessary punctuation
    text = re.sub(r"[^\w\s']", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_wordnet_pos(tag):
    """
    Converts a POS tag into a WordNet part of speech.
    """

    if tag.startswith("J"):
        return wordnet.ADJ

    if tag.startswith("V"):
        return wordnet.VERB

    if tag.startswith("N"):
        return wordnet.NOUN

    if tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


def lemmatize_text(text):
    """
    Lemmatizes words while preserving negation markers.
    """

    tokens = word_tokenize(text)

    # Temporarily remove the NOT_ prefix for POS tagging
    tagging_tokens = [
        token[4:] if token.lower().startswith("not_") else token
        for token in tokens
    ]

    tagged_words = pos_tag(tagging_tokens)

    lemmas = []

    for original_word, (_, tag) in zip(tokens, tagged_words):

        word = original_word.lower()
        pos = get_wordnet_pos(tag)

        # Preserve negation and lemmatize the following word
        if word.startswith("not_"):
            suffix = word[4:]
            lemma = lemmatizer.lemmatize(suffix, pos=pos)

            lemmas.append(f"NOT_{lemma}")
            continue

        # Preserve important words
        if word == "latest":
            lemmas.append(word)
            continue

        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmas.append(lemma)

    return " ".join(lemmas)


def clean_texts(texts):
    """
    Cleans and lemmatizes a list of texts.
    """

    processed_texts = []

    for text in texts:
        cleaned_text = clean_text(text)
        lemmatized_text = lemmatize_text(cleaned_text)

        processed_texts.append(lemmatized_text)

    return processed_texts


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

    processed_texts = clean_texts(texts)

    return vectorizer.fit_transform(processed_texts)


def transform_texts(vectorizer, texts):
    """
    Transforms new texts using the existing vocabulary.
    """

    processed_texts = clean_texts(texts)

    return vectorizer.transform(processed_texts)


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
