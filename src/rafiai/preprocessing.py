
from sklearn.feature_extraction.text import CountVectorizer


def create_vectorizer():
    return CountVectorizer()


def fit_transform_texts(vectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)


def run_preprocessing():
    texts = [
        "Adoro Python",
        "Adoro Java",
        " Python olas amanha"
    ]

    vectorizer = create_vectorizer()
    matrix = fit_transform_texts(vectorizer, texts)

    print("\nVocabulary:")
    print(vectorizer.get_feature_names_out())

    print("\nNumerical matrix:")
    print(matrix.toarray())
