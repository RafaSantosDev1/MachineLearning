
from rafiai.model_storage import load_model
from rafiai.preprocessing import clean_text

saved = load_model()
model = saved["model"]
vectorizer = saved["vectorizer"]

sentences = [
    "This app helps me finish routine tasks in half the time.",
    "The new process frees up time for other work."
]

feature_names = vectorizer.get_feature_names_out()

for sentence in sentences:
    matrix = vectorizer.transform([clean_text(sentence)]).tocoo()

    contributions = []

    for index, value in zip(matrix.col, matrix.data):
        word = feature_names[index]

        score = value * (
            model.feature_log_prob_[1, index]
            - model.feature_log_prob_[0, index]
        )

        contributions.append((word, score))

    print("\nSentence:", sentence)
    print("-" * 60)

    for word, score in sorted(
        contributions,
        key=lambda item: abs(item[1]),
        reverse=True
    ):
        print(f"{word}: {score:+.4f}")
