
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from rafiai.data_loader import load_dataset
from rafiai.preprocessing import clean_texts


data = load_dataset("training.csv")

X = data["text"]
y = data["label"]

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

alpha_values = [0.005, 0.02]

all_results = {}

for alpha in alpha_values:

    pipeline = Pipeline([
        ("cleaning", FunctionTransformer(
            clean_texts,
            validate=False
        )),
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=1
        )),
        ("classifier", MultinomialNB(alpha=alpha))
    ])

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro"
        ]
    )

    all_results[alpha] = results

    print(f"\n{'=' * 40}")
    print(f"ALPHA: {alpha}")
    print(f"{'=' * 40}")

    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        scores = results[f"test_{metric}"]

        print(
            f"{metric}: "
            f"Média = {scores.mean():.2%} | "
            f"Desvio = {scores.std():.2%}"
        )


# Paired comparison using the same 25 folds
scores_005 = all_results[0.005]["test_f1_macro"]
scores_002 = all_results[0.02]["test_f1_macro"]

differences = scores_002 - scores_005

wins = sum(difference > 0 for difference in differences)
losses = sum(difference < 0 for difference in differences)
ties = sum(difference == 0 for difference in differences)

print(f"\n{'=' * 40}")
print("COMPARAÇÃO EMPARELHADA")
print(f"{'=' * 40}")

print(f"Alpha 0.005: {scores_005.mean():.2%}")
print(f"Alpha 0.02:  {scores_002.mean():.2%}")

print(
    f"Diferença média (0.02 - 0.005): "
    f"{differences.mean() * 100:+.3f} pontos percentuais"
)

print(f"Alpha 0.02 ganhou: {wins} folds")
print(f"Alpha 0.005 ganhou: {losses} folds")
print(f"Empates: {ties} folds")
