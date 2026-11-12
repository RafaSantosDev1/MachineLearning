
import re

from rafiai.data_loader import load_dataset

data = load_dataset("training.csv")

features = [
    "in",
    "up",
    "routine",
    "routine tasks",
    "the time",
]

for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 60)

    pattern = re.compile(
        r"\b" + re.escape(feature) + r"\b",
        re.IGNORECASE
    )

    for label in ["positive", "negative"]:
        matches = data[
            data["text"].apply(
                lambda text: bool(pattern.search(text))
            )
            & (data["label"] == label)
        ]

        print(f"\n{label.upper()}: {len(matches)} examples")

        for sentence in matches["text"].head(10):
            print("-", sentence)
