import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def train_model():

    df = pd.read_csv(
        "sms.csv",
        sep="\t",
        encoding="latin-1"
    )

    df.columns = ["label", "text"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    return model


def predict_message(model, text):

    prediction = model.predict([text])[0]

    probability = max(model.predict_proba([text])[0])

    return [prediction, probability]