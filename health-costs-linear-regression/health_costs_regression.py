import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model():
    df = pd.read_csv("insurance.csv")

    df = pd.get_dummies(df, drop_first=True)

    y = df.pop("charges")
    X = df

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=[train_X.shape[1]]),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mae",
        metrics=["mae", "mse"]
    )

    model.fit(train_X, train_y, epochs=100, verbose=0)

    loss, mae, mse = model.evaluate(test_X, test_y, verbose=0)

    print("Testing set Mean Abs Error:", mae)

    return model