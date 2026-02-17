import time
import numpy as np
from matrix import load_data
from matrix import build_matrix
from split import split_data
from features import build_features
from train import train_models
from train import evaluate
from plots import compute_auroc
from plots import compute_roc

text, numbers = load_data("../data")
D, words = build_matrix(text)

y = np.array(numbers)
D_train, D_val, D_test, y_train, y_val, y_test = split_data(D, y)

n_train = D_train.shape[0]
n_val = D_val.shape[0]
train_text = text[:n_train]
val_text = text[n_train:n_train+n_val]
test_text = text[n_train+n_val:]
tfidf_train, tfidf_val, tfidf_test, tfidf_vocab = build_features(
    train_text, val_text, test_text, max_features=512)

for name, (Dtr, Dte) in {
    "BOW": (D_train, D_test),
    "TF-IDF": (tfidf_train, tfidf_test)
}.items():
    print("\n=====Using", name, "features=====")
    models = train_models(Dtr, y_train)
    for model_name, (model, time) in models.items():
        metrics = evaluate(model, Dte, y_test)
        auroc = compute_auroc(model, Dte, y_test)
        print(model_name, 
              "Time:", round(time,3),
              "Accuracy:", round(metrics["accuracy"],3),
              "Precision:", round(metrics["precision"],3),
              "Recall:", round(metrics["recall"],3),
              "Specificity", round(metrics["specificity"],3),
              "AUROC:", round(auroc,3))
        compute_roc(model, Dte, y_test, f"{model_name} ({name})",
            f"roc_{model_name}_{name}.png")