from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plot
from knn import predict_proba_knn

def compute_auroc(model, D, y):
    if hasattr(model, "decision_function"):
        score = model.decision_function(D)
    elif hasattr(model, "predict_proba"):
        score = model.predict_proba(D)[:,1]
    else:
        score = predict_proba_knn(model, D)

    return roc_auc_score(y, score)

def compute_roc(model, D, y, number, filename):
    if hasattr(model, "decision_function"):
        score = model.decision_function(D)
    elif hasattr(model, "predict_proba"):
        score = model.predict_proba(D)[:,1]
    else:
        score = predict_proba_knn(model, D)

    fp_rate, tp_rate, _ = roc_curve(y, score)

    plot.figure()
    plot.plot(fp_rate, tp_rate)
    plot.plot([0,1],[0,1],'--')
    plot.xlabel("False Pos Rate")
    plot.ylabel("True Pos Rate")
    plot.title(number)
    plot.savefig(filename)
    plot.close()