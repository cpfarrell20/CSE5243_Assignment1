import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(train_text, val_text, test_text, max_features=512):
    
    vectorizer = TfidfVectorizer(max_features=max_features, 
                                 sublinear_tf=False, 
                                 use_idf=True, smooth_idf=False)
    
    D_train = vectorizer.fit_transform(train_text)
    D_val = vectorizer.transform(val_text)
    D_test = vectorizer.transform(test_text)

    vocab = vectorizer.get_feature_names_out()

    return D_train, D_val, D_test, vocab

    