from sklearn.model_selection import train_test_split

def split_data(D, number, train = 0.6, validation = 0.2, 
               test = 0.2, random = 1234):

    assert abs(train + validation + test - 1.0) < 1e-6

    D_train, D_temp, number_train, number_temp = train_test_split(
        D, number, test_size=(1-train), stratify=number, random_state=random)
    
    validation_ratio = validation / (validation + test)
    D_val, D_test, number_val, number_test = train_test_split(
        D_temp, number_temp, test_size=(1-validation_ratio), 
        stratify=number_temp, random_state=random)
    
    return D_train, D_val, D_test, number_train, number_val, number_test
    