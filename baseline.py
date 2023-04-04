import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from data import load_train_val_test_ex
import sklearn.metrics
from metrics import compute_evo_table
from tqdm import tqdm
import numpy as np

reg = RandomForestRegressor()

train = [[0, 1, 2, 3],
         [0, 1, 2, 4],
         [0, 1, 3, 4],
         [0, 2, 3, 4],
         [1, 2, 3, 4]]

test = [4, 3, 2, 1, 0]
val = [5, 5, 5, 5, 5]

trues = []
preds = []

for tr, te, v in tqdm(zip(train, test, val)):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=tr,
                                                                                                val=v,
                                                                                                test=te,
                                                                                                ntl_type='viirs')

    X_train_resh = X_train.mean(axis=(-2, -3)).reshape(-1, 21*6)

    reg.fit(X_train_resh, y_train)

    X_test_resh = X_test.mean(axis=(-2, -3)).reshape(-1, 21*6)

    pred = reg.predict(X_test_resh)

    preds.append(pred)
    trues.append(y_test)

preds = np.array(preds)
trues = np.array(trues)

compute_evo_table({'rf':trues}, {'rf':preds}, results_path='dar_es_salam/zone-zanzibar_ps-32/all_test_examples/')
plt.show()


