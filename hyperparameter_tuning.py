import time 
from tqdm import tqdm
import os 
import pickle 
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def save_checkpoint(model, output_dir, filename):
    if not os.path.isfile(os.path.join(output_dir, filename)):
        with open(os.path.join(output_dir, filename), 'wb') as f:
            print('Saving model information to {}'.format(os.path.join(output_dir, filename)))
            pickle.dump(model, f)
def load_checkpoint(output_dir, filename):
    if os.path.isfile(os.path.join(output_dir, filename)):
        print('Loading model information from {}'.format(os.path.join(output_dir, filename)))
        with open(os.path.join(output_dir, filename), 'rb') as f:
            return pickle.load(f)  
    return None  
#params format: {"param1": [val1, val2, val3], "param2": [val1, val2, val3]}
def train(model_name, params, X_train, X_val, y_train, y_val, output_dir, prefix):
    train_scores = []
    val_scores = []
    fit_time = []
    models = []
    best_train_score = 0
    best_val_score = 0
    best_fit_time = None
    best_model = None 
    keys, values = zip(*params.items())
    params_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(params_combination)
    for params_comb in tqdm(params_combination):
        print()
        filename = prefix
        for key, value in params_comb.items():
            filename += key + '_' + str(value) + '_'
            print('{}: {}'.format(key, value)) 
        filename = filename[:-1] + '.pkl' 
        checkpoint_model = load_checkpoint(output_dir, filename)
        if checkpoint_model is not None: 
            continue
        if model_name == 'logistic':
            model = LogisticRegression(max_iter=10000)
        elif model_name == 'linear_svm':
            model = LinearSVC(max_iter=10000)
        elif model_name == 'kernel_svm':
            model = SVC()
        model.set_params(**params_comb)
        start = time.time()
        _ = model.fit(X_train, y_train)
        models.append(model)
        ft = time.time() - start
        fit_time.append(ft)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        train_scores.append(train_score)
        model_info = {"model": model, "train_score": train_score, "validation_score": val_score, "fit_time": ft}
        save_checkpoint(model_info, output_dir, filename)
        val_scores.append(val_score)
        if (val_score > best_val_score):
            best_train_score = train_score
            best_val_score = val_score 
            best_fit_time = ft
            best_model = model 
        print('Train score: {}'.format(train_score))
        print('Validation score: {}'.format(val_score))
        print('-'*100)
    best_model_info = {"model": best_model, "train_score": best_train_score, "validation_score": best_val_score, "fit_time": best_fit_time}
    save_checkpoint(best_model_info, output_dir, "best_model.pkl")
    return {"best_model": best_model, "best_score": best_val_score,"models": models, "train_scores": train_scores, "validation_scores": val_scores, "fit_time": fit_time}