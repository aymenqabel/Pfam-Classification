import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
import json


def read_data():
    data = {}
    data['train'] = pd.read_csv('data/train.csv')
    data['val'] = pd.read_csv('data/val.csv')
    data['test'] = pd.read_csv('data/test.csv')
    return data

def open_json(input_file):
    with open(input_file) as f:
        dictionary = json.load(f)
    return dictionary

def save_to_json(results, path_results, method):
    file_path = os.path.join(path_results, f"results_{method}.json")
    with open(file_path, 'w') as f:
        json.dump(results, f)
        
def get_results_identity(results, y_true, y_pred, names):
    identities = []
    identity_dict = open_json(f'data/identities.json')
    for name in names:
        identities.append(identity_dict[name])
    identities = np.array(identities)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    indices_less_than_50 = np.where((identities>= 0) & (identities <50))[0]
    results[f'accuracy_less_than_50'] = accuracy_score(y_true[indices_less_than_50], y_pred[indices_less_than_50])
    indices_greater_than_50 = np.where(identities>= 50)[0]
    results[f'accuracy_greater_than_50'] = accuracy_score(y_true[indices_greater_than_50], y_pred[indices_greater_than_50])
    indices_not_found = np.where(identities == -1)[0]
    results[f'accuracy_not_found'] = accuracy_score(y_true[indices_not_found], y_pred[indices_not_found])

    return results, identities

def compute_metrics(y_true, y_pred, path_results,  method = "BLAST", names=None):
    # Compute metrics and save then in a json file:
    results = {}
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    # results['precision'], results['recall'], results['fscore'] = precision.tolist(), recall.tolist(), fscore.tolist()
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    results['F1macro'] = np.mean(fscore)
    print(f"Balanced accuracy is : {results['balanced_acc']}")
    if names:
        results, identity_3 = get_results_identity(results, y_true, y_pred, names)
       
        df = pd.DataFrame(list(zip(names, y_true, y_pred, identity_3)),
               columns =['Names', 'True_class', 'Predicted_class', 'Identity_1e-3'])
        df.to_csv(os.path.join(path_results, f"results_{method}.csv"))
    # save results to json
    save_to_json(results, path_results, method)
    
    return results
