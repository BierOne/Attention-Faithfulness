from sklearn.metrics import *
import numpy as np
from pandas.io.json._normalize import nested_to_record
from collections import defaultdict
import pandas as pd
from IPython.display import display
import torch

def calculate_violators(top_ans_deltas, weights):
    max_ind = weights.abs().topk(k=1, dim=1)[1]  # first or last
    max_weights = weights.gather(dim=1, index=max_ind).squeeze()
    max_weight_deltas = top_ans_deltas.gather(dim=1, index=max_ind).squeeze()
    ind = (max_weights * max_weight_deltas) < 0
    return ind


def compute_rank_correlation(map1, map2, flags=1, map_size=None):
    """
    Function that measures Spearman’s correlation coefficient between two maps:
    """

    def _rank_correlation_(map1_rank, map2_rank, map_size):
        n = torch.tensor(map_size)
        upper = 6 * torch.sum((map2_rank - map1_rank).pow(2), dim=1)  # [batch] or [batch, g]
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down))

    if map_size is None:
        map_size = map1.shape[1]

    # get the rank for each element, we use sort function two times
    map1 = map1.argsort(dim=1)  # [batch , num_objs] or [batch, num_objs, g]
    map1_rank = map1.argsort(dim=1) # [batch , num_objs] or [batch, num_objs, g]

    map2 = map2.argsort(dim=1)  # [batch , num_objs] or [batch, num_objs, 1]
    map2_rank = map2.argsort(dim=1)  # [batch , num_objs] or [batch, num_objs, 1]

    correlation = _rank_correlation_(map1_rank.float(), map2_rank.float(), map_size)
    return correlation * flags


def calculate_acc(y_true, y_pred):
    y_true = np.array(y_true)
    if y_pred.shape[-1] == 2:
        y_pred = np.argmax(y_pred, axis=-1)
    score = y_true == y_pred
    return score


def calc_metrics_classification(target, predictions) :
    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]
        predictions = np.array([1 - predictions, predictions]).T

    predict_classes = np.argmax(predictions, axis=-1)
    if len(np.unique(target)) < 4 :
        rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    else :
        rep = {}
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})
    if predictions.shape[-1] == 2 :
        rep.update({'roc_auc' : roc_auc_score(target, predictions[:, 1])})
        rep.update({"pr_auc" : average_precision_score(target, predictions[:, 1])})
    return rep

def calc_metrics_qa(target, predictions) :
    rep = {'accuracy' : accuracy_score(target, predictions)}
    return rep

def calc_metrics_regression(target, predictions) :
    rep = {}
    rep['rmse'] = np.sqrt(mean_squared_error(target, predictions))
    rep['mae'] = mean_absolute_error(target, predictions)
    rep['r2'] = r2_score(target, predictions)

    return rep

def calc_metrics_multilabel(target, predictions) :
    rep = {}
    target = np.array(target)
    nlabels = target.shape[1]
    predict_classes = np.where(predictions > 0.5, 1, 0)
    for i in range(nlabels) :
        rep_i = nested_to_record(classification_report(target[:, i], predict_classes[:, i], output_dict=True), sep='/')
        rep_i.update({'accuracy' : accuracy_score(target[:, i], predict_classes[:, i])})
        rep_i.update({'roc_auc' : roc_auc_score(target[:, i], predictions[:, i])})
        rep_i.update({"pr_auc" : average_precision_score(target[:, i], predictions[:, i])})
        for k in list(rep_i.keys()) :
            rep_i['label_' + str(i) + '/' + k] = rep_i[k]
            del rep_i[k]
            
        rep.update(rep_i)
    
    macro_roc_auc = np.mean([v for k, v in rep.items() if 'roc_auc' in k])
    macro_pr_auc = np.mean([v for k, v in rep.items() if 'pr_auc' in k])
    
    rep['macro_roc_auc'] = macro_roc_auc
    rep['macro_pr_auc'] = macro_pr_auc
    
    return rep

metrics_map = {
    'Single_Label' : calc_metrics_classification, 
    'Multi_Label' : calc_metrics_multilabel,
    'Regression' : calc_metrics_regression,
    'qa' : calc_metrics_qa
}

def print_metrics(metrics) :
    tabular = {k:v for k, v in metrics.items() if '/' in k}
    non_tabular = {k:v for k, v in metrics.items() if '/' not in k}
    print(non_tabular)

    d = defaultdict(dict)
    for k, v in tabular.items() :
        if not k.startswith('label_') :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v
        if '/1/' in k or 'auc' in k:
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))
        