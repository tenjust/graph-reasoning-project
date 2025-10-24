import logging
from typing import List

import numpy as np
from sklearn.metrics import f1_score
from torch import Tensor
from functools import partial, wraps
from time import time

def get_accuracy(preds: Tensor, label: Tensor):
    """
    :param preds: shape (batch_size, num_classes)
    :param label: shape (batch_size)
    """
    return (preds.argmax(dim=1) == label).sum().item() / len(label) * 100


def get_preds_and_rank(preds: Tensor, labels: Tensor):
    """
    :param preds: shape (batch_size, num_classes)
    :param label: shape (batch_size)
    :return pred_classes: shape (batch_size). Predicted class for each instance.
    :return ranks: shape (batch_size). Rank of the correct class. 0 is the best rank.
    """
    pred_classes = preds.argmax(dim=1)
    # ranks = preds > preds[labels]

    # get logit for correct class for each instacne
    correct_logit = preds[range(preds.shape[0]), labels]
    # get number of logits that are higher than the correct logit for each instance
    ranks = (preds > correct_logit.unsqueeze(1)).sum(dim=1)

    return pred_classes, ranks


def mrr_score(ranks: np.ndarray):
    return np.average(1 / (1 + ranks))


def accuracy_score(pred_classes: np.ndarray, labels: np.ndarray):
    return (pred_classes == labels).mean()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        logging.info(f'Function {func.__name__} took {total_time:.2f} seconds')
        return result

    return timeit_wrapper


@timeit
def get_metrics(
        pred_classes: List[int], ranks: List[int], labels: List[int], entailed_by_texts: List[bool]
        ):
    pred_classes = np.array(pred_classes)
    ranks = np.array(ranks)
    labels = np.array(labels)
    entailed_by_texts = np.array(entailed_by_texts)

    micro_f1 = f1_score(y_true=labels, y_pred=pred_classes, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=pred_classes, average='macro')
    mrr = mrr_score(ranks)
    accuracy = accuracy_score(pred_classes, labels)

    text_pred_classes = pred_classes[entailed_by_texts]
    text_ranks = ranks[entailed_by_texts]
    text_labels = labels[entailed_by_texts]
    text_micro_f1 = f1_score(y_true=text_labels, y_pred=text_pred_classes, average='micro')
    text_macro_f1 = f1_score(y_true=text_labels, y_pred=text_pred_classes, average='macro')
    text_mrr = mrr_score(text_ranks)
    text_accuracy = accuracy_score(text_pred_classes, text_labels)

    graph_pred_classes = pred_classes[~entailed_by_texts]
    graph_ranks = ranks[~entailed_by_texts]
    graph_labels = labels[~entailed_by_texts]
    graph_micro_f1 = f1_score(y_true=graph_labels, y_pred=graph_pred_classes, average='micro')
    graph_macro_f1 = f1_score(y_true=graph_labels, y_pred=graph_pred_classes, average='macro')
    graph_mrr = mrr_score(graph_ranks)
    graph_accuracy = accuracy_score(graph_pred_classes, graph_labels)

    out = {
        'all/micro_f1': micro_f1,
        'all/macro_f1': macro_f1,
        'all/mrr': mrr,
        'all/accuracy': accuracy,
        'text/micro_f1': text_micro_f1,
        'text/macro_f1': text_macro_f1,
        'text/mrr': text_mrr,
        'text/accuracy': text_accuracy,
        'graph/micro_f1': graph_micro_f1,
        'graph/macro_f1': graph_macro_f1,
        'graph/mrr': graph_mrr,
        'graph/accuracy': graph_accuracy,
    }
    return out
