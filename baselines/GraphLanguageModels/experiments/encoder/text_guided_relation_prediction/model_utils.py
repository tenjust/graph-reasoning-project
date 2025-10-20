import logging

from torch import nn

from baselines.GraphLanguageModels.models.graph_T5.classifier import GraphT5Classifier


def freeze_params(s:str, model:GraphT5Classifier) -> None:
    """
    :param s: string that specifies which parameters to train
    :param model: model
    """
    if s == "all":  # all
        pass
    elif s == "head":  # only head
        for param in model.t5model.parameters():
            param.requires_grad = False
    elif s == "PE":  # PE and head
        for param in model.t5model.parameters():  # set all encoder parameters to not trainable
            param.requires_grad = False
        for l in model.t5model.encoder.block:  # set PE to trainable
            if l.layer[0].SelfAttention.has_relative_attention_bias:
                logging.info('setting relative_attention_bias to trainable')
                l.layer[0].SelfAttention.relative_attention_bias.requires_grad = True
                assert l.layer[0].SelfAttention.relative_attention_bias.requires_grad
    else:
        raise ValueError(f"unknown parameters_to_train {s}")

def reset_params(model:GraphT5Classifier) -> None:
    for param in model.parameters():
        nn.init.normal_(param)
    for module in model.t5model.modules():
        model.t5model._init_weights(module)