import json
import logging
from typing import Dict, List

import h5py
import torch
from torch import tensor

from baselines.GraphLanguageModels.models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from baselines.GraphLanguageModels.models.graph_T5.wrapper_functions import Graph, Data, \
    add_text_to_graph_data, graph_to_set_of_triplets, graph_to_graphT5


class OpenData:
    def __init__(self, use_graph: bool, entailed_triplets_only: bool):
        use_graph_name = '' if use_graph else '-triplet_only'
        entailed_triplets_only_name = '-text_entailed_only' if entailed_triplets_only else ''

        self.fn = f'data/rebel_dataset/rebel{use_graph_name}{entailed_triplets_only_name}.hdf5'

    def __enter__(self):
        self.f = h5py.File(self.fn, 'r')
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()


def data_to_dataT5(
        graph: Graph, text: str, mask_origin: str, tokenizer: T5Tokenizer, label: int,
        graph_representation: str, eos: str, use_text: str
        ):
    """
    :param graph: graph to convert
    :param text: text to convert
    :param mask_origin: whether the mask is entailed by text (--> text) or not (--> graph)
    :param tokenizer: tokenizer of model
    :param label: label of the relation
    :param graph_representation: how to represent the graph.
    :param eos: end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which preserves locality when using the local GLM
    :param use_text: whether and how to use text as input. Can be `False` for not using text, `FullyConnected` having a full attention matrix with T2G and G2T attention.
    """

    if graph_representation == 'lGLM':
        data = graph_to_graphT5(graph, tokenizer, how='local', eos=eos)
    elif graph_representation == 'set':
        data = graph_to_set_of_triplets(graph, tokenizer, order='random')
    elif graph_representation == 'gGLM':
        data = graph_to_graphT5(graph, tokenizer, how='global', eos=eos)
    elif graph_representation == 'list':
        data = graph_to_set_of_triplets(graph, tokenizer, order='alphabetical')
    else:
        raise ValueError(f"unknown graph_representation {graph_representation}")

    add_text_to_graph_data(data=data, text=text, tokenizer=tokenizer, use_text=use_text)

    data.label = tensor(label)
    data.mask_origin = mask_origin
    return data


def get_data_instances(
        data: OpenData, split: str, data_indices: List[int], tokenizer: T5Tokenizer,
        graph_representation: str, eos: str, use_text: str
        ) -> List[Data]:
    """
    :param data: data
    :param split: split of data
    :param data_indices: indices of data instances to get
    :param tokenizer: tokenizer of model
    :param graph_representation: how to represent the graph.
    :param eos: how to handle end of sentence token
    :param use_text: how to handle text
    """

    with data as dat:
        ds = [json.loads(dat[split][i]) for i in data_indices]

    try:
        data_instances = [
            data_to_dataT5(graph=Graph(d['triplets']), text=d['text'], mask_origin=d['mask_origin'],
                           tokenizer=tokenizer, label=d['label'],
                           graph_representation=graph_representation, eos=eos, use_text=use_text)
            for d in ds
        ]
    except Exception as e:
        logging.error(f"error when processing {ds}")
        d = ds[0]
        graph = Graph(d['triplets'])
        logging.debug(graph)
        raise e
    return data_instances


def get_batch(
        data: OpenData, split: str, data_indices: List[int], device: str, tokenizer: T5Tokenizer,
        graph_representation: str, eos: str, use_text: str, max_seq_len: int, predict_source: bool,
        source_to_index: Dict[str, int]
        ):
    """
    can be implemented more efficiently with nested tensors, but they are currently unstable
    """
    data_instances = get_data_instances(data=data, split=split, data_indices=data_indices,
                                        tokenizer=tokenizer,
                                        graph_representation=graph_representation, eos=eos,
                                        use_text=use_text)

    current_max_seq_len = max([data.input_ids.shape[1] for data in data_instances])
    max_seq_len = min(max_seq_len, current_max_seq_len)

    if data_instances[0].relative_position is None:
        # the undefined option, should never go here
        assert data_instances[0].sparsity_mask is None
        assert data_instances[0].use_additional_bucket is None
        is_sequence_transformer = True
    else:
        assert data_instances[0].sparsity_mask is not None
        assert data_instances[0].use_additional_bucket is not None
        is_sequence_transformer = False

    # intialize tensors
    input_ids = torch.ones((len(data_instances), max_seq_len), dtype=torch.long,
                           device=device) * tokenizer.pad_token_id
    if not is_sequence_transformer:
        relative_position = torch.zeros((len(data_instances), max_seq_len, max_seq_len),
                                        dtype=torch.long, device=device)
        sparsity_mask = torch.zeros((len(data_instances), max_seq_len, max_seq_len),
                                    dtype=torch.bool, device=device)
        use_additional_bucket = torch.zeros((len(data_instances), max_seq_len, max_seq_len),
                                            dtype=torch.bool, device=device)

    # fill tensors
    for i, data in enumerate(data_instances):
        instance_len = min(data.input_ids.shape[1], max_seq_len)
        input_ids[i, :instance_len] = data.input_ids[:, :instance_len]
        if not is_sequence_transformer:
            assert data.input_ids.shape[1] == data.relative_position.shape[1] == \
                   data.relative_position.shape[2] == data.sparsity_mask.shape[1] == \
                   data.sparsity_mask.shape[2] == data.use_additional_bucket.shape[1] == \
                   data.use_additional_bucket.shape[2]
            relative_position[i, :instance_len, :instance_len] = data.relative_position[:,
                                                                 :instance_len, :instance_len]
            sparsity_mask[i, :instance_len, :instance_len] = data.sparsity_mask[:, :instance_len,
                                                             :instance_len]
            use_additional_bucket[i, :instance_len, :instance_len] = data.use_additional_bucket[:,
                                                                     :instance_len, :instance_len]

    if is_sequence_transformer:
        relative_position = None
        sparsity_mask = None
        use_additional_bucket = None

    indices = [data.indices for data in data_instances]
    label = torch.tensor([data.label for data in data_instances], device=device)
    entailed_by_text = torch.tensor([data.mask_origin == 'text' for data in data_instances],
                                    device=device, dtype=torch.bool)

    if predict_source:
        source_label = torch.tensor([source_to_index[data.mask_origin] for data in data_instances],
                                    device=device, dtype=torch.long)
    else:
        source_label = None

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label


def chunker(data_list: list, batch_size: int):
    """
    returns a generator that yields batches of size batch_size
    """
    return (data_list[pos:pos + batch_size] for pos in range(0, len(data_list), batch_size))