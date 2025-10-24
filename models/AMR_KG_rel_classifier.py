from __future__ import annotations

import warnings
from itertools import chain

from torch import nn
from torch import Tensor
import torch
import logging

from transformers import T5Config
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Block, T5LayerFF, \
    T5LayerSelfAttention, T5Stack, T5Attention
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models.GraphLanguageModels.experiments.encoder.text_guided_relation_prediction.data_utils import \
    data_to_dataT5
from models.GraphLanguageModels.models.graph_T5.wrapper_functions import Graph, Data
from data.utils import load_hdf5_data


def clamp_inf_values(states: Tensor) -> Tensor:
    """ Clamp inf values to enable fp16 training - fully from original T5 implementation """
    if states.dtype == torch.float16 and torch.isinf(states).any():
        clamp_value = torch.finfo(states.dtype).max - 1000
        return torch.clamp(states, min=-clamp_value, max=clamp_value)
    return states


def init_tensors(batch_dim: int, max_seq_len: int, device=None) -> tuple[Tensor ,...]:
    """
    Initialize relative position, sparsity mask, and additional bucket mask tensors.
    They are the same except for their data types.
    :param batch_dim: batch size
    :param max_seq_len: maximum sequence length
    :param device: device to create the tensors on
    :return: tuple of initialized tensors
    """
    relative_position = torch.zeros(
        (batch_dim, max_seq_len, max_seq_len), dtype=torch.long, device=device
    )
    sparsity_mask = torch.zeros(
        (batch_dim, max_seq_len, max_seq_len), dtype=torch.bool, device=device
    )
    use_additional_bucket = torch.zeros(
        (batch_dim, max_seq_len, max_seq_len), dtype=torch.bool, device=device
    )
    return relative_position, sparsity_mask, use_additional_bucket


def shape_GLM_tensors(
    data: tuple[Tensor, ...], tokenizer: T5TokenizerFast, model_max_length: int, device
) -> tuple[Tensor ,...]:
    """
    Shape the GLM tensors for a batch of data instances. The returned tensors are:
    - input_ids,
    - relative_position,
    - sparsity_mask,
    - use_additional_bucket
    :param data: list of tensors (input_ids, relative_position, sparsity_mask, use_additional_bucket)
    :param tokenizer: tokenizer to use for padding
    :param model_max_length: maximum sequence length for the model
    :param device: device to create the tensors on
    :return: tuple of tensors: input_ids, relative_position, sparsity_mask, use_additional_bucket
    """
    batch_max_seq_len = max([d[0].shape[1] for d in data])
    max_seq_len = min(model_max_length, batch_max_seq_len)
    data_input_ids = data[0]
    data_tensors = data[1:]
    num_instances = data_input_ids.shape[0]
    # refining input_ids tensor
    input_ids = torch.ones(
        (num_instances, max_seq_len), dtype=torch.long, device=device,
    ) * tokenizer.pad_token_id
    # tensors: relative_position, sparsity_mask, use_additional_bucket
    tensors = init_tensors(num_instances, max_seq_len, device)
    # check that all tensors have the same seq_len (taking into account the first batch dim)
    shapes = [data_input_ids.shape[2], *chain.from_iterable(t.shape[2:] for t in data_tensors)]
    assert len(set(shapes)) == 1, f"Inconsistent shapes: {shapes}"
    instance_len = min(data_input_ids.shape[2], max_seq_len)
    for batch in range(data_input_ids.shape[0]):
        # fill in the tensors with data from the instance
        input_ids[batch, :instance_len] = data_input_ids[:, :instance_len]
        for j, d_tensor in enumerate(data_tensors):
            tensors[j][batch, :instance_len, :instance_len] = \
                d_tensor[:, :instance_len, :instance_len]

    return input_ids, *tensors


def preprocess(data_instance: dict, tokenizer: T5TokenizerFast, use_amr: bool) -> Data:
    """
    Preprocess a data instance into a Data object suitable for GraphT5 model
    :param data_instance: input data instance with triplets, text, mask_origin, and label
    :param tokenizer: tokenizer to use for text processing
    : use_amr: whether to use AMR-based triplets (not supported yet)
    """
    if use_amr and "AMR_based_triplets" in data_instance:
        raise NotImplementedError("Using AMR_based_triplets is not supported yet, sorry!")
    elif use_amr and "AMR_based_triplets" not in data_instance:
        warnings.warn(
            "use_amr is True, but no AMR_based_triplets found in data_instance! "
            "(not supported yet)"
        )
    args = ["triplets", "text", "mask_origin", "label"]
    if any(arg not in data_instance for arg in args):
        raise ValueError(f"Data instance must contain fields: {args}")
    kg = Graph(data_instance['triplets'])
    processed_instance = data_to_dataT5(
        graph=kg,
        text=data_instance['text'],
        mask_origin=data_instance['mask_origin'],
        tokenizer=tokenizer,
        label=data_instance['label'],
        graph_representation="gGLM",
        eos='False',  # following GLM paper, no eos token
        use_text='FullyConnected',
    )
    return processed_instance


class Decorators:
    """ Collection of decorators for Graph2GraphRelationClassifier model methods """
    @staticmethod
    def check_stack_kwargs(stack_forward):
        """ Wrapper to call the T5Stack forward method with the correct arguments """
        def inner_fn(self, **kwargs):
            # Optional arguments:
            # "head_mask", "attention_mask", "inputs_embeds", "output_attentions",
            # "output_hidden_states", "return_dict"
            obligatory_args = {
                "input_ids", "relative_position", "sparsity_mask", "use_additional_bucket"
            }
            key_args = set(kwargs.keys())
            assert obligatory_args.issubset(key_args), \
                f"Expected keys: {obligatory_args}, but got {key_args}"
            return stack_forward(self, **kwargs)
        return inner_fn

    @staticmethod
    def check_block_attn_kwargs(block_forward):
        """ Wrapper to call the T5Block forward method with the correct arguments """
        def inner_fn(self, **kwargs):
            obligatory_args = {
                "mask", # attention_mask is meant (renamed due to kwargs clash)
                "hidden_states", "position_bias", "layer_head_mask",  # new GLM
                "output_attentions", "relative_position", "sparsity_mask",  # new GLM
                "use_additional_bucket",  # new GLM
            }
            key_args = set(kwargs.keys())
            missing_args = obligatory_args.difference(key_args)
            if missing_args:
                raise ValueError(f"Missing obligatory key-word arguments: {missing_args}")
            else:
                assert obligatory_args == key_args, f"Expected keys: {obligatory_args}, but got {key_args}"
            return block_forward(self, **kwargs)
        return inner_fn

    @staticmethod
    def graph_preprocessing(forward_fn):
        """ Wrapper for preprocessing the graph-related inputs before passing them to the model """
        def inner_fn(self, data: dict | list[dict], use_amr: bool) -> tuple[Tensor, ...]:
            """
            Preprocess the input data (single instance or batch) and prepare tensors for the model.
            The code of this function is heavily inspired by the GLM processing pipeline
            (see get_batch function in train_LM.py).

            :param self: model instance
            :param data: input data instance or batch of instances
            :param use_amr: whether to use AMR-based triplets (not supported yet)
            :return: tensors of preprocessed inputs for the model are passed to forward_fn
            """
            logging.debug(f"using a decorator for graph preprocessing, use_amr={use_amr}")
            if type(data) is list:
                # batch of data instances
                data = [preprocess(data_instance, self.tokenizer, use_amr) for data_instance in data]
                input_ids = torch.cat([d.input_ids.unsqueeze(0) for d in data], dim=0)
                relative_position = torch.cat([d.relative_position.unsqueeze(0) for d in data], dim=0)
                sparsity_mask = torch.cat([d.sparsity_mask.unsqueeze(0) for d in data], dim=0)
                use_additional_bucket = torch.cat([d.use_additional_bucket.unsqueeze(0) for d in data], dim=0)
                all_tensors = (input_ids, relative_position, sparsity_mask, use_additional_bucket)
                tensors = shape_GLM_tensors(all_tensors, self.tokenizer, self.config.model_max_length, self.device)
                # TODO: redefine or stack?
                self.labels = torch.tensor([d.label for d in data], device=device)
                return forward_fn(self, *tensors)
            else:
                # single data instance
                warnings.warn(
                    "Single data instance passed to the model, double-check the shape handling!"
                )
                data = preprocess(data, self.tokenizer, use_amr)
                self.labels = torch.tensor([data], device=device)
                outputs = (data.input_ids,
                           data.relative_position,
                           data.sparsity_mask,
                           data.use_additional_bucket)
                return forward_fn(self, *outputs)
        return inner_fn


class GraphAttention(T5Attention):
    FORWARD_RUNS = 0  # class variable to count the number of forward runs

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        rel_attn_num_additional_buckets = config.to_dict().get('rel_attn_num_additional_buckets', 0)
        # Relative attention buckets are increased by the number of additional buckets
        config.relative_attention_num_buckets += rel_attn_num_additional_buckets

        super().__init__(config, has_relative_attention_bias)
        self.rel_attn_num_additional_buckets: int = rel_attn_num_additional_buckets
        self.model_name_size = config.model_name_size
        self.default_num_buckets: int = (
                self.relative_attention_num_buckets - self.rel_attn_num_additional_buckets
        )

        # Removed the possibility from the GLM project for init_additional_buckets_from to be None,
        # as that would lead to random initialization of the additional buckets, while we want to
        # implement the version with initialization at infinity
        self.additional_buckets_init_dist = torch.inf
        if self.has_relative_attention_bias:
            self.init_relative_position_bias()

    def _relative_position_bucket(
        self,
        relative_position: Tensor,
        bidirectional: bool = False,
        additional_bucket_id: int | Tensor = None,
    ) -> Tensor:
        """
        Method overridden to support additional buckets for global graph-to-graph relative position.
        [GLM] Copied from transformers.models.t5.modeling_t5.T5Attention with slight modifications

        :param relative_position: Tensor of shape (query_length, key_length) with relative positions
        :param bidirectional: whether the attention is bidirectional
        :param additional_bucket_id: Tensor of shape (query_length, key_length) with a mask -
                                     boolean values indicate whether to use additional bucket for the
                                     corresponding position. If None, no additional buckets are used.
        :return: Tensor of shape (query_length, key_length) with bucketed relative positions
        """
        if not self.relative_attention_num_buckets:
            # num_buckets: Tensor = 32,
            raise ValueError(f"relative_attention_num_buckets must be > 0, "
                             f"not {self.relative_attention_num_buckets}")

        if not self.relative_attention_max_distance:
            # max_distance: Tensor = 128,
            raise ValueError(f"relative_attention_max_distance must be > 0, "
                             f"not {self.relative_attention_max_distance}")

        # relative_position shape: (query_length, key_length)
        relative_buckets = T5Attention._relative_position_bucket(
            relative_position,
            bidirectional,
            num_buckets=self.default_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # use additional bucket id to shift the relative buckets,
        # only for positions corresponding to True in additional_bucket_id
        if additional_bucket_id is not None:
            # In GLM:
            # `relative_buckets[use_additional_bucket] = \
            #                               relative_position[use_additional_bucket] + num_buckets`
            # This line never worked here (caused index issues, as values in relative_buckets
            # ended up exceeding the Embedding size) for a reason I couldn't discover, so I followed
            # the idea of additional buckets taking the most distant position in gGLM

            # -1 to keep indexing safe
            relative_buckets[additional_bucket_id] = self.relative_attention_num_buckets - 1
        return relative_buckets

    def init_relative_position_bias(self) -> None:
        """
        [GLM] Initializes parameters for relative position bias. This is necessary if additional
        buckets are used, as then the weights are not initialized automatically when calling
        `from_pretrained`.

        The relative position bias is initialized from a pretrained T5 model
        (the same instance as the one being loaded).
        """
        logging.debug('init normal buckets')
        if self.is_decoder:
            raise NotImplementedError("Decoder is not implemented.")

        with torch.no_grad():
            loaded_model_bias = self.relative_attention_bias.weight

            if not self.rel_attn_num_additional_buckets:
                return

            additional_init_distances = Tensor(
                [self.additional_buckets_init_dist] * self.rel_attn_num_additional_buckets
            )
            init_add_buckets_from = self._relative_position_bucket(
                relative_position=additional_init_distances,
                bidirectional=False,
                additional_bucket_id=None
            )

            logging.debug('initializing relative position bias..')
            add_bucket_start_inx = self.default_num_buckets - self.rel_attn_num_additional_buckets
            for bucket_inx, init_inx in enumerate(init_add_buckets_from, add_bucket_start_inx):
                init_inx = int(init_inx.item())
                self.relative_attention_bias.weight[bucket_inx, :] = loaded_model_bias[init_inx, :]

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        device=None,
        relative_position: Tensor = None,
        additional_bucket_id: int | Tensor = None,
    ) -> Tensor:
        """
        [T5] Compute binned relative position bias.

        :param query_length: length of the query sequence
        :param key_length: length of the key sequence
        :param device: device to perform the computation on
        :param relative_position: Tensor of shape (query_length, key_length) with relative positions.
            If None, relative positions are computed as in a standard sequence-to-sequence model.
        :param additional_bucket_id: Tensor of shape (query_length, key_length) with a mask -
                                     boolean values indicate
        :return: whether to use additional bucket for the corresponding position.
        """
        if query_length != key_length:
            raise ValueError("The length of query and key are different!")
            # => then it makes sense to have to vars
        if device is None:
            device = self.relative_attention_bias.weight.device

        # In GLM code, once created, relative_position is not updated anymore
        # - does a whole given sequence use only one relative_position?
        # If so, becomes rather *absolute*_position? On the other hand,
        # "In this work we thus focus on relative PE."
        if relative_position is None:
            logging.debug("creating relative position from scratch")
            # shape: (seq_length, 1)
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            # shape: (1, seq_length)
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            logging.debug(f"received relative_position of shape {relative_position.shape}")
        else:
            logging.debug(f"using passed relative_position: {relative_position.shape}")
            # Using the following line leads to shape mismatch later on as context_position
            # unsqueezes relative_position again
            # (It was also commented out in GLM implementation)
            # context_position = relative_position[:, None].to(device) # shape: (seq_length, 1)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (query_length, key_length)
            bidirectional=(not self.is_decoder),
            additional_bucket_id=additional_bucket_id,
        )
        max_rel_pos = relative_position_bucket.max()
        assert max_rel_pos < self.relative_attention_num_buckets, \
            (f"Max bucket index {max_rel_pos} exceeds number of buckets "
             f"{self.relative_attention_num_buckets}")

        values = self.relative_attention_bias(relative_position_bucket)
        # values.shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # values.shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        query_length=None,
        output_attentions=False,
        use_cache=False,  # None in GLM
        layer_head_mask=None,  # new to GLM
        relative_position=None,  # new to GLM
        sparsity_mask=None,  # new to GLM
        use_additional_bucket=None,  # new to GLM
    ):
        """
        [GLM] Self-attention (if key_value_states is None) or attention over source sentence
        (provided by key_value_states). Due to decoder not being used, only self-attention is calculated.

        :param hidden_states: input to the attention layer. It is a tensor of shape
                              [batch_size, seq_length, dim].
        :param mask: attention mask. It is a tensor of shape (batch_size, key_length) (non-causal)
                     or (batch_size, key_length, key_length)
        :param position_bias: position bias for the attention. If `None`, it will be computed as
                              in a standard sequence-to-sequence model. If not `None`, it will be
                              used as the position bias for the attention. It is a tensor of shape
                              [batch_size, n_heads, query_length, key_length].
        :param query_length: length of the query sequence
        :param output_attentions: whether to output attention weights
        :param use_cache: whether to use cache (not used in GLM)
        :param layer_head_mask: head mask for the attention. If `None`, it will be computed as in
                                a standard sequence-to-sequence model. If not `None`, it will be
                                used as the head mask for the attention. It is a tensor of shape
                                [n_heads].
        :param relative_position: [GLM] relative position for the attention. If `None`, it will be
                                  computed as in a standard sequence-to-sequence model. If not `None`,
                                  it will be used as the relative position for the attention.
                                  It is a tensor of shape [batch_size, query_length, key_length].
        :param sparsity_mask: [GLM] sparsity mask for the attention. If `None`, it will be computed
                              as in a standard sequence-to-sequence model. If not `None`, it will
                              be used as the sparsity mask for attention. It is a tensor of shape
                              [batch_size, query_length, key_length]. A value of 1 means that
                              the corresponding attention weight is not masked, and a value of
                              0 means that the corresponding attention weight is masked. Hence,
                              the sparsity mask is a binary mask that (kind of) can be used like a
                              multiplicative mask.
        :param use_additional_bucket: [GLM] whether to use additional buckets for the attention.
                                      If `None`, only standard positional encodings will be used.
                                      If not `None`, additional buckets will be used for the
                                      relative position. It is a tensor of shape
                                      [batch_size, query_length, key_length]. A value of False means
                                      that the corresponding position is a standard relative position,
                                      and a value of True means that the corresponding additional
                                      bucket should be used.
        """
        GraphAttention.FORWARD_RUNS += 1
        logging.debug(f"running GraphAttention forward, run {GraphAttention.FORWARD_RUNS}")

        batch_size, seq_length = hidden_states.shape[:2]

        # get query states: (batch_size, n_heads, seq_length, dim_per_head)
        query_states = self.shape(self.q(hidden_states))

        # get key/value states
        key_states = self.project(hidden_states, self.k)
        value_states = self.project(hidden_states, self.v)

        # [T5] compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states),
        # compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # always true in GLM, because position_bias is hardcoded to None in T5Block.forward
        if position_bias is None:
            logging.debug("no position_bias, computing it")
            if not self.has_relative_attention_bias:
                logging.debug("no relative attention bias, initializing position_bias from zeros")
                position_bias_shape = (batch_size, self.n_heads, seq_length, seq_length)
                position_bias = torch.zeros(position_bias_shape, device=scores.device, dtype=scores.dtype)
                # self.gradient_checkpointing is hardcoded to False both in source code and GLM
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                logging.debug("relative attention bias is passed to GraphAttention forward")
                # new GLM stuff:
                # relative_position is created in get_batch (train_LM.py)
                # and passed through all the way to here
                if relative_position is None:
                    logging.debug("relative position is not passed to GraphAttention forward")
                    assert use_additional_bucket is None
                    position_bias = self.compute_bias(
                        seq_length, seq_length, device=scores.device, additional_bucket_id=None)
                else:
                    logging.debug("relative position is passed to GraphAttention forward")
                    # use_additional_bucket, relative_position: (batch_size, seq_length, key_length)
                    position_bias = []
                    for position, bucket_ids in zip(relative_position, use_additional_bucket):
                        # bucket_ids: (seq_length, key_length): position
                        position_bias += self.compute_bias(
                            *bucket_ids.shape,
                            device=scores.device,
                            relative_position=position,
                            additional_bucket_id=bucket_ids
                        )
                    position_bias = torch.stack(position_bias, dim=0) # ensure batch dimension

            if mask:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias

        if sparsity_mask is not None:
            assert sparsity_mask.dtype == torch.bool, f"{relative_position.dtype} should be torch.bool"
            # [GLM] add extra dimension for heads and negate for indexing the masked positions
            sparsity_mask_mirorred = ~ sparsity_mask.unsqueeze(1)
            sparsity_mask_mirorred = sparsity_mask_mirorred.expand_as(position_bias)

            # >>> masked softmax >>>
            # [GLM] Sparsity_mask is initialized in _get_graphT5_relativeposition_sparsitymask
            # (wrapper_functions.py). This works in the backward pass, because potential nan values
            # that the softmax produces in the forward pass are not used in backpropagation,
            # because the "=" is independent of the value that the entry had previously.
            # This is not the case for "+=", which is why we need to set the values to -inf
            # instead of adding -inf.
            scores[sparsity_mask_mirorred] = float('-inf')

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        # [GLM] replace nan values in the attention weights with 0. nan happens if all positions
        # are masked for one token, as then all inputs to softmax are -inf for that token
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # <<< masked softmax <<<

        assert type(self.dropout) is float, "Expected float type for dropout probability"
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        if layer_head_mask:
            attn_weights = attn_weights * layer_head_mask

        attn_output = self.unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output) # (batch_size, seq_length, dim)

        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs # (attn_output, position_bias, opt: attn_weights)

    def shape(self, states):
        """ [T5] Project states to (batch_size, n_heads, seq_length, key_value_proj_dim) """
        batch_size = states.size(0)
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(self, states):
        """ [T5] Reshape states back to (batch_size, seq_length, inner_dim) """
        batch_size = states.size(0)
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(self, hidden_states, proj_layer):
        """ [T5] Projects hidden states correctly to key/query states """
        # self-attn
        # (batch_size, n_heads, key_length, dim_per_head)
        hidden_states = self.shape(proj_layer(hidden_states))
        # self-attn, too, but past_key_value is never used in GLM
        # hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        return hidden_states


class Graph2GraphRelationClassifier(PreTrainedModel):
    """
    This class implements a KG relation classifier that is based on T5 model architecture
    and guided by AMR-based graphs. This implementation follows the idea of Graph Language
    Models (GLM) project while incorporating the original code of T5 whenever possible.
    The configuration is simplified to gGLM and focuses on the relation classification task.
    """
    def __init__(self, config: T5Config):
        if config.is_decoder:
            warnings.warn("decoder is not implemented, setting 'is_decoder' to False.")
        config.is_decoder = False
        if config.use_cache:
            warnings.warn("use_cache is not needed for encoder-only model, setting use_cache to False.")
            config.use_cache = False
        super().__init__(config, layer_idx=None)
        self.model_name_size: str = self.config.model_name_size

        var = "rel_attn_num_additional_buckets"
        assert hasattr(config, var), \
            f"For running Graph Language Model's method T5Config must have attribute '{var}'"

        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.model_name_size,
            model_max_length=self.config.model_max_length,
        )
        self.t5 = T5EncoderModel.from_pretrained(
            self.model_name_size, config=self.config, ignore_mismatched_sizes=True,
        )
        self.configure_t5()

        self.classifier = nn.Linear(
            self.t5.shared.embedding_dim, self.config.num_classes, bias=True
        )
        self.softmax = nn.Softmax(dim=-1)

        logging.debug(f"model architecture:\n{self}")

        self.labels = None  # to be set externally if needed

    def configure_t5(self):
        """ Configure the T5 model to fit the needs of this implementation """
        self.t5.model_parallel = False
        self.t5.device_map = None

        # monkey-patching the forward method of T5Stack
        self.t5.encoder.forward = self.stack_forward.__get__(self.t5.encoder)
        self.t5.encoder.gradient_checkpointing = False

        for layer, block in enumerate(self.t5.encoder.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))
            block.forward = self.block_forward.__get__(block)
            # remove the original packaging for self-attention and feed-forward layers
            # to add them to the T5Block outside the ModuleList
            # for easier access during forward pass and readability
            block.layer = nn.Identity()
            # create a separate self-attention layer
            first_layer = True if layer == 0 else False
            block.self_attn = T5LayerSelfAttention(
                self.config, has_relative_attention_bias=first_layer
            )
            # monkey-patch the forward method of self-attention
            block.self_attn.forward = self.self_attn_forward.__get__(block.self_attn)
            # create a new feed forward layer
            block.feed_forward = T5LayerFF(self.config)
            # initialize GraphAttention instead of T5Attention
            block.self_attn.SelfAttention = GraphAttention(
                self.config, has_relative_attention_bias=first_layer
            )

    # @Decorators.graph_preprocessing  # commented out to allow direct tensor input
    def forward(
            self,
            input_ids: Tensor,
            relative_position: Tensor,
            sparsity_mask: Tensor,
            use_additional_bucket: Tensor,
            # indices: dict,
            attention_mask: Tensor = None
    ) -> Tensor:
        """
        Forward method of the classifier. One is expected to pass loaded rebel data
        and a flag for whether to use AMR triples for guidance instead of the raw text.
        The data will be preprocessed and converted into tensors suitable for the model
        in the decorated graph_preprocessing method.

        :param input_ids: torch.Tensor of shape (batch_size, seq_length)
        :param relative_position: torch.Tensor of shape (batch_size, seq_length, seq_length
        :param sparsity_mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param use_additional_bucket: torch.Tensor of shape (batch_size, seq_length,
            seq_length)
        :param attention_mask: not used and is only here for compatibility with transformers.Trainer
        :return: logits tensor of shape (batch_size, seq_length, num_classes)
        """
        logging.debug(f"running on device: {self.device}")
        args = ["input_ids", "relative_position", "sparsity_mask", "use_additional_bucket"]
        tensors = (input_ids, relative_position, sparsity_mask, use_additional_bucket)
        tensors = shape_GLM_tensors(tensors, self.tokenizer, self.config.model_max_length, self.device)
        kwargs = {arg: tensors[i] for i, arg in enumerate(args)}
        logging.debug('T5 encoder model is called')
        # outputs: (last_hidden_state, hidden_states, attentions)
        output = self.t5.encoder(**kwargs)  # (batch_size, seq_len, hidden_size)

        logging.debug('classifier is called')
        logits = self.classifier(output[0])  # (batch_size, seq_len, num_classes)
        predictions = self.get_classes(logits)  # scalar class predictions: (seq_len)
        return predictions

    def get_prob_distribution(self, logits: Tensor) -> Tensor:
        """ [GLM] Get the probability distribution over classes """
        return self.softmax(logits)  # (batch_size, seq_len, num_classes)

    @staticmethod
    def get_classes(logits: Tensor) -> Tensor:
        """ Get the classes with the highest probability """
        return torch.argmax(logits, dim=-1).squeeze()

    @Decorators.check_stack_kwargs
    def stack_forward(self: T5Stack, **kwargs):
        """
        Stacked forward function to replace the original forward function of T5EncoderModel

        :param input_ids: torch.Tensor of shape (batch_size, seq_length)
        :param relative_position: torch.Tensor of shape (batch_size, seq_length, seq_length
        :param sparsity_mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param use_additional_bucket: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param inputs_embeds: [Optional] torch.Tensor of shape (batch_size, seq_length, embed
        :param attention_mask: [Optional] torch.Tensor of shape (batch_size, seq_length)
        :param head_mask: [Optional] list of torch.Tensor of shape (num_heads, seq_length, seq_length)
        :param output_attentions: [Optional] bool
        :param output_hidden_states: [Optional] bool
        :param return_dict: [Optional] bool
        :return: BaseModelOutputWithPastAndCrossAttentions
        """
        logging.debug(
            f"Stacking forward method for T5EncoderModel with arguments: {list(kwargs.keys())}"
        )
        input_ids, inputs_embeds = kwargs.pop("input_ids", None), kwargs.pop("inputs_embeds", None)
        input_ids_passed = input_ids is not None
        inputs_embeds_passed = inputs_embeds is not None
        if input_ids_passed and inputs_embeds_passed:
            raise ValueError(
                f"You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids_passed:
            logging.debug("no inputs_embeds, creating them from input_ids")
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds_passed:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(f"You have to specify either input_ids or inputs_embeds")


        batch_size, seq_length = input_shape
        # TODO: seq_length might need to be masked after all
        extended_attention_mask = None
        if kwargs.pop("attention_mask", None):
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)
            # [GLM] We can provide a self-attention mask of dimensions
            # [batch_size, from_seq_length, to_seq_length]
            # ourselves, in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        hidden_states = self.dropout(inputs_embeds)
        head_mask = self.get_head_mask(kwargs.pop("head_mask", None), self.config.num_layers)

        args_to_forward = ("relative_position", "sparsity_mask", "use_additional_bucket")
        passed_args = set(kwargs.keys())
        assert set(args_to_forward) == passed_args, \
            f"Obligatory keys were not passed: {set(args_to_forward).intersection(passed_args)}"

        for i, block in enumerate(self.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # layer_outputs is a tuple with the following fields:
            # hidden-states, (self-attn position bias), (self-attn weights)
            # key-value-states are removed (used for cache, not needed here)
            layer_outputs = block.forward(
                hidden_states=hidden_states,
                mask=extended_attention_mask, # attention_mask renamed to mask
                position_bias=None,
                # encoder_decoder_position_bias=None,
                layer_head_mask=head_mask[i],
                # past_key_value=None,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # [T5] Add last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return_dict = bool(kwargs.pop("return_dict", self.config.use_return_dict))
        if not return_dict:
            variables = [
                hidden_states,
                # no present_key_value_states
                all_hidden_states,
                all_attentions,
                # no all_cross_attentions
            ]
            return tuple(v for v in variables if v)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @Decorators.check_block_attn_kwargs
    def block_forward(self: T5Block, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """
        Forward function for a single T5 block

        :param hidden_states: torch.Tensor of shape (batch_size, seq_length, dim)
        :param mask: torch.Tensor of shape (batch_size, seq_length, seq_length) - attention_mask
        :param position_bias: torch.Tensor of shape (batch_size, n_heads, seq_length, seq_length)
        :param layer_head_mask: torch.Tensor of shape (n_heads, seq_length, seq_length)
        :param output_attentions: bool
        :param relative_position: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param sparsity_mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param use_additional_bucket: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :return: tuple of (hidden_states, position_bias, attn_weights)
        """
        # (hidden_states, position_bias, opt: attn_weights)
        self_attn_outputs = self.self_attn(**kwargs)
        # clamping twice allows for more stable training with fp16
        hidden_states = self_attn_outputs[0]
        hidden_states = clamp_inf_values(hidden_states)

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = clamp_inf_values(hidden_states)

        # [T5] keep self-attention outputs and relative position weights
        outputs = (hidden_states, *self_attn_outputs[1:])
        return outputs  # (hidden-states, ([self-attention] position_bias), ([self-attention: opt] attn_weights)

    @Decorators.check_block_attn_kwargs
    def self_attn_forward(self: T5LayerSelfAttention, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """
        Forward function for T5LayerSelfAttention layer to replace the original forward function

        :param hidden_states: torch.Tensor of shape (batch_size, seq_length, dim)
        :param mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param position_bias: torch.Tensor of shape (batch_size, n_heads, seq_length, seq_length)
        :param layer_head_mask: torch.Tensor of shape (n_heads, seq_length, seq_length)
        :param output_attentions: bool
        :param relative_position: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param sparsity_mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param use_additional_bucket: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :return: tuple of (hidden_states, position_bias, attn_weights)
        """
        hidden_states = kwargs.pop("hidden_states")
        normed_hidden_states = self.layer_norm(hidden_states)
        # (attn_output, position_bias, opt: attn_weights)
        graph_attn_output = self.SelfAttention(normed_hidden_states, **kwargs)
        hidden_states += self.dropout(graph_attn_output[0])
        return hidden_states, graph_attn_output[1:] # (hidden_states, position_bias, opt: attn_weights)


if __name__ == "__main__":
    from transformers import T5Config

    config = T5Config.from_pretrained("t5-small")
    config.num_classes = 5
    config.model_name_size = "t5-small"
    config.model_max_length = 512
    config.rel_attn_num_additional_buckets = 2  # number of additional buckets for graph-to-graph attention
    config.use_cache = False

    logging.basicConfig(
        level=logging.INFO,                    # or DEBUG for more detail
        format="%(asctime)s [%(levelname)s]: %(message)s"
    )

    model = Graph2GraphRelationClassifier(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    batch_size = 2
    seq_length = 10
    # this script is not in the root folder, so paths should be relative
    data_path = "../data/rebel_dataset/REBEL_AMR_TRIPLES.train.hdf5"
    data = load_hdf5_data(data_path, split="train", num_samples=10)
    outputs = model(data, use_amr=False)
    print("outputs", outputs.shape)  # should be (batch_size, seq_length, num_classes)