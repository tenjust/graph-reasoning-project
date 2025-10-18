from __future__ import annotations

import warnings

from torch import nn
from torch import Tensor
import torch
import logging

from transformers import T5Config
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Block, T5LayerFF, T5LayerSelfAttention, T5Stack, \
    T5Attention
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def clamp_inf_values(states: Tensor) -> Tensor:
    """ Clamp inf values to enable fp16 training - fully from original T5 implementation """
    if states.dtype == torch.float16 and torch.isinf(states).any():
        clamp_value = torch.finfo(states.dtype).max - 1000
        return torch.clamp(states, min=-clamp_value, max=clamp_value)
    return states


class GraphAttention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, init_additional_buckets_from: int | list[int] = None):
        # Relative attention buckets are increased by the number of additional buckets
        config.relative_attention_num_buckets += config.rel_attn_num_additional_buckets
        super().__init__(config, has_relative_attention_bias)

        self.model_name_size = config.model_name_size

        if self.has_relative_attention_bias and init_additional_buckets_from:
            self.init_relative_position_bias(init_additional_buckets_from, True)
        elif self.has_relative_attention_bias and not init_additional_buckets_from:
            raise ValueError(f"No init_additional_buckets_from in GraphAttention")

    def _relative_position_bucket(
            self,
            relative_position: Tensor,
            bidirectional: bool = False,
            num_buckets: int = 32,
            max_distance: int = 128,
            additional_bucket_id: int = None,
    ) -> Tensor:
        """ Method overridden to support additional buckets for global graph-to-graph relative position """
        # Copied from transformers.models.t5.modeling_t5.T5Attention with slight modifications
        if not self.relative_attention_num_buckets:
            # num_buckets: Tensor = 32,
            raise ValueError(f"relative_attention_num_buckets must be > 0, not {self.relative_attention_num_buckets}")

        if not self.relative_attention_max_distance:
            # max_distance: Tensor = 128,
            raise ValueError(f"relative_attention_max_distance must be > 0, not {self.relative_attention_max_distance}")

        relative_buckets = T5Attention._relative_position_bucket(
            relative_position,
            bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        if additional_bucket_id:
            assert type(additional_bucket_id) == int, f"type of additional_bucket_id must (?) be int, but got {type(additional_bucket_id)}"
            relative_buckets[additional_bucket_id] += self.relative_attention_num_buckets
        return relative_buckets

    def init_relative_position_bias(self, init_additional_buckets_from: int | list[int] = None, same_model: bool = False):
        """
        Initializes parameters for relative position bias. This is necessary if additional buckets are used, as then the weights are not initialized automatically when calling `from_pretrained`.
        :param init_additional_buckets_from: the indices of the buckets from which the additional buckets should be initialized. If this is an int, then all additional buckets are initialized from the same bucket. If this is a list, then the list should have the same length as the number of additional buckets, and the i-th entry of the list determines from which bucket the i-th additional bucket is initialized. Setting this to None (or an element in the list to None) means that the additional bucket is not initialized, i.e. it is left unchanged.
        :param same_model: whether the model from which the relative position bias should be inherited is the same as the current model. If this is True, then the relative position bias is initialized from the current model. If this is False, then the relative position bias is initialized from a pretrained T5 model specified by `self.model_name_size`. Note that this only works if `self.is_decoder` is False, i.e. for encoder-only models.
        """
        if self.is_decoder:
            raise NotImplementedError("Decoder is not implemented.")

        # logging.debug('Loading model from which relative position bias should be inherited..')
        logging.debug('Getting relative position bias from parent model..')
        # TODO: Why wasn't in called from the original model in the first place?
        # TODO: is the number of relative position buckets changed?
        if same_model:
            parent_bias = self.relative_attention_bias.weight
        else:
            parent_model = T5EncoderModel.from_pretrained(self.model_name_size)
            parent_bias = parent_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
        # parent_bias share: (num_buckets, num_heads) # TODO: double check if these are the dimensions
        # TODO: remove if the same model will be used
        parent_num_buckets, parent_num_heads = parent_bias.shape
        rel_attn_buckets_num, rel_attn_heads_num = self.relative_attention_bias.weight.shape
        assert parent_num_buckets == rel_attn_heads_num, f"{parent_num_buckets} should be {rel_attn_heads_num}"
        assert parent_num_heads <= rel_attn_buckets_num, f"{parent_num_heads} should be <= {rel_attn_buckets_num}"

        logging.debug('init normal buckets')
        # TODO: if the same model, it can be removed
        with torch.no_grad():
            self.relative_attention_bias.weight[:parent_num_heads, :] = parent_bias

        logging.debug('get parent buckets for additional buckets')
        if not init_additional_buckets_from:
            return
        # num_additional_buckets = 0, if the same model is used
        num_additional_buckets = rel_attn_buckets_num - parent_num_heads
        if not num_additional_buckets:
            return
        raise NotImplementedError("num_additional_buckets > 0, so additional buckets need to be initialized!")
        if not isinstance(init_additional_buckets_from, list):
            init_additional_buckets_from = [init_additional_buckets_from] * num_additional_buckets
        assert len(init_additional_buckets_from) == num_additional_buckets, \
            f"{len(init_additional_buckets_from)} should be {num_additional_buckets}"

        skip_bucket = [idx is None for idx in init_additional_buckets_from]
        init_additional_buckets_from = [0 if idx is None else idx for idx in init_additional_buckets_from]
        init_additional_buckets_from = torch.tensor(init_additional_buckets_from, dtype=torch.long)
        init_additional_buckets_from = self._relative_position_bucket(
            relative_position=init_additional_buckets_from,
            bidirectional=False,
            additional_bucket_id=None
        )
        logging.debug('initializing relative position bias..')
        with torch.no_grad():
            for bucket, (skip, init_inx) in enumerate(zip(skip_bucket, init_additional_buckets_from), num_buckets):
                if not skip:
                   self.relative_attention_bias.weight[bucket, :] = parent_bias[init_inx, :]

    def compute_bias(
            self,
            query_length: int,
            key_length: int,
            device=None,
            relative_position: Tensor = None,
            additional_bucket_id: int = None,
    ) -> Tensor:
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        # >>> different in GLM implementation
        # because relative_position stays the same during computation
        if relative_position:
            context_position = relative_position[:, None].to(device)
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position  # shape (query_length, key_length)
        else:
            # was commented out in GLM implementation
            relative_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # <<< different in GLM implementation
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            max_distance=self.relative_attention_max_distance,
            additional_bucket_id=additional_bucket_id,
        )
        values = self.relative_attention_bias(
            relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            hidden_states,
            mask=None,
            position_bias=None,
            query_length=None,
            output_attentions=False,
            use_cache=False,  # None in GLM
            layer_head_mask=None,  # new GLM
            relative_position=None,  # new GLM
            sparsity_mask=None,  # new GLM
            use_additional_bucket=None,  # new GLM
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        GLM new, differs from the original. Due to decoder not being used, only self-attention is calculated (right?).

        :param relative_position: [MP] relative position for the attention. If `None`, it will be computed as in a standard sequence-to-sequence model. If not `None`, it will be used as the relative position for the attention. It is a tensor of shape [batch_size, query_length, key_length].
        :param sparsity_mask: [MP] sparsity mask for the attention. If `None`, it will be computed as in a standard sequence-to-sequence model. If not `None`, it will be used as the sparsity mask for the attention. It is a tensor of shape [batch_size, query_length, key_length]. A value of 1 means that the corresponding attention weight is not masked, and a value of 0 means that the corresponding attention weight is masked. Hence, the sparsity mask is a binary mask that (kind of) can be used like a multiplicative mask.
        :param use_additional_bucket: [MP] whether to use additional buckets for the attention. If `None`, only standard positional encodings will be used. If not `None`, additional buckets will be used for the relative position. It is a tensor of shape [batch_size, query_length, key_length]. A value of False means that the corresponding position is a standard relative position, and a value of True means that the corresponding additional bucket should be used.
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def project(hidden_states, proj_layer):
            """projects hidden states correctly to key/query states"""
            # self-attn (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
            # self-attn
            # (batch_size, n_heads, key_length, dim_per_head)
            # hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            return hidden_states

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(hidden_states, self.k)
        value_states = project(hidden_states, self.v)

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None: # always true in GLM, because position_bias is hardcoded to None in T5Block.forward
            if not self.has_relative_attention_bias:
                # new GLM? initialize to zeros only for layer 0
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
                )
                # self.gradient_checkpointing is hardcoded to False in GLM?
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                # new GLM stuff: relative_position is created in get_batch (train_LM.py) and passed through all the way to here
                if relative_position is None:
                    assert use_additional_bucket is None
                    position_bias = self.compute_bias(seq_length, seq_length, device=scores.device,
                                                      additional_bucket_id=None)
                    # position_bias = torch.cat(tuple(position_bias for _ in range(batch_size)), dim=0)
                else:
                    position_bias = torch.cat(tuple(
                        self.compute_bias(seq_length, seq_length, device=scores.device, relative_position=position,
                                          additional_bucket_id=bucket_id) for position, bucket_id in
                        zip(relative_position, use_additional_bucket)), dim=0)

            logging.debug(f"position_bias = {position_bias.shape if position_bias is not None else position_bias}")
            logging.debug(f"mask = {mask.shape if mask is not None else mask}")
            if mask:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if sparsity_mask:
            assert sparsity_mask.dtype == torch.bool, f"{relative_position.dtype} should be torch.bool"

            sparsity_mask = ~ sparsity_mask.unsqueeze(
                1)  # add extra dimension for heads and negate for indexing the masked positions
            sparsity_mask = sparsity_mask.expand_as(position_bias)

        scores += position_bias

        # >>> masked softmax >>>
        if sparsity_mask: # sparsity_mask is initialized in _get_graphT5_relativeposition_sparsitymask (wrapper_functions.py)
            # this works in the backward pass, because potential nan values that the softmax produces
            # in the forward pass are not used in backpropagation, because the "=" is independent of
            # the value that the entry had previously. This is not the case for "+=", which is why we
            # need to set the values to -inf instead of adding -inf.
            scores[sparsity_mask] = float('-inf')

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

        # replace nan values in the attention weights with 0. nan happens if all positions are masked for one token, as then all inputs to softmax are -inf for that token
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # <<< masked softmax <<<

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class Graph2GraphRelationClassifier(PreTrainedModel):
    """
    This class implements a KG relation classifier that is based on T5 model architecture
    and guided by AMR-based graphs. This implementation follows the idea of Graph Language
    Models (GLM) project while incorporating the original code of T5 whenever possible.
    The configuration is simplified to gGLM and focuses on the relation classification task.
    """
    def __init__(self, config, init_additional_buckets_from: list[int] = None):
        if config.is_decoder:
            warnings.warn("Decoder is not implemented, setting is_decoder to False.")
        config.is_decoder = False
        if config.use_cache:
            warnings.warn("use_cache is not needed for encoder-only model, setting use_cache to False.")
        config.use_cache = False
        super().__init__(config, layer_idx=None)

        var = "rel_attn_num_additional_buckets"
        assert hasattr(config, var), \
            f"For running Graph Language Model's method T5Config must have attribute '{var}'"

        # main variables of the method - TODO: get from config and feed directly into Attention?
        # relative_position:
        # sparsity_mask:
        # use_additional_bucket:
        # TODO: is the same question gets passed around?
        # TODO: to rename init_additional_buckets_from
        self.init_additional_buckets_from = init_additional_buckets_from

        self.model_name_size: str = self.config.model_name_size
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.model_name_size, model_max_length=self.config.model_max_length
        )

        self.t5 = T5EncoderModel.from_pretrained(
            self.model_name_size, config=self.config, ignore_mismatched_sizes=True)
        self.configure_t5()

        self.classifier = nn.Linear(
            self.t5.shared.embedding_dim, self.config.num_classes, bias=True
        )
        self.softmax = nn.Softmax(dim=-1)

    def configure_t5(self):
        """ Configure the T5 model to fit our needs """
        self.t5.model_parallel = False
        self.t5.device_map = None
        self.t5.forward = self.simple_forward.__get__(self.t5)
        # TODO: add a toggle for Self-Attention
        #  if it's the first layer, run init_relative_position_bias in init
        self.t5.encoder.forward = self.stack_forward.__get__(self.t5.encoder)
        # self_attn, feed_forward are added to the T5Block outside the ModuleList (the ones in the list are not changed)
        for layer, block in enumerate(self.t5.encoder.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))
            block.forward = self.block_forward.__get__(block)
            block.layer = None
            # create a new self-attention layer
            first_layer = True if layer == 0 else False
            block.self_attn = T5LayerSelfAttention(
                # has_relative_attention_bias can be omitted here?
                self.config, has_relative_attention_bias=first_layer
            )
            block.self_attn.SelfAttention = GraphAttention(
                self.config, has_relative_attention_bias=first_layer, init_additional_buckets_from=self.init_additional_buckets_from if first_layer else None
            )
            # create a new feed forward layer
            block.feed_forward = T5LayerFF(self.config)

        # if self.config.rel_attn_num_additional_buckets:
        #     self.t5.encoder.block[0].self_attn.SelfAttention.init_relative_position_bias(
        #         self.init_additional_buckets_from, # model=self.t5,
        #     )

    def forward(self, **kwargs: Tensor) -> Tensor:
        """ Forward method of the classifier """
        logging.debug('T5 encoder model')
        args = ["input_ids", "relative_position", "sparsity_mask", "use_additional_bucket"]
        assert set(args) == set(kwargs.keys()), f"Expected keys: {args}, but got {kwargs.keys()}"
        # outputs: (last_hidden_state, hidden_states, attentions)
        logging.debug('T5 encoder model is called')
        output = self.t5(**kwargs)  # (batch_size, seq_len, hidden_size)
        logging.debug('Classifier is called')
        logits = self.classifier(output[0])  # (batch_size, seq_len, num_classes)
        return logits

    def get_prob_distribution(self, logits: Tensor) -> Tensor:
        """ Get the probability distribution over classes """
        return self.softmax(logits)  # (batch_size, seq_len, num_classes)

    @staticmethod
    def get_class(logits: Tensor) -> int:
        """ Get the class with the highest probability """
        return int(torch.argmax(logits, dim=-1).item())

    @staticmethod
    def simple_forward(encoder: T5EncoderModel, **kwargs):
        """ Simple forward function to replace the original forward function of T5EncoderModel """
        logging.debug(f"Monkey parch forward method for T5EncoderModel with arguments: {kwargs}")
        return_dict = bool(kwargs.pop("return_dict", None)) # TODO: remove?
        return encoder(return_dict=return_dict, **kwargs)

    @staticmethod
    def stack_forward(encoder: T5Stack, **kwargs):
        """ Stacked forward function to replace the original forward function of T5EncoderModel """
        logging.debug(f"Stacking forward method for T5EncoderModel with arguments: {kwargs}")
        arg_keys = (
            "input_ids", "attention_mask", "inputs_embeds", "head_mask", "output_attentions",
            "output_hidden_states", "return_dict", "relative_position", "sparsity_mask", "use_additional_bucket" # TODO: rename use_additional_bucket
        )
        assert set(kwargs.keys()) == set(arg_keys), f"Expected keys: {arg_keys}, but got {kwargs.keys()}"
        input_ids, inputs_embeds = kwargs.pop("input_ids", None), kwargs.pop("inputs_embeds", None)
        if input_ids and inputs_embeds:
            raise ValueError(
                f"You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            logging.debug("no inputs_embeds, creating them from input_ids")
            inputs_embeds = encoder.embed_tokens(input_ids)
        elif inputs_embeds:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(f"You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        # TODO: seq_length might need to be masked after all
        extended_attention_mask = None
        if kwargs.pop("attention_mask", None):
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = encoder.get_extended_attention_mask(attention_mask, input_shape)

        output_attentions = kwargs.pop("output_attentions", encoder.config.output_attentions)
        output_hidden_states = kwargs.pop("output_hidden_states", encoder.config.output_hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        hidden_states = encoder.dropout(inputs_embeds)
        head_mask = encoder.get_head_mask(kwargs.pop("head_mask", None), encoder.config.num_layers)

        for i, block in enumerate(encoder.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=head_mask[i],
                past_key_value=None,
                output_attentions=output_attentions,
                relative_position=kwargs.pop("relative_position", None),
                sparsity_mask=kwargs.pop("sparsity_mask", None),
                use_additional_bucket=kwargs.pop("use_additional_bucket", None),
            )

            # layer_outputs is a tuple with:
            # hidden-states, (key-value-states), (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            # key-value-states are always set to None (used for cache, not needed here)
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            if output_attentions:
                all_attentions += (layer_outputs[3],)

        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return_dict = bool(kwargs.pop("return_dict", encoder.config.use_return_dict))
        if not return_dict:
            variables = [
                hidden_states,
                None, # present_key_value_states
                all_hidden_states,
                all_attentions,
                None, # all_cross_attentions
            ]
            return tuple(v for v in variables if v)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @staticmethod
    def block_forward(block: T5Block, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """ Forward function for a single T5 block """
        args = [
            "hidden_states", "attention_mask", "position_bias", "layer_head_mask", # new GLM
            "output_attentions", "relative_position", "sparsity_mask", # new GLM
            "use_additional_bucket", # new GLM
        ]
        assert set(kwargs.keys()) == set(args), f"Expected keys: {args}, but got {kwargs.keys()}"
        # TODO: good chance to use self-defined attention
        self_attention_outputs = block.self_attention(**kwargs) # TODO: double-check the attention outputs
        # clamping twice allows for more stable training with fp16
        hidden_states = self_attention_outputs[0]
        hidden_states = clamp_inf_values(hidden_states)

        hidden_states = block.feed_forward(hidden_states)
        hidden_states = clamp_inf_values(hidden_states)

        # Keep self-attention outputs and relative position weights
        return hidden_states, self_attention_outputs[2:]  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

    @staticmethod
    def self_attention_forward(attn_layer: T5LayerSelfAttention, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """ Forward function for self-attention layer """
        args = [
            "hidden_states", "attention_mask", "position_bias", "layer_head_mask",
            "output_attentions", "relative_position", "sparsity_mask", "use_additional_bucket"
        ]
        assert set(kwargs.keys()) == set(args), f"Expected keys: {args}, but got {kwargs.keys()}"
        hidden_states = kwargs.pop("hidden_states")
        normed_hidden_states = attn_layer.layer_norm(hidden_states)
        attention_output = attn_layer.SelfAttention(normed_hidden_states, **kwargs)
        hidden_states = hidden_states + attn_layer.dropout(attention_output[0])
        return hidden_states, attention_output[1:]


if __name__ == "__main__":
    from transformers import T5Config

    config = T5Config.from_pretrained("t5-small")
    config.num_classes = 10
    config.model_name_size = "t5-small"
    config.model_max_length = 512
    config.rel_attn_num_additional_buckets = 2  # number of additional buckets for graph-to-graph attention

    model = Graph2GraphRelationClassifier(config, init_additional_buckets_from=[None, 1])
    print(model)

    batch_size = 2
    seq_length = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    print("input_ids", input_ids)
    relative_position = torch.randint(-4, 4, (batch_size, seq_length, seq_length))
    print("relative_position", relative_position)
    sparsity_mask = torch.randint(0, 2, (batch_size, seq_length, seq_length)).bool()
    print("sparsity_mask", sparsity_mask)
    use_additional_bucket = torch.randint(0, 2, (batch_size, seq_length, seq_length)).bool()
    print("use_additional_bucket", use_additional_bucket)

    outputs = model(
        input_ids=input_ids,
        relative_position=relative_position,
        sparsity_mask=sparsity_mask,
        use_additional_bucket=use_additional_bucket
    )
    print(outputs.shape)  # should be (batch_size, seq_length, num_classes)