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
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        # Relative attention buckets are increased by the number of additional buckets
        rel_attn_num_additional_buckets = config.to_dict().get('rel_attn_num_additional_buckets', 0)
        config.relative_attention_num_buckets += rel_attn_num_additional_buckets

        super().__init__(config, has_relative_attention_bias)
        self.rel_attn_num_additional_buckets = rel_attn_num_additional_buckets
        self.model_name_size = config.model_name_size

        # removed the possibility from Moritz's code for init_additional_buckets_from to be None,
        # as that would lead to random initialization of the additional buckets, while we want to
        # implement the version with infinity
        self.additional_buckets_init_dist = torch.inf
        if self.has_relative_attention_bias:
            self.init_relative_position_bias(same_model=True)

    def _relative_position_bucket(
            self,
            relative_position: Tensor,
            bidirectional: bool = False,
            num_buckets: int = 32,
            max_distance: int = 128,
            additional_bucket_id: int | Tensor = None,
    ) -> Tensor:
        """ Method overridden to support additional buckets for global graph-to-graph relative position """
        # Copied from transformers.models.t5.modeling_t5.T5Attention with slight modifications
        if not self.relative_attention_num_buckets:
            # num_buckets: Tensor = 32,
            raise ValueError(f"relative_attention_num_buckets must be > 0, not {self.relative_attention_num_buckets}")

        if not self.relative_attention_max_distance:
            # max_distance: Tensor = 128,
            raise ValueError(f"relative_attention_max_distance must be > 0, not {self.relative_attention_max_distance}")

        # relative_position shape: (query_length, key_length)
        relative_buckets = T5Attention._relative_position_bucket(
            relative_position,
            bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # use additional bucket id to shift the relative buckets,
        # only for positions corresponding to True in use_additional_bucket_id
        if additional_bucket_id is not None:
            relative_buckets[additional_bucket_id] += self.relative_attention_num_buckets
        return relative_buckets

    def init_relative_position_bias(self, same_model: bool = False):
        """
        Initializes parameters for relative position bias. This is necessary if additional buckets are used, as then the weights are not initialized automatically when calling `from_pretrained`.
        :param same_model: whether the model from which the relative position bias should be inherited is the same as the current model. If this is True, then the relative position bias is initialized from the current model. If this is False, then the relative position bias is initialized from a pretrained T5 model specified by `self.model_name_size`. Note that this only works if `self.is_decoder` is False, i.e. for encoder-only models.
        """
        if self.is_decoder:
            raise NotImplementedError("Decoder is not implemented.")

        # logging.debug('Loading model from which relative position bias should be inherited..')
        logging.debug('Getting relative position bias from parent model..')
        # TODO: Why wasn't in called from the original model in the first place?
        # TODO: is the number of relative position buckets changed?
        if same_model:
            loaded_model_bias = self.relative_attention_bias.weight
        else:
            raise NotImplementedError("Loading from different model is not necessary.")
            # parent_model = T5EncoderModel.from_pretrained(self.model_name_size)
            # parent_bias = parent_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
        # parent_bias shape: (num_buckets, num_heads)
        # TODO: remove if the same model will be used
        loaded_model_num_buckets, loaded_model_num_heads = loaded_model_bias.shape
        # rel_attn_buckets_num, rel_attn_heads_num = self.relative_attention_bias.weight.shape
        # assert parent_num_buckets == rel_attn_heads_num, f"{parent_num_buckets} should be {rel_attn_heads_num}"
        # assert parent_num_heads <= rel_attn_buckets_num, f"{parent_num_heads} should be <= {rel_attn_buckets_num}"

        logging.debug('init normal buckets')
        with torch.no_grad():
            # TODO: if the same model, it can be removed
            # self.relative_attention_bias.weight[:parent_num_heads, :] = parent_bias
            logging.debug('get parent buckets for additional buckets')
            if not self.rel_attn_num_additional_buckets:
                return
            additional_init_distances = Tensor(
                [self.additional_buckets_init_dist] * self.rel_attn_num_additional_buckets
            )
            init_additional_buckets_from = self._relative_position_bucket(
                relative_position=additional_init_distances,
                bidirectional=False,
                additional_bucket_id=None
            )
            logging.debug('initializing relative position bias..')
            additional_bucket_start_inx = loaded_model_num_buckets - self.rel_attn_num_additional_buckets
            for bucket_inx, init_inx in enumerate(init_additional_buckets_from, additional_bucket_start_inx):
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
        """Compute binned relative position bias"""
        if query_length != key_length:
            raise ValueError("The length of query and key are different!") # => then it makes sense to have to vars
        if device is None:
            device = self.relative_attention_bias.weight.device
        # >>> different in GLM implementation
        # because relative_position stays the same during computation
        if relative_position is not None:
            # context_position = relative_position[:, None].to(device) # - from original code?
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position  # shape (query_length, key_length)
        else:
            raise ValueError("Relative position is None!") # in GLM, relative_position is always passed?
            # was commented out in GLM implementation
            # relative_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # <<< different in GLM implementation
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
            additional_bucket_id=additional_bucket_id,
        )
        print("self.relative_attention_num_buckets", self.relative_attention_num_buckets) # 34
        print("self.n_heads", self.n_heads) # 8
        print("self.relative_attention_bias", self.relative_attention_bias) # Embedding(34, 8)
        print("relative_position_bucket", relative_position_bucket.shape) # torch.Size([10, 10])
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # shape (1, num_heads, query_length, key_length)
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
                    position_bias = self.compute_bias(
                        seq_length, seq_length, device=scores.device, additional_bucket_id=None)
                    # position_bias = torch.cat(tuple(position_bias for _ in range(batch_size)), dim=0)
                else:
                    # use_additional_bucket, relative_position: (batch_size, seq_length, key_length), seq_length = key_length
                    position_bias = []
                    for position, bucket_ids in zip(relative_position, use_additional_bucket):
                        # bucket_ids: (seq_length, key_length): position
                        position_bias += self.compute_bias(
                            *bucket_ids.shape,
                            device=scores.device,
                            relative_position=position,
                            additional_bucket_id=bucket_ids
                        )
                    position_bias = torch.cat(position_bias, dim=0)

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

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout.item(), training=self.training)

        # Mask heads if we want to
        if layer_head_mask:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        print("outputs of GraphAttention", len(outputs))
        return outputs


class Decorators:
    @staticmethod
    def check_stack_kwargs(stack_forward):
        """ Wrapper to call the T5Stack forward method with the correct arguments """
        def inner_fn(self, **kwargs):
            # Optional arguments:
            # "head_mask", "attention_mask", "inputs_embeds", "output_attentions", "output_hidden_states", "return_dict"
            # TODO: rename use_additional_bucket
            obligatory_args = {"input_ids", "relative_position", "sparsity_mask", "use_additional_bucket"}
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


class Graph2GraphRelationClassifier(PreTrainedModel):
    """
    This class implements a KG relation classifier that is based on T5 model architecture
    and guided by AMR-based graphs. This implementation follows the idea of Graph Language
    Models (GLM) project while incorporating the original code of T5 whenever possible.
    The configuration is simplified to gGLM and focuses on the relation classification task.
    """
    def __init__(self, config: T5Config):
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
        # self.init_additional_buckets_from = init_additional_buckets_from

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

        # Monkey patching forward methods
        # self.t5.forward = self.simple_forward.__get__(self.t5)
        self.t5.encoder.forward = Graph2GraphRelationClassifier.stack_forward.__get__(self.t5.encoder)
        # T5Block.forward = Graph2GraphRelationClassifier.block_forward

        # self_attn and feed_forward are added to the T5Block outside the ModuleList
        # for easier access during forward pass and readability
        for layer, block in enumerate(self.t5.encoder.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))
            block.forward = self.block_forward.__get__(block)
            # remove the original packaging for self-attention and feed-forward layers
            block.layer = nn.Identity()
            # create a separate self-attention layer
            first_layer = True if layer == 0 else False
            block.self_attn = T5LayerSelfAttention(
                # TODO: has_relative_attention_bias can be omitted here?
                self.config, has_relative_attention_bias=first_layer
            )
            block.self_attn.forward = self.self_attn_forward.__get__(block.self_attn)
            # create a new feed forward layer
            block.feed_forward = T5LayerFF(self.config)
            block.self_attn.SelfAttention = GraphAttention(
                self.config, has_relative_attention_bias=first_layer
            )

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
        output = self.t5.encoder(**kwargs)  # (batch_size, seq_len, hidden_size)
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

    # @staticmethod
    # def simple_forward(encoder: T5EncoderModel, **kwargs):
    #     """ Simple forward function to replace the original forward function of T5EncoderModel """
    #     logging.debug(f"Monkey parch forward method for T5EncoderModel with arguments: {kwargs}")
    #     return_dict = bool(kwargs.pop("return_dict", None)) # TODO: remove?
    #     print("encoder", encoder)
    #     print("encoder.forward", encoder.forward)
    #     print("encoder.super", encoder.super)
    #     return encoder.__super__.forward(**kwargs, return_dict=return_dict)

    @Decorators.check_stack_kwargs
    def stack_forward(self: T5Stack, **kwargs):
        """
        Stacked forward function to replace the original forward function of T5EncoderModel

        :meta-param encoder: T5Stack object this method is bound to
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
        logging.debug(f"Stacking forward method for T5EncoderModel with arguments: {kwargs}")
        input_ids, inputs_embeds = kwargs.pop("input_ids", None), kwargs.pop("inputs_embeds", None)
        input_ids_passed = input_ids is not None
        inputs_embeds_passed = inputs_embeds is not None
        if input_ids_passed and inputs_embeds_passed:
            raise ValueError(
                f"You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids_passed:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            logging.debug("no inputs_embeds, creating them from input_ids")
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
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            #  ourselves, in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        hidden_states = self.dropout(inputs_embeds)
        head_mask = self.get_head_mask(kwargs.pop("head_mask", None), self.config.num_layers)

        args_to_forward = ("relative_position", "sparsity_mask", "use_additional_bucket")
        assert set(args_to_forward) == set(kwargs.keys()), \
            f"Obligatory keys were lost: {set(args_to_forward).intersection(kwargs.keys())}"

        for i, block in enumerate(self.block):
            assert type(block) == T5Block, "Expected T5Block, got {}".format(type(block))
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

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

            # layer_outputs is a tuple with the following fields:
            # hidden-states, (key-value-states), (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            # key-value-states are always set to None (used for cache, not needed here)
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            if output_attentions:
                all_attentions += (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return_dict = bool(kwargs.pop("return_dict", self.config.use_return_dict))
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

    @Decorators.check_block_attn_kwargs
    def block_forward(self: T5Block, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """
        Forward function for a single T5 block

        :meta-param block: T5Block object this method is bound to
        :param hidden_states: torch.Tensor of shape (batch_size, seq_length, dim)
        :param mask: torch.Tensor of shape (batch_size, seq_length, seq_length) - attention_mask
        :param position_bias: torch.Tensor of shape (batch_size, n_heads, seq_length, seq_length)
        :param layer_head_mask: torch.Tensor of shape (n_heads, seq_length, seq_length)
        :param output_attentions: bool
        :param relative_position: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param sparsity_mask: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :param use_additional_bucket: torch.Tensor of shape (batch_size, seq_length, seq_length)
        :return: tuple of (hidden_states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights))
        """
        self_attn_outputs = self.self_attn(**kwargs) # TODO: double-check the attention outputs
        # clamping twice allows for more stable training with fp16
        hidden_states = self_attn_outputs[0]
        hidden_states = clamp_inf_values(hidden_states)

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = clamp_inf_values(hidden_states)

        # Keep self-attention outputs and relative position weights
        outputs = (hidden_states,) + self_attn_outputs[2:]
        print("outputs of T5Block forward", len(outputs))
        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

    @Decorators.check_block_attn_kwargs
    def self_attn_forward(self: T5LayerSelfAttention, **kwargs) -> tuple[Tensor, list[Tensor]]:
        """
        Forward function for T5LayerSelfAttention layer to replace the original forward function

        :meta-param self_attn: T5LayerSelfAttention object this method is bound to
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
        attention_output = self.SelfAttention(normed_hidden_states, **kwargs)
        hidden_states += self.dropout(attention_output[0])
        print("attention_output of SelfAttention forward", len(attention_output))
        return hidden_states, attention_output[1:]


if __name__ == "__main__":
    from transformers import T5Config

    config = T5Config.from_pretrained("t5-small")
    config.num_classes = 10
    config.model_name_size = "t5-small"
    config.model_max_length = 512
    config.rel_attn_num_additional_buckets = 2  # number of additional buckets for graph-to-graph attention
    config.use_cache = False

    model = Graph2GraphRelationClassifier(config)
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
        use_additional_bucket=use_additional_bucket,
    )
    print(outputs.shape)  # should be (batch_size, seq_length, num_classes)