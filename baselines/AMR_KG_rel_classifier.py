from torch import nn
from torch import Tensor
import torch
import logging
from transformers.modeling_utils import PreTrainedModel
# transformers.src
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class Graph2GraphRelationClassifier(PreTrainedModel):
    """
    This class implements a KG relation classifier that is based on T5 model architecture
    and guided by AMR-based graphs. This implementation follows the idea of Graph Language
    Models (GLM) project while incorporating the original code of T5 whenever possible.
    The configuration is simplified to gGLM and focuses on the relation classification task.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_size: str = self.config.model_size
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.model_size, model_max_length=self.config.model_max_length
        )

        self.t5 = T5EncoderModel.from_pretrained(
            self.model_size, config=self.config, ignore_mismatched_sizes=True)
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

    @staticmethod
    def simple_forward(encoder, **kwargs):
        """ Simple forward function to replace the original forward function of T5EncoderModel """
        logging.debug(f"Monkey parch forward method for T5EncoderModel with arguments: {kwargs}")
        return_dict = bool(kwargs.pop("return_dict", None)) # TODO: remove?
        return encoder(return_dict=return_dict, **kwargs)

    @staticmethod
    def stack_forward(encoder, **kwargs):
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

    def block_forward(self, **kwargs):
        """ Forward function for a single T5 block """
        args = [
            "hidden_states", "attention_mask", "position_bias", "layer_head_mask",
            "output_attentions", "relative_position", "sparsity_mask", "use_additional_bucket",
        ]
        assert set(kwargs.keys()) == set(args), f"Expected keys: {args}, but got {kwargs.keys()}"
        # TODO: good chance to use self-defined attention
        self_attention_outputs = T5LayerSelfAttention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,  # new GLM
            output_attentions=output_attentions,
            relative_position=relative_position,  # cache position
            sparsity_mask=sparsity_mask,  # new GLM
            use_additional_bucket=use_additional_bucket,  # new GLM
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = T5LayerFF(hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, attention_outputs)

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

    def forward(self, **kwargs: Tensor) -> Tensor:
        """ Forward method of the classifier """
        logging.debug('T5 encoder model')
        assert {"input_ids", "relative_position", "sparsity_mask", "use_additional_bucket"} == set(kwargs.keys()), \
        (f"Expected keys: ['input_ids', 'relative_position', 'sparsity_mask', 'use_additional_bucket'], "
         f"but got {kwargs.keys()}")
        # outputs: (last_hidden_state, hidden_states, attentions)
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
