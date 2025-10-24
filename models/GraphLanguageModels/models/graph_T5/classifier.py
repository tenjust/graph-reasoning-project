import torch
import torch.nn as nn
import logging

from transformers.modeling_utils import PreTrainedModel
from graph_t5 import T5Config, T5EncoderModel
from graph_t5 import T5TokenizerFast as T5Tokenizer


class GraphT5Classifier(PreTrainedModel):
    config_class = T5Config

    def __init__(
        self,
        config: T5Config,
    ):
        super().__init__(config=config)
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.modelsize, model_max_length=self.config.model_max_length)
    
        self.t5model = T5EncoderModel.from_pretrained(self.config.modelsize, config=config, ignore_mismatched_sizes=True)  # when intializing the model with .from_pretrained, the weights are loaded from the pretrained model, so the t5 parameters are not actually used in that case. Loading them here is unnecessary overhead.
        self.hidden_size = self.t5model.config.d_model
        self.classification_head = nn.Linear(self.hidden_size, self.config.num_classes, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_config(num_classes:int, modelsize:str="t5-base", num_additional_buckets:int=0, model_max_length:int=512) -> T5Config:
        config = T5Config.from_pretrained(modelsize)
        config.num_classes = int(num_classes)
        config.modelsize = str(modelsize)
        config.relative_attention_num_additional_buckets = int(num_additional_buckets)
        config.model_max_length = int(model_max_length)
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        relative_position: torch.Tensor,
        sparsity_mask: torch.Tensor,
        use_additional_bucket: torch.Tensor,
    ) -> torch.Tensor:
        logging.debug('t5 encoder model')
        output = self.t5model(input_ids=input_ids, relative_position=relative_position, sparsity_mask=sparsity_mask, use_additional_bucket=use_additional_bucket)  # (batch_size, seq_len, hidden_size)
        logging.debug('classification head')
        logits = self.classification_head(output[0])  # (batch_size, seq_len, num_classes)

        return logits

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)  # (batch_size, seq_len, num_classes)
    
    def get_label(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

class DualGraphT5Classifier(PreTrainedModel):
    """
    Same as GraphT5Classifier, but with two classification heads
    """
    config_class = T5Config

    def __init__(
        self,
        config: T5Config,
    ):
        super().__init__(config=config)
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.modelsize, model_max_length=self.config.model_max_length)
    
        self.t5model = T5EncoderModel.from_pretrained(self.config.modelsize, config=config, ignore_mismatched_sizes=True)
        self.hidden_size = self.t5model.config.d_model
        self.classification_head1 = nn.Linear(self.hidden_size, self.config.num_classes1, bias=True)
        self.classification_head2 = nn.Linear(self.hidden_size, self.config.num_classes2, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_config(num_classes1:int, num_classes2:int, modelsize:str="t5-base", num_additional_buckets:int=0, model_max_length:int=512) -> T5Config:
        config = T5Config.from_pretrained(modelsize)
        config.num_classes1 = int(num_classes1)
        config.num_classes2 = int(num_classes2)
        config.modelsize = str(modelsize)
        config.relative_attention_num_additional_buckets = int(num_additional_buckets)
        config.model_max_length = int(model_max_length)
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        relative_position: torch.Tensor,
        sparsity_mask: torch.Tensor,
        use_additional_bucket: torch.Tensor,
    ) -> torch.Tensor:
        logging.debug('t5 encoder model')
        output = self.t5model(input_ids=input_ids, relative_position=relative_position, sparsity_mask=sparsity_mask, use_additional_bucket=use_additional_bucket)  # (batch_size, seq_len, hidden_size)
        logging.debug('classification head 1')
        logits1 = self.classification_head1(output[0])  # (batch_size, seq_len, num_classes1)
        logging.debug('classification head 2')
        logits2 = self.classification_head2(output[0])  # (batch_size, seq_len, num_classes2)

        return logits1, logits2

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)  # (batch_size, seq_len, num_classes)
    
    def get_label(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


if __name__ == "__main__":
    config = T5Config.from_pretrained("t5-small")
    config.num_classes = 10
    config.modelsize = "t5-small"
    config.model_max_length = 512
    config.rel_attn_num_additional_buckets = 2  # number of additional buckets for graph-to-graph attention
    # init_additional_buckets_from=[None, 1]
    model = GraphT5Classifier(config)
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
    print(outputs.shape)
