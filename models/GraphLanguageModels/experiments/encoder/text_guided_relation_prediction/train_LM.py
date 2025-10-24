from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import json 
from tqdm import tqdm
import torch
from torch import nn
from typing import List, Dict, Optional
import random
import numpy as np
import logging
import wandb

from baselines.GraphLanguageModels.experiments.encoder.text_guided_relation_prediction.data_utils import \
    chunker, get_batch, OpenData
from baselines.GraphLanguageModels.experiments.encoder.text_guided_relation_prediction.eval_utils import \
    get_preds_and_rank, get_metrics, get_accuracy
from baselines.GraphLanguageModels.experiments.encoder.text_guided_relation_prediction.model_utils import \
    reset_params, freeze_params
from baselines.GraphLanguageModels.experiments.encoder.text_guided_relation_prediction.str_utils import \
    str2optimizer, str2criterion, str2logging_level, str2bool, str2int, str2path
from baselines.GraphLanguageModels.models.graph_T5.classifier import GraphT5Classifier, DualGraphT5Classifier
from baselines.GraphLanguageModels.models.graph_T5.wrapper_functions import get_embedding

def add_args_shared(parser: ArgumentParser):
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        help="wandb mode. For example `disabled` to disable wandb, which can be useful for debugging.",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        default="t5-small",
        help="size of the model",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="number of epochs",
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=1,
        help='number of epochs without improvement before stopping training',
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str2optimizer,
        default="AdamW",
        help="optimizer",
    )
    parser.add_argument(
        "--criterion",
        type=str2criterion,
        default="CrossEntropyLoss",
        help="criterion, i.e. loss function",
    )
    parser.add_argument(
        "--logging_level",
        type=str2logging_level,
        default="INFO",
        help="logging level",
    )
    parser.add_argument(
        "--wandb_name_prefix",
        type=str,
        default="",
        help="prefix to run name in wandb",
    )

def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--params_to_train",
        type=str,
        default="all",
        help="which parameters to train. 'all' means all parameters. 'head' means only the parameters that are added on top of the pretrained model.",
    )
    parser.add_argument(
        "--graph_representation",
        type=str,
        default="lGLM",
        help="How the graph is represented. 'lGLM' means local graph language model. 'set' means that the graph is represented as a set of triplets (random order) and that the model is a sequence model. 'gGLM' means global GLM, i.e. the same as lGLM but the attention is not sparse and non-neighboring relations and concepts have a PE of the maximum distance. 'list' means that the graph is represented as a list of triplets (alphabetical oder) and that the model is a sequence model.",
    )
    parser.add_argument(
        "--reset_params",
        type=str2bool,
        default=False,
        help="whether to reset the parameters of the model before training. This removes pretrained weights.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps. Effective batch size is `train_batch_size * gradient_accumulation_steps`",
    )
    parser.add_argument(
        "--eos_usage",
        type=str,
        default="False",
        help="Only relevant when using GLM. eos stands for end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which perceives locality when using the local GLM"
    )
    parser.add_argument(
        "--num_evals_per_epoch",
        type=int,
        default=100,
        help="number of evaluation on dev and test set per epoch. Has to be at least 1. Metrics on the train set are computed for each sub-epoch independently, so they are computed on different subsets of the train set. Dev and test metrics are always evaluated on the entire dev and test set, respectively. This value does not impact the behavior of early stopping. However, it changes the impact of the random seed, so small variations in performance are to be expected when changing this parameter."
    )
    parser.add_argument(
        "--num_additional_buckets",
        type=int,
        default=None,
        help="number of additional buckets for relative position embedding. If None, then the default depending on the graph_representation is chosen."
    )
    parser.add_argument(
        "--init_additional_buckets_from",
        type=str2int,
        default=1e6,
        help="Specifies from which bucket of the parent model the additional buckets are initialized. init_additional_buckets_from gives the relative position, and the bucket is the one which corresponds to that relative position. If None, then the additional buckets are initialized randomly as determined by from_pretrained().",
    )
    parser.add_argument(
        "--use_text",
        type=str,
        default="FullyConnected",
        help="whether and how to use text as input. Can be `False` for not using text, or `FullyConnected` having a full attention matrix with T2G and G2T attention.",
    )
    parser.add_argument(
        "--use_graph",
        type=str2bool,
        default=True,
        help="Whether to use the graph at all. If False, then only the triplet with the masked relation is used.",
    )
    parser.add_argument(
        "--entailed_triplets_only",
        type=str2bool,
        default=False,
        help="Whether to use only entailed triplets. If True, then the model is trained on (i) entailed triplets and (ii) no-relation triplets only. If False, then the model is trained on all triplets. ",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="maximum sequence length. Sequences longer than this are truncated.",
    )
    parser.add_argument(
        "--run_eval",
        type=str2bool,
        default=False,
        help="whether to run evaluation during training.",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str2path,
        default=None,
        help="directory to save model to. If None, then the model is not saved. Otherwise, it is saved every subepoch (even if run_eval is False).",
    )
    parser.add_argument(
        "--save_at",
        type=str,
        default="log_seen_instances",
        help="when to save the model. Can be `epoch` for saving at the end of each subepoch, or `log_seen_instances` for saving after specific numbers of seen instances or `all` for both.",
    )
    parser.add_argument(
        "--stop_training_after_seen_instances",
        type=str2bool,
        default=True,
        help="whether to stop training after a specific number of seen instances. If True, then the training stops after `stop_training_after_seen_instances` seen instances. If False, then the training stops after `num_epochs` epochs or based on early_stopping.",
    )
    parser.add_argument(
        "--continue_training",
        type=str2bool,
        default=False,
        help="Whether to continue training from the model in save_model_dir. The model weights and training data are used as intended, but second order momentum etc from the optimizer is not initialized correctly.",
    )
    parser.add_argument(
        "--predict_source",
        type=str2bool,
        default=False,
        help="Whether to predict the source of the relation. If True, then the model is a DualGraphT5Classifier and the number of classes is the number of relations and the number of sources (3 for text, graph and no_relation). If False, then the model is a GraphT5Classifier and the number of classes is the number of relations.",
    )
    parser.add_argument(
        "--source_prediction_weight",
        type=float,
        default=0.1,
        help="Weight of the source prediction loss. Only relevant when predict_source is True.",
    )


def get_args(parser: ArgumentParser):
    args = parser.parse_args()

    if args.num_additional_buckets is None:
        args.num_additional_buckets = 0
        if args.graph_representation in ["set", "list","lGLM"]:
            args.num_additional_buckets += 0
        elif args.graph_representation in ["gGLM"]:
            args.num_additional_buckets += 1
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")
        if args.use_text in ['False']:
            args.num_additional_buckets += 0
        elif args.use_text in ['FullyConnected']:
            args.num_additional_buckets += 2
        else:
            raise ValueError(f"unknown use_text {args.use_text}")

    if args.eos_usage != 'False' and args.graph_representation not in ['lGLM', 'gGLM']:
        raise ValueError(f"eos_usage can only be used with lGLM or gGLM, but not with {args.graph_representation}")
    return args

def run_eval_epoch(model:GraphT5Classifier, data:OpenData, batch_size:int, device:str, split:str, graph_representation:str, eos_usage:str, use_text:str, max_seq_len:int, predict_source:bool, source_to_index:Dict[str,int]):
    with torch.no_grad():
        # losses = []
        labels = []
        pred_classes = []
        ranks = []  # ranks of the correct class. 0 is the highest rank.
        entailed_by_texts = []

        if predict_source:
            label_sources = []
            pred_sources = []
            ranks_sources = []


        with data as d:
            data_indicess = list(range(len(d[split]))) 

        for data_indices in tqdm(chunker(data_indicess, batch_size), total=len(data_indicess)//batch_size):
            # create batch
            logging.debug("get batch")
            input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label = get_batch(data=data, split=split, data_indices=data_indices, device=device, tokenizer=model.tokenizer, graph_representation=graph_representation, eos=eos_usage, use_text=use_text, max_seq_len=max_seq_len, predict_source=predict_source, source_to_index=source_to_index)

            logging.debug("forward")
            logits = model.forward(
                input_ids=input_ids,
                relative_position=relative_position,
                sparsity_mask=sparsity_mask,
                use_additional_bucket=use_additional_bucket
            )

            if predict_source:
                logits_source = logits[1]
                logits = logits[0]
                

            logits = torch.cat([
                get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_indices))
            ], dim=0)

            pred_class, rank = get_preds_and_rank(preds=logits, labels=label)
            pred_classes = pred_classes + pred_class.tolist()
            ranks = ranks + rank.tolist()
            labels = labels + label.tolist()
            entailed_by_texts = entailed_by_texts + entailed_by_text.tolist()

            if predict_source:
                logits_source = torch.cat([
                    get_embedding(sequence_embedding=logits_source[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                    for i in range(len(data_indices))
                ], dim=0)
                pred_source, rank_source = get_preds_and_rank(preds=logits_source, labels=source_label)
                pred_sources = pred_sources + pred_source.tolist()
                ranks_sources = ranks_sources + rank_source.tolist()
                label_sources = label_sources + source_label.tolist()

    logging.debug("get metrics")
    metrics = get_metrics(pred_classes=pred_classes, ranks=ranks, labels=labels, entailed_by_texts=entailed_by_texts)
    if predict_source:
        metrics_sources = get_metrics(pred_classes=pred_sources, ranks=ranks_sources, labels=label_sources, entailed_by_texts=entailed_by_texts)
        return metrics, metrics_sources
    return metrics

def run_train_epoch(model:GraphT5Classifier, data:OpenData, data_indicess:List[int], criterion:nn.Module, optimizer:torch.optim.Optimizer, batch_size:int, gradient_accumulation_steps:int, device:str, split:str, graph_representation:str, eos_usage:str, use_text:str, predict_source:bool, source_to_index:Dict[str,int], save_model_after_seen_instances:List[int], save_model_dir:Optional[Path], num_seen_instances:int, stop_training_after_seen_instances:bool):
    losses = []
    accuracies = []
    weights = []
    accuracies_source = []
    optimizer.zero_grad()

    random.shuffle(data_indicess)

    for i, data_indices in tqdm(enumerate(chunker(data_indicess, batch_size)), total=len(data_indicess)//batch_size):
        # create batch
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label = get_batch(data=data, split=split, data_indices=data_indices, device=device, tokenizer=model.tokenizer, graph_representation=graph_representation, eos=eos_usage, use_text=use_text, max_seq_len=args.max_seq_len, predict_source=predict_source, source_to_index=source_to_index)
        num_seen_instances += len(label)
        logits = model.forward(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )
        if predict_source:
            logits_source = logits[1]
            logits = logits[0]

        logits = torch.cat([
            get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
            for i in range(len(data_indices))
        ], dim=0)
        
        loss1 = criterion(logits, label)

        if predict_source:
            logits_source = torch.cat([
                get_embedding(sequence_embedding=logits_source[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_indices))
            ], dim=0)
            loss2 = criterion(logits_source, source_label)

            loss = (1 - args.source_prediction_weight) * loss1 + args.source_prediction_weight * loss2
        else:
            loss = loss1

        loss.backward()

        if (i+1) % gradient_accumulation_steps == 0 or (i+1) == len(data_indicess)//batch_size:
            optimizer.step()
            optimizer.zero_grad()
        else:
            if num_seen_instances in save_model_after_seen_instances:
                logging.warning(f'saving for {num_seen_instances=} actually saw less instances, because gradients were not updated')

        accuracy = get_accuracy(logits, label)
        if predict_source:
            accuracy_source = get_accuracy(logits_source, source_label)
        else:
            accuracy_source = 0
        losses.append(loss.item())
        accuracies.append(accuracy)
        accuracies_source.append(accuracy_source)
        weights.append(len(label)) 
        if num_seen_instances in save_model_after_seen_instances:
            logging.info(f'saving model after {sum(weights)} seen instances')
            if save_model_dir is not None:
                model.save_pretrained(save_model_dir.joinpath(f'seen_instances_{num_seen_instances}'))
                tmp_loss = np.average(losses, weights=weights)
                tmp_accuracy = np.average(accuracies, weights=weights)
                tmp_accuracy_source = np.average(accuracies_source, weights=weights)
                wandb_log = {
                    "train/accuracy": tmp_accuracy, 
                    "train/loss": tmp_loss, 
                    "train/accuracy_source": tmp_accuracy_source,
                    "num_seen_instances": num_seen_instances,
                }
                wandb.log(wandb_log)
            else:
                logging.warning('not saving model')
        if num_seen_instances >= max(save_model_after_seen_instances) and stop_training_after_seen_instances:
            break

    loss = np.average(losses, weights=weights)
    accuracy = np.average(accuracies, weights=weights)
    accuracy_source = np.average(accuracies_source, weights=weights)
    return loss, accuracy, accuracy_source, num_seen_instances

def main(args):
    if args.continue_training:
        assert args.save_model_dir is not None
        assert args.save_model_dir.exists()
    else:
        if args.save_model_dir is not None:
            if args.save_model_dir.exists():
                assert len(list(args.save_model_dir.glob('*'))) == 0, f'{args.save_model_dir} is not empty: {list(args.save_model_dir.glob("*"))}'
            args.save_model_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'saving model to {args.save_model_dir}')
        else:
            logging.warning('not saving model')

    if not args.device.startswith('cuda'):
        logging.warning(f'using CPU {args.device}, training might be slow.')
    else:
        logging.info(f'using GPU {args.device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logging.info('connect to data (will be loaded on the fly)')
    data = OpenData(use_graph=args.use_graph, entailed_triplets_only=args.entailed_triplets_only)

    with data as d:
        num_classes = d.attrs['num_labels']
        num_sources = d.attrs['num_sources']
        source_to_index = json.loads(d.attrs['source_to_index'])

    if not args.continue_training:    
        logging.info('load model')
        if args.predict_source:
            model = DualGraphT5Classifier(config=DualGraphT5Classifier.get_config(num_classes1=num_classes, num_classes2=num_sources, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
        else:
            model = GraphT5Classifier(config=GraphT5Classifier.get_config(num_classes=num_classes, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
        if args.num_additional_buckets != 0:
            logging.info(f'init relative position bias with {args.num_additional_buckets} additional buckets')
            model.t5model.init_relative_position_bias(modelsize=args.modelsize, init_decoder=False, init_additional_buckets_from=args.init_additional_buckets_from)
        if args.reset_params:
            logging.info('resetting model parameters')
            reset_params(model=model)
        last_epoch = -1
    else:
        logging.info('load model from save_model_dir')
        fns = list(args.save_model_dir.glob('epoch_*'))
        assert len(list(fns)) > 0, f'no model found in {args.save_model_dir}'
        fn = max(fns, key=lambda fn: float(fn.name.split('_')[-1]))
        last_epoch = float(fn.name.split('_')[-1])
        logging.info(f'loading model from {fn}')

        if args.predict_source:
            model = DualGraphT5Classifier.from_pretrained(fn)
        else:
            model = GraphT5Classifier.from_pretrained(fn)
    model.to(args.device)

    # loss and optimizer
    criterion = args.criterion()

    freeze_params(s=args.params_to_train, model=model)
    optimizer = args.optimizer(model.parameters(), lr=args.learning_rate)

    best_epoch = 0
    best_dev_accuracy = 0
    best_dev_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    if args.save_at in ['log_seen_instances', 'all']:
        save_model_after_seen_instances = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        assert 512 % (args.train_batch_size * args.gradient_accumulation_steps) == 0, f'{args.train_batch_size}, {args.gradient_accumulation_steps}'
    else:
        save_model_after_seen_instances = []

    logging.info('train the model')
    num_seen_instances = 0
    for train_epoch in range(args.num_epochs):
        with data as d:
            train_indices = list(range(len(d['train'])))
        random.shuffle(train_indices)

        for sub_epoch, sub_train_indices in enumerate(chunker(train_indices, len(train_indices) // args.num_evals_per_epoch), 1):
            epoch = train_epoch + (sub_epoch / args.num_evals_per_epoch)

            if args.continue_training and epoch <= last_epoch:
                    num_seen_instances += len(sub_train_indices)
                    continue

            logging.debug(f'epoch {epoch}')

            train_loss, train_accuracy, train_accuracy_source, num_seen_instances = run_train_epoch(model=model, data=data, data_indicess=sub_train_indices, criterion=criterion, optimizer=optimizer, batch_size=args.train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device, split='train', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, predict_source=args.predict_source, source_to_index=source_to_index, save_model_after_seen_instances=save_model_after_seen_instances, save_model_dir=args.save_model_dir, num_seen_instances=num_seen_instances, stop_training_after_seen_instances=args.stop_training_after_seen_instances)
            logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f} # {train_accuracy_source = :.2f}')

            if args.save_model_dir is not None:
                assert args.num_evals_per_epoch <= 100, 'need to adjust the number of digits in the epoch number in the filename. Might also need to be adjusted for loading the model.'
                model.save_pretrained(args.save_model_dir.joinpath(f'epoch_{epoch:.2f}'))

            if args.run_eval:
                # get dev scores
                dev_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='val', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
                if args.predict_source:
                    dev_metrics_sources = dev_metrics[1]
                    dev_metrics = dev_metrics[0]
                logging.info(f'dev   - {epoch = } # {dev_metrics["all/accuracy"] = :.2f}')

                # get test scores
                test_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='test', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
                if args.predict_source:
                    test_metrics_sources = test_metrics[1]
                    test_metrics = test_metrics[0]
                    
                logging.info(f'test   - {epoch = } # {test_metrics["all/accuracy"] = :.2f}')

                if dev_metrics['all/accuracy'] > best_dev_accuracy:
                    best_epoch = epoch
                    best_dev_accuracy = dev_accuracy
                    best_test_accuracy = test_accuracy
            else:
                dev_accuracy = float('nan')
                test_accuracy = float('nan')

            wandb_log = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, "train/accuracy_source": train_accuracy_source,
                "num_seen_instances": num_seen_instances,
            }
            if args.run_eval:
                dev_metrics = {f'dev/{k}': v for k, v in dev_metrics.items()}
                test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
                wandb_log = {**wandb_log, **dev_metrics, **test_metrics}
                if args.predict_source:
                    dev_metrics_sources = {f'dev/source/{k}': v for k, v in dev_metrics_sources.items()}
                    test_metrics_sources = {f'test/source/{k}': v for k, v in test_metrics_sources.items()}
                    wandb_log = {**wandb_log, **dev_metrics_sources, **test_metrics_sources}
            wandb.log(wandb_log)

            last_epoch = epoch

            if args.run_eval and (epoch - best_epoch >= args.early_stopping):
                logging.info(f'stopped early at epoch {epoch}')
                stopped_early = True
                break
            if num_seen_instances >= max(save_model_after_seen_instances) and args.stop_training_after_seen_instances:
                logging.info(f'stopped early at epoch {epoch}. {num_seen_instances=}. {save_model_after_seen_instances=}')
                stopped_early = True
                break
        if stopped_early:
            break


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter  # makes wandb log the default values
    )
    add_args_shared(parser)
    add_args(parser)
    args = get_args(parser)

    # args.device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    # logging
    logging.basicConfig(
        level=args.logging_level,
        # format=f"%(asctime)s [%(levelname)s] %(message)s (Line %(lineno)d in %(filename)s)",
        format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
        datefmt=f"%H:%M:%S",
    )

    # wandb
    if args.reset_params:
        assert args.graph_representation in ['lGLM', 'gGLM'], f"reset_params can only be used with lGLM or gGLM, but not with {args.graph_representation}"
        gr_name = args.graph_representation[0] + 'GT'
    else:
        gr_name = args.graph_representation
    name = f'{args.wandb_name_prefix}{gr_name:_<4}_{args.params_to_train:_<4}_{args.modelsize}_t={args.use_text}_g={args.use_graph}_ps={args.predict_source}_eto={args.entailed_triplets_only}'
    wandb_run = wandb.init(
        mode=args.wandb_mode,
        project="GLM-ShortTrain-text_guided_relation_prediction",
        name=name,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_lr={args.learning_rate}_resetparams={args.reset_params}_modelsize={args.modelsize}_eos={args.eos_usage}',
        tags=['LM']
    )

    main(args)

    logging.info('done with main')
