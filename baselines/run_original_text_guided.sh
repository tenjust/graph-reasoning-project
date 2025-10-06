#!/bin/bash
# Run the paper’s best setup: gGLM-large with finetuning (text + graph)

cd ../GraphLanguageModels || {
    echo "Error: GraphLanguageModels repo not found. Please clone it next to this repo."
    exit 1
}

modelsize=t5-small # t5-small t5-base t5-large

python -m experiments.encoder.text_guided_relation_prediction.evaluate_LM \
    --seed 0 \
    --params_to_train all \
    --graph_representation gGLM \
    --reset_params False \
    --modelsize $modelsize \
    --eval_batch_size 128 \
    --eos_usage False \
    --init_additional_buckets_from 1e6 \
    --device cpu \
    --logging_level INFO \
    --use_text FullyConnected \
    --use_graph True \
    --get_dev_scores False \
    --eval_epochs 1 \
    --predict_source True \
    --eval_by_num_seen_instances True \
    --entailed_triplets_only False \
