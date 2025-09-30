#!/bin/bash
# Run GLM with graph-only input (testing setup with T5-small)

cd ../GraphLanguageModels

modelsize=t5-small # t5-small t5-base t5-large

python experiments/encoder/text_guided_relation_prediction/evaluate_LM.py \
    --seed 0 \
    --params_to_train all \
    --graph_representation gGLM \
    --reset_params False \
    --modelsize $modelsize \
    --eval_batch_size 128 \
    --eos_usage False \
    --init_additional_buckets_from 1e6 \
    --device cuda \
    --logging_level INFO \
    --use_text False \
    --use_graph True \
    --get_dev_scores False \
    --eval_epochs 1 \
    --predict_source False \
    --eval_by_num_seen_instances True \
    --entailed_triplets_only False \
    --save_preds True
