#!/bin/bash
# Train a text-guided Graph Language Model (REBEL dataset)

cd ../GraphLanguageModels

modelsize=t5-small  # options: t5-small, t5-base, t5-large

python -m experiments.encoder.text_guided_relation_prediction.train_LM \
    --graph_representation gGLM \
    --reset_params False \
    --modelsize $modelsize \
    --use_text FullyConnected \
    --use_graph True \
    --params_to_train all \
    --seed 0 \
    --entailed_triplets_only False \
    --device cpu \
    --logging_level INFO \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --num_epochs 1 \
    --save_model True
