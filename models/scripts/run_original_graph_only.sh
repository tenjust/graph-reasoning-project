#!/bin/bash
# Run GLM with graph-only input (testing setup with T5-small)

cd ~ || 1
cd "$(find . -type d -name 'graph-reasoning-project' -print -quit)" || {
  echo "Error: Not able to locate project directory."
  exit 1
}
cd models/GraphLanguageModels || {
    echo "Error: GraphLanguageModels repo not found. Please clone it next to this repo."
    exit 1
}

modelsize=t5-small # t5-small t5-base t5-large

echo "Running graph-only baseline (T5-small, gGLM)..."

python -m experiments.encoder.relation_prediction.evaluate_LM \
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

echo "Done."