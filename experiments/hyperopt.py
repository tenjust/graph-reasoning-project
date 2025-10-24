import optuna
from subprocess import run
import ast

def objective(trial):
    # === 1. Define search space ===
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    num_epochs = trial.suggest_int("num_epochs", 1, 3)
    use_amr = False # or True, depending on which experiment you are running

    # === 2. Construct command for training ===
    cmd = [
        "python", "experiments/train.py",
        "--train_data", "baselines/GraphLanguageModels/data/rebel_dataset/REBEL_AMR_TRIPLES.train.hdf5",
        "--model_name", "t5-small",
        "--output_dir", f"models/optuna_trial_{trial.number}",
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate)
    ]
    if use_amr:
        cmd.append("--use_amr")

    # === 3. Run training script ===
    result = run(cmd, capture_output=True, text=True)

    # === 4. Extract final loss ===
    loss = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if "train_loss" in line:
            # print("Trying to parse line:", line)
            try:
                d = ast.literal_eval(line)
                # print("Parsed dict:", d)
                loss = float(d.get("train_loss", 10.0))
                # print("Extracted loss:", loss)
            except Exception as e:
                print("Error parsing line:", e)

    if loss is None:
        loss = 10.0  # fallback if not found (bad trial)

    print(f"Trial {trial.number}: loss={loss}, lr={learning_rate}, batch={batch_size}, amr={use_amr}")
    return loss


if __name__ == "__main__":
    sampler = optuna.integration.SkoptSampler()  # uses Gaussian Process internally
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_params)
