from pathlib import Path

# Automatically detect the project root (the folder containing this file)
ROOT = Path(__file__).resolve().parent

# Useful pre-defined subpaths
DATA_DIR = ROOT / "baselines" / "GraphLanguageModels" / "data"
REBEL_DIR = DATA_DIR / "rebel_dataset"

# Optional helper
def print_paths():
    print("ROOT:", ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("REBEL_DIR:", REBEL_DIR)