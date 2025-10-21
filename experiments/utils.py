import h5py
import json
from transformers import T5Tokenizer

def load_hdf5_data(path, split="train", num_samples=None):
    with h5py.File(path, "r") as f:
        data = f[split]
        samples = []
        for i, item in enumerate(data):
            if num_samples and i >= num_samples:
                break
            sample_json = json.loads(item.decode("utf-8"))
            samples.append(sample_json)
    return samples


def load_tokenizer(model_name):
    return T5Tokenizer.from_pretrained(model_name)
