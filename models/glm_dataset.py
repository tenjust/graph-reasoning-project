from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

@dataclass(frozen=True)
class Edge:
    src: int; rel: str; dst: int

@dataclass(frozen=True)
class GraphExample:
    node_labels: List[str]
    edges: List[Edge]
    label: Optional[int] = None

class GraphTokenizer:
    def __init__(self, name="t5-small"):
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.sep = self.tok.convert_tokens_to_ids("<sep>") if "<sep>" in self.tok.get_vocab() else None
        if self.sep is None:
            self.tok.add_special_tokens({"additional_special_tokens": ["<sep>"]})
            self.sep = self.tok.convert_tokens_to_ids("<sep>")

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

class GraphDataset(Dataset):
    """
    For each example:
      - Flatten nodes and edges into a single token list (input_ids)
      - Build a token-level attention mask so:
           • tokens within the same node span see each other
           • tokens of an edge span see tokens of its src & dst node spans
    """
    def __init__(self, path: str, tokenizer: GraphTokenizer):
        self.tok = tokenizer
        self.data: List[GraphExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                nodes = j["nodes"]
                edges = [Edge(*e) for e in j["edges"]]
                label = j.get("label")
                self.data.append(GraphExample(nodes, edges, label))

    def __len__(self): return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ex = self.data[i]
        ids: List[int] = []
        node_spans: List[Tuple[int,int]] = []

        # 1) pack nodes: [ node_tokens, <sep> ]
        for lbl in ex.node_labels:
            s = len(ids)
            ids += self.tok.encode(lbl)
            ids += [self.tok.sep]
            node_spans.append((s, len(ids)))

        # 2) pack edges similarly: [ rel_tokens, <sep> src_index, <sep> dst_index, <sep> ]
        edge_spans: List[Tuple[int,int]] = []
        for e in ex.edges:
            s = len(ids)
            ids += self.tok.encode(e.rel) + [self.tok.sep]
            ids += self.tok.encode(str(e.src)) + [self.tok.sep]
            ids += self.tok.encode(str(e.dst)) + [self.tok.sep]
            edge_spans.append((s, len(ids)))

        L = len(ids)
        input_ids = torch.tensor(ids, dtype=torch.long)

        # 3) build attention mask (L x L) boolean
        mask = torch.zeros((L, L), dtype=torch.bool)

        # intra-span full attention
        for s,e in node_spans + edge_spans:
            mask[s:e, s:e] = True

        # edge <-> node connectivity
        for k, e in enumerate(ex.edges):
            es, ee = edge_spans[k]
            ns, ne = node_spans[e.src]
            ds, de = node_spans[e.dst]
            mask[es:ee, ns:ne] = True; mask[ns:ne, es:ee] = True
            mask[es:ee, ds:de] = True; mask[ds:de, es:ee] = True

        label = torch.tensor(ex.label if ex.label is not None else -100, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": mask, "labels": label}
