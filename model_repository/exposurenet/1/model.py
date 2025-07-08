import os
import torch
import torch.nn as nn
import numpy as np
import math
import itertools
from collections import defaultdict
import triton_python_backend_utils as pb_utils
import json
import logging

logging.basicConfig(level=logging.INFO)


def normalize_text(text):
    return str(text).strip().lower()


def create_embedding_matrix(categorical_feature_cols, list_feature_cols, vocab_lookup):
    embedding_dict = nn.ModuleDict()
    for feature_col in categorical_feature_cols:
        embedding_dict[feature_col] = nn.Embedding(
            len(vocab_lookup[feature_col]) + 1,
            math.ceil(2 * math.sqrt(len(vocab_lookup[feature_col]))),
            sparse=False,
        )
    for feature_col in list_feature_cols:
        embedding_dict[feature_col] = nn.EmbeddingBag(
            len(vocab_lookup[feature_col]) + 1,
            math.ceil(2 * math.sqrt(len(vocab_lookup[feature_col]))),
            sparse=False,
            mode="sum",
        )
    return embedding_dict


class ExposureNet(nn.Module):
    def __init__(
        self,
        dense_feature_cols,
        categorical_feature_cols,
        list_feature_cols,
        vocab_lookup,
        dense_stats,
        label_quantiles,
        num_classes,
    ):
        super(ExposureNet, self).__init__()
        self.vocab_lookup = vocab_lookup
        self.dense_stats = dense_stats
        self.label_quantiles = label_quantiles

        self.dense_feature_cols = dense_feature_cols
        self.categorical_feature_cols = categorical_feature_cols
        self.list_feature_cols = list_feature_cols
        self.num_classes = num_classes

        self.embedding_dict = create_embedding_matrix(
            categorical_feature_cols, list_feature_cols, vocab_lookup
        )

        embed_dims = sum(e.embedding_dim for e in self.embedding_dict.values())
        self.linear = nn.Linear(embed_dims + len(dense_feature_cols), self.num_classes)

    def forward(self, x):
        dense_features = []
        for col in self.dense_feature_cols:
            val = float(x.get(col, 0.0))
            mean, var = self.dense_stats[col]
            std_val = (val - mean) / math.sqrt(var)
            dense_features.append(torch.tensor([std_val], dtype=torch.float32))

        dense = torch.cat(dense_features, dim=-1).unsqueeze(0)

        embedding_list = []
        for col in self.categorical_feature_cols:
            val = x.get(col, "oov")
            idx = self.vocab_lookup[col].get(normalize_text(val), 0)
            idx_tensor = torch.tensor([idx], dtype=torch.long)
            embedding_list.append(self.embedding_dict[col](idx_tensor))

        for col in self.list_feature_cols:
            val_list = x.get(col, "")
            if isinstance(val_list, str):
                val_list = val_list.split(",")
            idxs = [
                self.vocab_lookup[col].get(normalize_text(item), 0)
                for item in val_list
                if item.strip()
            ]
            offsets = [0]
            embedding_input = torch.tensor(idxs, dtype=torch.long)
            embedding_offsets = torch.tensor(offsets, dtype=torch.long)
            embedding_list.append(
                self.embedding_dict[col](embedding_input, embedding_offsets)
            )

        embeddings = torch.cat(embedding_list, dim=-1)
        combined = torch.cat([dense, embeddings], dim=-1)
        return self.linear(combined)


def get_predicted_range(pred_class: int, label_quantiles: list | np.ndarray) -> tuple:
    if pred_class == 0:
        return (float("-inf"), label_quantiles[0])  # ≤ first threshold
    elif pred_class >= len(label_quantiles):
        return (label_quantiles[-1], float("inf"))  # > last threshold
    else:
        return (label_quantiles[pred_class - 1], label_quantiles[pred_class])


class TritonPythonModel:
    def initialize(self, args):
        try:
            logging.info("Loading model...")
            path = os.path.join(os.path.dirname(__file__), "exposure_model.pt")
            checkpoint = torch.load(path, map_location="cpu")

            self.vocab_lookup = checkpoint["vocab_lookup"]
            self.dense_stats = checkpoint["dense_stats"]
            self.label_quantiles = checkpoint["label_quantiles"]
            self.dense_feature_cols = checkpoint["dense_feature_cols"]
            self.categorical_feature_cols = checkpoint["categorical_feature_cols"]
            self.list_feature_cols = checkpoint["list_feature_cols"]

            self.model = ExposureNet(
                dense_feature_cols=self.dense_feature_cols,
                categorical_feature_cols=self.categorical_feature_cols,
                list_feature_cols=self.list_feature_cols,
                vocab_lookup=self.vocab_lookup,
                dense_stats=self.dense_stats,
                label_quantiles=self.label_quantiles,
                num_classes=len(self.label_quantiles) + 1,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            logging.info("Model Loaded Successfully...")
        except Exception as e:
            logging.error("Error loading model: %s", str(e))
            raise pb_utils.TritonModelException(
                "Failed to load the ExposureNet model: {}".format(str(e))
            )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "input_str")
                if input_tensor is None:
                    raise ValueError("Missing input tensor 'input_str'")
                json_bytes = input_tensor.as_numpy()[0]
                # raw_dict = eval(json_bytes.decode("utf-8"))
                raw_dict = json.loads(json_bytes.decode("utf-8"))
                logging.info("Got request: %s", raw_dict)

                with torch.no_grad():
                    logits = self.model(raw_dict)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                logging.info("Predicted class: %s", pred_class)
                range_ = get_predicted_range(pred_class, self.label_quantiles)
                print(f"Predicted class: {pred_class}, Range: {range_}")
                out = pb_utils.Tensor(
                    "output_str",
                    np.array(
                        [f"Predicted class: {pred_class}".encode("utf-8")], dtype=object
                    ),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:
                logging.error("Error processing request: %s", str(e))
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonModelException(
                        "Failed to process request: {}".format(str(e))
                    )
                )
                responses.append(error_response)

        return responses
