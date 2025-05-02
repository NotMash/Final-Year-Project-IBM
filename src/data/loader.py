import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_dataset(json_path):
    data = load_json(json_path)
    return data["courses"] if isinstance(data, dict) else data

# src/model_training_evaluation/security_fine_tuner.py
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import torch.nn as nn

class SBERTFineTuner:
    def __init__(self, base_model_path):
        self.model = self.build_model(base_model_path)

    def build_model(self, path):
        word_embedding_model = models.Transformer(path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        dense = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=128,
            activation_function=nn.GELU()
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense])

    def train(self, train_examples, val_examples, output_path, epochs=4, batch_size=8):
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        loss_func = losses.MultipleNegativesRankingLoss(self.model)
        self.model.fit(
            train_objectives=[(train_loader, loss_func)],
            epochs=epochs,
            output_path=output_path
        )