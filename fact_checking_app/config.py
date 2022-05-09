import os

import torch
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification


class Config:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streamlit_dir: str = os.path.abspath("streamlit_app")
    model_dir: str = os.path.join(streamlit_dir, "model")
    model_config_path: str = os.path.join(model_dir, "config.json")
    tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
    # fetch fine-tuned model and config
    config: DistilBertConfig = DistilBertConfig.from_json_file(model_config_path)
    model: DistilBertForSequenceClassification = \
        DistilBertForSequenceClassification.from_pretrained(model_dir, config=config).to(device)
