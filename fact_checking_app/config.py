import os

import torch
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification, AutoTokenizer, \
    AutoModelForSequenceClassification, BertForSequenceClassification, AutoConfig, BertConfig


class Config:
    def get_base_dir() -> str:
        ret_val: str = os.path.abspath("..") if os.getcwd().split("/")[-1] == "training" else os.path.abspath(".")
        subfolder: str = os.path.join(ret_val, "fact_checking_app")
        return subfolder if os.path.exists(subfolder) else ret_val

    COVID_CREDIBILITY_CLASSIFIER = "covid-credibility-classifier"
    DOCKER_COMPOSE_SERVICE_NAME = "ccc"
    DOCKER_IMAGE_CREDIBILITY_SERVICE = "konstantinschulz/covid-credibility-classifier:v1"
    DOCKER_PORT_CREDIBILITY = 8000
    HOST_PORT_CREDIBILITY = 8181
    base_dir: str = get_base_dir()
    streamlit_dir: str = os.path.join(base_dir, "streamlit_app")
    model_dir: str = os.path.join(streamlit_dir, "gbert_fang_covid_classifier")  # "deepset/gbert-base"
    model_config_path: str = os.path.join(model_dir, "config.json")
    # fetch fine-tuned model and config
    config: BertConfig = AutoConfig.from_pretrained(model_config_path)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fang_covid_dir: str = os.path.abspath('../../../fang-covid/articles')
    model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                                                              config=config).to(device)
    max_length: int = 512
    # model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "gbert_base"))


Config.model.eval()
