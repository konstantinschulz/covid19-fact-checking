from typing import Any, Dict
import torch

from transformers import BatchEncoding
from transformers.file_utils import PaddingStrategy, TensorType
from transformers.modeling_outputs import SequenceClassifierOutput

from config import Config
from elg import FlaskService
from elg.model import ClassificationResponse


class CovidCredibilityClassifier(FlaskService):

    def convert_outputs(self, content: str) -> ClassificationResponse:
        encodings: BatchEncoding = Config.tokenizer(
            content, truncation=True, padding=PaddingStrategy.MAX_LENGTH, max_length=Config.max_length,
            return_tensors=TensorType.PYTORCH)
        inputs: dict = {key: val.to(Config.device) for key, val in encodings.data.items()}
        sco: SequenceClassifierOutput = Config.model(**inputs)
        score_dict: Dict[str, float] = dict(credibility_score=round(float(torch.sigmoid(sco.logits[0][0])), 5))
        return ClassificationResponse(classes=[{"class": k, "score": v} for k, v in score_dict.items()])

    def process_text(self, content: Any) -> ClassificationResponse:
        return self.convert_outputs(content.content)


ccc: CovidCredibilityClassifier = CovidCredibilityClassifier(Config.COVID_CREDIBILITY_CLASSIFIER,
                                                             path=f"/process/{Config.DOCKER_COMPOSE_SERVICE_NAME}")
app = ccc.app
