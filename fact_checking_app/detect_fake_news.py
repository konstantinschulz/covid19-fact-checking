from typing import List
from spacy.lang.en import English
from transformers import Trainer, BatchEncoding, TrainingArguments
from classes import CovidDataset
from config import Config


def detect_fake_news(sentences: List[str]) -> List[int]:
    finalResults = []  # an array of the final results - 0 for regular sentences, for suspicious sentences - 1, if no relevant claims are found or an array of relevant claims, if there are any
    # tokenize sentences
    test_encodings: BatchEncoding = Config.tokenizer(sentences, truncation=True, padding=True, max_length=256)
    # create dataset, with 0 as default label
    X_test = CovidDataset(test_encodings, [0 for x in sentences])
    train_args: TrainingArguments = TrainingArguments(
        output_dir="tmp_trainer", report_to=["all"], log_level="warning", per_device_eval_batch_size=128,
        per_device_train_batch_size=128, dataloader_pin_memory=False)
    trainer = Trainer(model=Config.model, args=train_args)
    # make predictions
    results = trainer.predict(test_dataset=X_test).predictions.argmax(-1)
    for i, result in enumerate(results):
        finalResults.append(int(result))
    return finalResults


def calculate_credibility(text: str) -> float:
    nlpSpacy = English()
    nlpSpacy.add_pipe('sentencizer')
    nlpSpacy.Defaults.stop_words |= {"we", "a", "the", "this"}
    data = []  # an array of text
    sentence_scores: List[int] = []
    if text != '':
        # parse text into sentences
        for sent in nlpSpacy(text).sents:
            data.append(sent.text)
        sentence_scores = detect_fake_news(data)
    return sum(sentence_scores) / min(1, len(sentence_scores))


# raw_input: str = "COVID-19 causes fibrosis in the lungs. BIll Gates contributed to the spread of coronavirus. A law allows people to go for a run during the state of alarm in Spain. Ghana has 307 ambulances with mobile ventilators. We demonstrated that women with mental treatment history, those in the first trimester of pregnancy and the ones that are single or in an informal relationship tend to experience higher levels of psychological distress and anxiety. Information from Guangzhou CDC was also screened. The need to physical distance requires rethinking how we deliver ophthalmic care. The world has witnessed rapid advancement and changes since the COVID-19 pandemic emerged in Wuhan, China."
# print(Config.device)
# for i in tqdm(range(25)):
#     detect_fake_news(" ".join([raw_input] * 40).split("."))
