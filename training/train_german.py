from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import TrainingArguments, Trainer, IntervalStrategy
from transformers.integrations import TensorBoardCallback

from classes import FangCovidDataset
from config import Config


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {'accuracy': acc, 'tn': tn, 'tp': tp, 'fp': fp, 'fn': fn}


def train_german():
    fcds: FangCovidDataset = FangCovidDataset()
    train_indices, val_indices = train_test_split(fcds.indices, test_size=0.005, random_state=42)  # 0.05
    train_dataset = Subset(fcds, indices=train_indices)
    val_dataset = Subset(fcds, indices=val_indices)
    eval_steps: int = 1  # 12
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 8
        per_device_eval_batch_size=64,  # 64
        logging_dir='./logs',
        logging_steps=eval_steps,  # 100
        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=eval_steps * 2,  # 2000
        dataloader_pin_memory=False,
        save_total_limit=3,
        gradient_accumulation_steps=1,  # 32
        eval_steps=eval_steps,
        # report_to=['wandb'],
        # disable_tqdm=True,
    )
    trainer = Trainer(
        model=Config.model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[TensorBoardCallback],
    )
    trainer.train()


# train_german()
