import re
import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    BertForTokenClassification
)
from ..utils.util import accuracy_precision_recall_f1
import csv

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sents, labels, rationales=None):
        self.sents = sents
        self.labels = labels
        self.rationales = rationales

    def __getitem__(self, index):
        sent = self.sents[index]
        label = self.labels[index]
        if self.rationales is not None:
            rationale = self.rationales[index]
            return sent, label, rationale
        return sent, label

    def __len__(self):
        return len(self.sents)

def my_collate_fn(batch):
    collate_sents = []
    collate_labels = []

    for sent, label in batch:
        collate_sents.append(sent)
        collate_labels.append(label)
    return [collate_sents, torch.as_tensor(collate_labels)]
    
def my_collate_fn_token(batch):
    collate_sents = []
    collate_labels = []

    for sent, label in batch:
        collate_sents.append(sent)
        collate_labels.append(torch.as_tensor(label))
    collate_labels = torch.nn.utils.rnn.pad_sequence(collate_labels, batch_first=True)
    return [collate_sents, collate_labels]

def my_collate_fn_rationale(batch):
    collate_sents = []
    collate_labels = []
    collate_rationales = []

    for sent, label, rationale in batch:
        collate_sents.append(sent)
        collate_labels.append(label)
        collate_rationales.append(torch.as_tensor(rationale))
    collate_rationales = torch.nn.utils.rnn.pad_sequence(collate_rationales, batch_first=True)
    return [collate_sents, torch.as_tensor(collate_labels), collate_rationales]

def load_sst(path, tokenizer, dataset, num_labels, rationale_path, token_cls, lower=False):
    dataset_orig = []
    sents, labels, rationales = [], [], []

    if num_labels == 2:
        label_idx = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    else:
        label_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    if dataset == 'coco':
        with open(path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            if token_cls:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    line = line.split('\t')
                    if num_labels == 2 and label_idx[line[0].strip()] == 2:
                        continue
                    label = list(map(int, line_rationale.strip().split()))
                    sent = (line[1].strip(), line[2].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig
            elif rationale_path is not None:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    line = line.split('\t')
                    label = label_idx[line[0].strip()]
                    if num_labels == 2 and label == 2:
                        continue
                    sent = (line[1].strip(), line[2].strip())
                    rationale = list(map(int, line_rationale.strip().split(' ')))
                    sents.append(sent)
                    labels.append(label)
                    rationales.append(rationale)
                    dataset_orig.append((sent, label, rationale))
                return MyDataset(sents, labels, rationales), dataset_orig
            else:
                for line in lines:
                    line = line.split('\t')
                    label = label_idx[line[0].strip()]
                    if num_labels == 2 and label == 2:
                        continue
                    sent = (line[1].strip(), line[2].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig
    else:
        with open(path, mode="r", encoding="utf-8") as f:
            lines = csv.reader(f)
            lines.__next__()

            if token_cls:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    if num_labels == 2 and label_idx[line[1].strip()] == 2:
                        continue
                    label = list(map(int, line_rationale.strip().split()))
                    sent = (line[2].strip(), line[3].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig
            elif rationale_path is not None:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    label = label_idx[line[1].strip()]
                    if num_labels == 2 and label == 2:
                        continue
                    sent = (line[2].strip(), line[3].strip())
                    rationale = list(map(int, line_rationale.strip().split(' ')))
                    sents.append(sent)
                    labels.append(label)
                    rationales.append(rationale)
                    dataset_orig.append((sent, label, rationale))
                return MyDataset(sents, labels, rationales), dataset_orig
            else:
                for line in lines:
                    label = label_idx[line[1].strip()]
                    if num_labels == 2 and label == 2:
                        continue
                    sent = (line[2].strip(), line[3].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig

class SentimentClassificationSST(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model)

    def prepare_data(self):
        # assign to use in dataloaders
        if not hasattr(self, "train_dataset") or not hasattr(
            self, "train_dataset_orig"
        ):
            self.train_dataset, self.train_dataset_orig = load_sst(
                self.hparams.train_filename, self.tokenizer, self.hparams.dataset, self.hparams.num_labels, rationale_path=self.hparams.train_rationale, token_cls=self.hparams.token_cls
            )
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_dataset_orig"):
            self.val_dataset, self.val_dataset_orig = load_sst(
                self.hparams.val_filename, self.tokenizer, self.hparams.dataset, self.hparams.num_labels, rationale_path=self.hparams.val_rationale, token_cls=self.hparams.token_cls
            )

    def train_dataloader(self, shuffle=True):
        if self.hparams.token_cls:
            collate_fn = my_collate_fn_token
        elif self.hparams.train_rationale is not None:
            collate_fn = my_collate_fn_rationale
        else:
            collate_fn = my_collate_fn
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=8
        )

    def val_dataloader(self):
        if self.hparams.token_cls:
            collate_fn = my_collate_fn_token
        elif self.hparams.val_rationale is not None:
            collate_fn = my_collate_fn_rationale
        else:
            collate_fn = my_collate_fn
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, collate_fn=collate_fn, num_workers=8
        )

    def training_step(self, batch, batch_idx=None):
        # input_ids, mask, labels = batch
        inputs = self.tokenizer.batch_encode_plus(batch[0], pad_to_max_length=True, return_tensors='pt').to('cuda')
        input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = batch[1]
        rationale_ids = None

        if self.hparams.token_cls:
            loss, logits = self.forward(input_ids, mask, token_type_ids, labels=labels)
            logit = torch.masked_select(logits.argmax(-1).view(-1), (mask.view(-1)) == 1)
            label = torch.masked_select(labels.view(-1), (mask.view(-1)) == 1)
            acc, _, _, f1 = accuracy_precision_recall_f1(
                logit, label, average=True
            )
        else:
            if len(batch) == 3:
                rationale_ids = batch[2]
            logits = self.forward(input_ids, mask, token_type_ids, rationale_ids=rationale_ids)[0]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(
                -1
            )
            acc, _, _, f1 = accuracy_precision_recall_f1(
                logits.argmax(-1), labels, average=True
            )

        outputs_dict = {
            "acc": acc,
            "f1": f1,
        }

        outputs_dict = {
            "loss": loss,
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        return outputs_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in ("val_acc", "val_f1")
        }

        outputs_dict = {
            "val_loss": -outputs_dict["val_f1"],
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class BertSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        config = BertConfig.from_pretrained(self.hparams.model)
        config.num_labels = self.hparams.num_labels
        config.rationale = None
        if self.hparams.token_cls:
            self.net = BertForTokenClassification.from_pretrained(
            self.hparams.model, config=config
        )
        else:
            if self.hparams.train_rationale is not None:
                config.rationale = True
            self.net = BertForSequenceClassification.from_pretrained(
                self.hparams.model, config=config
            )

    def forward(self, input_ids, mask, token_type_ids, rationale_ids=None, labels=None):
        if self.hparams.token_cls:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=labels)
        else:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, rationale_ids=rationale_ids, labels=labels)


class RecurrentSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.emb = BertForSequenceClassification.from_pretrained(
            hparams.model
        ).bert.embeddings.word_embeddings.requires_grad_(False)

        self.gru = torch.nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=self.emb.embedding_dim,
            batch_first=True,
        )

        self.classifier = torch.nn.Linear(self.emb.embedding_dim, 5)

    def forward(self, input_ids, mask, labels=None):
        x = self.emb(input_ids)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(-1), batch_first=True, enforce_sorted=False
        )

        _, h = self.gru(x)

        return (self.classifier(h[0]),)
