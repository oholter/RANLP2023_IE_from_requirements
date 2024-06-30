import torch
import logging
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, Linear, Softmax, Dropout, ReLU, LSTM
from transformers import RobertaModel

DROPOUT_PROBA = 0.5

class RobertaLargeClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size=None, emb_path=None, hidden_size=200, full_finetuning=False):
        super(RobertaLargeClassifier, self).__init__()
        self.full_finetuning = full_finetuning
        self._roberta = RobertaModel.from_pretrained("roberta-large", cache_dir="./cache")
        self._dropout = Dropout(DROPOUT_PROBA)
        #self._hidden_layer = Linear(2 * hidden_size, num_labels)
        self._hidden_layer = Linear(1024, num_labels)
        self._criterion = CrossEntropyLoss()
        self._softmax = Softmax(2)
        self._relu = ReLU()
        #self._lstm = LSTM(input_size = 1024,
        #                  hidden_size = hidden_size,
        #                  bidirectional=True,
        #                  batch_first=True,
        #                  num_layers=2)

    def forward(self, encoded_input):
        """
        :param idx: torch.LongTensor word ids
        :return:
        """
        ids = encoded_input['input_ids']
        attention_masks = encoded_input['attention_masks']
        if 'labels' in encoded_input:
            labels = encoded_input['labels']
        else:
            labels = None

        # idx: [ 1, sent_length ]
        if self.full_finetuning:
            x = self._roberta(input_ids = ids, attention_mask = attention_masks)[0]
        else:
            with torch.no_grad():  # not training the Bert model
                x = self._roberta(input_ids = ids, attention_mask = attention_masks)[0]
        # xs : [1, 512, 1024]

        #x = self._dropout(x)
        #h_n, _ = self._lstm(x)
        #x = self._relu(h_n)

        x = self._dropout(x)
        x = self._hidden_layer(x)
        x = self._relu(x)

        #x = self._dropout(x)
        #x = self._output_layer(x)
        #x = self._relu(x)


        #  xs : [1, 4, out_dim]
        if labels is not None:
            loss = self._criterion(x.squeeze(), labels.squeeze())
            return loss, self._softmax(x)
        else:
            return None, self._softmax(x)



class RobertaClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size=None, emb_path=None, hidden_size=200, full_finetuning=False):
        super(RobertaClassifier, self).__init__()
        self.full_finetuning = full_finetuning
        self._roberta = RobertaModel.from_pretrained("roberta-base", cache_dir="./cache")
        #self._roberta.num_labels = num_labels
        self._dropout = Dropout(DROPOUT_PROBA)
        self._hidden_layer = Linear(2 * hidden_size, num_labels)
        #self._output_layer = Linear(hidden_size, num_labels)
        self._criterion = CrossEntropyLoss()
        self._softmax = Softmax(2)
        self._relu = ReLU()
        self._lstm = LSTM(input_size = 768,
                          hidden_size = hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=2)

    def forward(self, encoded_input):
        """
        :param idx: torch.LongTensor word ids
        :return:
        """
        ids = encoded_input['input_ids']
        attention_masks = encoded_input['attention_masks']
        if 'labels' in encoded_input:
            labels = encoded_input['labels']
        else:
            labels = None

        # idx: [ 1, sent_length ]
        if self.full_finetuning:
            x = self._roberta(input_ids = ids, attention_mask = attention_masks)[0]
        else:
            with torch.no_grad():  # not training the Bert model
                x = self._roberta(input_ids = ids, attention_mask = attention_masks)[0]
        # xs : [1, 512, 768]

        x = self._dropout(x)
        h_n, _ = self._lstm(x)
        x = self._relu(h_n)

        x = self._dropout(x)
        x = self._hidden_layer(x)
        x = self._relu(x)

        #x = self._dropout(x)
        #x = self._output_layer(x)
        #x = self._relu(x)


        #  xs : [1, 4, out_dim]
        if labels is not None:
            loss = self._criterion(x.squeeze(), labels.squeeze())
            return loss, self._softmax(x)
        else:
            return None, self._softmax(x)


class DomainSpecificClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size, emb_path, hidden_size=200, full_finetuning=False):
        super(DomainSpecificClassifier, self).__init__()
        self.full_finetuning = full_finetuning
        logging.info("loading pretrained embeddings, this may take some time")
        embs_npa = np.load(emb_path)
        self._emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        self.emb_dim = self._emb.embedding_dim
        self._hidden_layer = Linear(2*hidden_size, num_labels)
        #self._output_layer = Linear(hidden_size, num_labels)
        self._dropout = Dropout(p=0.5)
        self._softmax = Softmax(2)
        self._criterion = CrossEntropyLoss()
        self._relu = ReLU()
        self._lstm = LSTM(input_size = self.emb_dim,
                          hidden_size = hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=2)

    def forward(self, encoded_input):
        ids = encoded_input['input_ids']
        if 'labels' in encoded_input:
            labels = encoded_input['labels']
        else:
            labels = None

        if self.full_finetuning:
            x = self._emb(ids)
        else:
            with torch.no_grad():
                x = self._emb(ids)

        x = self._dropout(x)
        h_n, _ = self._lstm(x)
        x = self._relu(h_n)

        x = self._dropout(x)
        x = self._hidden_layer(x)
        x = self._relu(x)

        #x = self._dropout(x)
        #x = self._output_layer(x)
        #x = self._relu(x)

        # x: [1, sent_len, out_dim]
        if labels is not None:
            loss = self._criterion(x.squeeze(), labels.squeeze())
            return loss, self._softmax(x)
        else:
            return None, self._softmax(x)


class CombinedClassifier(nn.Module):
    """
    todo: implement
    todo: how to best combine word embeddings and word piece embeddings?
          e.g., what is the last hidden feature of roBerta?
    """
    def __init__(self, num_labels, vocab_size, emb_path, hidden_size=200, full_finetuning=False, flair=False):
        super(CombinedClassifier, self).__init__()
        self.full_finetuning = full_finetuning

        # domain emb
        logging.info("loading pretrained embeddings, this may take some time")
        embs_npa = np.load(emb_path)
        self._emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        self.emb_dim = self._emb.embedding_dim

        # roberta
        self._roberta = RobertaModel.from_pretrained("roberta-base", cache_dir="./cache")

        # network
        self._hidden_layer = Linear(2 * hidden_size, num_labels)
        #self._output_layer = Linear(hidden_size, num_labels)
        self._dropout = Dropout(p=0.5)
        self._softmax = Softmax(2)
        self._criterion = CrossEntropyLoss()
        self._relu = ReLU()
        self.flair = flair
        if flair:
            input_emb_size = self.emb_dim + 768 + 2048 # roberta emb_dim + flair emb_dim
        else:
            input_emb_size = self.emb_dim + 768
        self._lstm = LSTM(input_size = input_emb_size,
                          hidden_size = hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=2)


    def forward(self, encoded_input):
        attention_masks = encoded_input['attention_masks']
        if 'labels' in encoded_input:
            labels = encoded_input['labels']
        else:
            labels = None

        d_ids = encoded_input['input_ids2']
        r_ids = encoded_input['input_ids']
        if self.flair:
            f_embs = encoded_input['flair_embeddings']

        # domain_ids
        if self.full_finetuning:
            domain_x = self._emb(d_ids)

        else:
            with torch.no_grad():
                domain_x = self._emb(d_ids)

        # roberta_ids
        if self.full_finetuning:
            roberta_x = self._roberta(input_ids = r_ids, attention_mask = attention_masks)[0]
        else:
            with torch.no_grad():  # not training the Bert model
                roberta_x = self._roberta(input_ids = r_ids, attention_mask = attention_masks)[0]

        if self.flair:
            x = torch.cat((roberta_x, domain_x, f_embs), dim=2)
        else:
            x = torch.cat((roberta_x, domain_x), dim=2)

        x = self._dropout(x)
        h_n, _ = self._lstm(x)
        x = self._relu(h_n)

        x = self._dropout(x)
        x = self._hidden_layer(x)
        x = self._relu(x)

        #x = self._dropout(x)
        #x = self._output_layer(x)
        #x = self._relu(x)

        # x: [sent_len, out_dim]
        if labels is not None:
            loss = self._criterion(x.squeeze(), labels.squeeze())
            return loss, self._softmax(x)
        else:
            return None, self._softmax(x)

class FlairClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size=None, emb_path=None, hidden_size=200, full_finetuning=False):
        super(FlairClassifier, self).__init__()
        self.full_finetuning = full_finetuning
        self._hidden_layer = Linear(2*hidden_size, num_labels)
        #self._output_layer = Linear(hidden_size, num_labels)
        self._dropout = Dropout(p=0.5)
        self._softmax = Softmax(2)
        self._criterion = CrossEntropyLoss()
        self._relu = ReLU()
        self.emb_dim = 2048 # flair emb dim = 2048
        self._lstm = LSTM(input_size = self.emb_dim,
                          hidden_size = hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=2)

    def forward(self, encoded_input):
        #ids = encoded_input['input_ids']
        embs = encoded_input['flair_embeddings']
        if 'labels' in encoded_input:
            labels = encoded_input['labels']
        else:
            labels = None

        #if self.full_finetuning:
            #x = self._emb(ids)
        #else:
            #with torch.no_grad():
                #x = self._emb(ids)

        x = self._dropout(embs)
        h_n, _ = self._lstm(x)
        x = self._relu(h_n)

        x = self._dropout(x)
        x = self._hidden_layer(x)
        x = self._relu(x)

        #x = self._dropout(x)
        #x = self._output_layer(x)
        #x = self._relu(x)

        # x: [1, sent_len, out_dim]
        if labels is not None:
            loss = self._criterion(x.squeeze(), labels.squeeze())
            return loss, self._softmax(x)
        else:
            return None, self._softmax(x)

if __name__ == "__main__":
    #model = DomainSpecificClassifier(3, 285055, 400, emb_path="/home/ole/src/Concat_ner/embeddings/model.txt")
    model = DomainSpecificClassifier(3, 285055, 4)
    enc_input = {}
    X = torch.LongTensor([[111, 11, 1, -1]])
    Y = torch.LongTensor([[0, 0, 1, 0]])
    enc_input["input_ids"] = X
    enc_input['labels'] = Y
    print(model(enc_input))
