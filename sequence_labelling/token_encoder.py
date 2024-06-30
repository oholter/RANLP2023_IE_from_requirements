import logging
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from concat_ner.label_encoder import SPECIAL_TAG
import torch
from textblob import Word
import spacy
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence


UNK_IDX = 0
UNK_TOK = '[UNK]'
PUNCT_IDX = 1
PUNCT_TOK = '[PUNCT]'
PAD_IDX = 2
PAD_TOK = '[PAD]'
STOP_IDX = 3
STOP_TOK = '[STOP]'

SPECIAL_TOKENS = ["[UNK]", "[PUNKCT]", "[PAD]", "[STOP]"]
PUNCTS = ['.', ',', ':', ';']
SPECIAL_IDXS = [0, 1, 2, 3]

en = spacy.load('en_core_web_sm')
STOPWORDS = en.Defaults.stop_words

#ROBERTA_MODEL = "roberta-base"
#ROBERTA_MODEL = "roberta-large"
ROBERTA_MODEL = "bert-large-cased"

class TokenEncoder(ABC):
    @abstractmethod
    def encode_with_labels(self, examples):
        pass

    @abstractmethod
    def encode(self, tokens):
        pass

    @abstractmethod
    def decode(self, enc_tokens):
        pass


class FlairTokenEncoder(TokenEncoder):
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def encode_with_labels(self, examples):
        encoded = {}
        #X, Y = zip(*[(self.encode(tokens).embedding, self.label_encoder.encode_labels(labels))
                     #for tokens, labels in examples])

        X = torch.Tensor(0, len(examples[0][0]), 2048)
        for tokens, _ in examples:
            enc_tokens = self.encode(tokens)
            embs = torch.Tensor(0, 2048)
            for enc_token in enc_tokens:
                #print(embs.size())
                #print(enc_token.embedding.unsqueeze(0).size())
                embs = torch.cat([embs, enc_token.embedding.unsqueeze(0)], dim=0)
            #print(X.size())
            #print(embs.unsqueeze(0).size())
            X = torch.cat([X, embs.unsqueeze(0)], dim=0)

        #X = [self.encode(tokens) for tokens, _ in examples]
        Y = [self.label_encoder.encode_labels(labels) for _, labels in examples]
        #encoded['input_ids'] = torch.LongTensor(X)
        #print("X: {}".format(X))
        #print("X[0]: {}".format(X[0]))
        encoded['flair_embeddings'] = X
        encoded['labels'] = torch.LongTensor(Y)
        encoded['attention_mask'] = None
        return encoded

    def encode(self, tokens):
        fe = FlairEmbeddings('news-forward')
        sent = Sentence(tokens)
        #print(sent)
        embedded_sentence = fe.embed(sent)
        #print("emb.sent: {}".format(embedded_sentence))

        # for some reason f.embed wraps the sent in a list (!?)
        return embedded_sentence[0]


    def decode(self, ids):
        pass

class ClassicTokenEncoder(TokenEncoder):
    def __init__(self, label_encoder, vocab_path):
        self.label_encoder = label_encoder
        logging.info("loading vocab...")
        self.idx2word = np.load(vocab_path)
        self.word2idx = {}
        for i in range(len(self.idx2word)):
            self.word2idx[self.idx2word[i]] = i
        self.vocab_size = len(self.idx2word)

    def encode_with_labels(self, examples):
        encoded = {}
        X, Y = zip(*[(self.encode(tokens), self.label_encoder.encode_labels(labels))
                     for tokens, labels in examples])
        encoded['input_ids'] = torch.LongTensor(X)
        encoded['labels'] = torch.LongTensor(Y)
        encoded['attention_mask'] = None
        return encoded

    def encode(self, tokens):
        encoded_tokens = []
        for tok in tokens:
            lemma = Word(tok).lemmatize()
            if tok in self.word2idx:
                encoded_tokens.append(self.word2idx[tok])
            elif tok.lower() in self.word2idx:
                encoded_tokens.append(self.word2idx[tok.lower()])
            elif lemma in self.word2idx:
                encoded_tokens.append(self.word2idx[lemma])
            elif lemma.lower() in self.word2idx:
                encoded_tokens.append(self.word2idx[lemma.lower()])
            elif tok in PUNCTS:
                encoded_tokens.append(self.word2idx[PUNCT_TOK])
            elif tok in STOPWORDS:
                encoded_tokens.append(self.word2idx[STOP_TOK])
            else:
                encoded_tokens.append(self.word2idx[UNK_TOK])
        return encoded_tokens


    def decode(self, ids):
        return [self.idx2word[idx] for idx in ids]


class RobertaTokenEncoder(TokenEncoder):
    def __init__(self, label_encoder):
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL,
                                                       fast=True,
                                                       add_prefix_space=True)
        self.label_encoder = label_encoder

    def encode_with_labels(self, batch):
        encoded = self.tokenize_and_align_labels(batch)
        for key, value in encoded.items():
            encoded[key] = torch.LongTensor(value)
        return encoded

    def encode(self, tokens):
        return self.tokenizer(tokens,
                              is_split_into_words=True,
                              truncation=True)

    def tokenize_and_align_labels(self, example):
        token_batch = example['tokens']
        label_batch = example['labels']

        tokenized_input = self.tokenizer(token_batch,
                                         is_split_into_words=True,
                                         truncation=True,
                                         padding=True)
        aligned_batch_labels = []
        for batch_idx in range(len(token_batch)):
            org_labels = label_batch[batch_idx]
            aligned_labels = []
            previous_word_idx = None
            for word_idx in tokenized_input.word_ids(batch_index=batch_idx):
                if word_idx is None:
                    aligned_labels.append(self.label_encoder.encode_label(SPECIAL_TAG))
                elif word_idx != previous_word_idx:
                    aligned_labels.append(self.label_encoder.encode_label(org_labels[word_idx]))
                else:
                    aligned_labels.append(self.label_encoder.encode_label(SPECIAL_TAG))
                previous_word_idx = word_idx

            aligned_batch_labels.append(aligned_labels)


        tokenized_input['labels'] = aligned_batch_labels
        return tokenized_input

    def decode(self, enc_tokens):
        #return self.tokenizer.decode(enc_tokens, skip_special_tokens=False)
        return self.tokenizer.convert_ids_to_tokens(enc_tokens)

class CombinedEncoder(TokenEncoder):
    def __init__(self, label_encoder, vocab_path, flair):
        # roberta
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base",
                                                               fast=True,
                                                               add_prefix_space=True)
        self.label_encoder = label_encoder
        self.flair = flair
        logging.info("loading vocab...")
        self.idx2word = np.load(vocab_path)
        self.word2idx = {}
        self.vocab_size = len(self.idx2word)
        for i in range(len(self.idx2word)):
            self.word2idx[self.idx2word[i]] = i

    def encode_with_labels(self, examples):
        tokenized_texts_and_labels = [
            self.tokenize_and_align_labels(example) for example in examples
        ]
        input_ids = [example['input_ids'] for example in tokenized_texts_and_labels]
        input_ids2 = [example['input_ids2'] for example in tokenized_texts_and_labels]
        labels = [example['labels'] for example in tokenized_texts_and_labels]
        attention_mask = [example['attention_mask'] for example in tokenized_texts_and_labels]
        if self.flair:
            flair_embeddings = torch.cat([example['flair_embeddings'].unsqueeze(0) for example in tokenized_texts_and_labels], dim=0)

        encoded = {}
        encoded['input_ids'] = torch.LongTensor(input_ids)
        encoded['input_ids2'] = torch.LongTensor(input_ids2)
        encoded['labels'] = torch.LongTensor(labels)
        encoded['attention_mask'] = torch.LongTensor(attention_mask)
        if self.flair:
            encoded['flair_embeddings'] = flair_embeddings

        return encoded

    def build_vocab(self, emb_lines, vocab_size):
        logging.info("Building vocab, this may take some time")
        for i, line in tqdm(enumerate(emb_lines[1:]), total=vocab_size):
            self.word2idx[line.split()[0]] = i
            self.idx2word[i] = line.split()[0]

    def tokenize_and_align_labels(self, example):
        tokens, labels = example
        tokenized_input = self.roberta_tokenizer(tokens,
                                         is_split_into_words=True,
                                         truncation=True)
        aligned_labels = []
        aligned_tokens = []
        previous_word_idx = None
        for word_idx in tokenized_input.word_ids(batch_index=0):
            if word_idx is None:
                aligned_labels.append(self.label_encoder.encode_label(SPECIAL_TAG))
                aligned_tokens.append(PAD_TOK)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label_encoder.encode_label(labels[word_idx]))
                aligned_tokens.append(tokens[word_idx])
            else:
                aligned_labels.append(self.label_encoder.encode_label(SPECIAL_TAG))
                aligned_tokens.append(PAD_TOK)
            previous_word_idx = word_idx

        tokenized_input['labels'] = aligned_labels
        tokenized_input['input_ids2'] = self.classic_encode(aligned_tokens)
        if self.flair:
            tokenized_input['flair_embeddings'] = self.flair_encode(aligned_tokens)
        return tokenized_input

    def classic_encode(self, tokens):
        encoded_tokens = []
        for tok in tokens:
            lemma = Word(tok).lemmatize()
            if lemma in self.word2idx:
                encoded_tokens.append(self.word2idx[lemma])
            elif tok in PUNCTS:
                encoded_tokens.append(self.word2idx[PUNCT_TOK])
            elif tok in STOPWORDS:
                encoded_tokens.append(self.word2idx[STOP_TOK])
            else:
                encoded_tokens.append(self.word2idx[UNK_TOK])
        return encoded_tokens

    def flair_encode(self, tokens):
        fe = FlairEmbeddings('news-forward')
        sent = Sentence(tokens)
        # for some reason f.embed wraps the sent in a list (!?)
        enc_tokens = fe.embed(sent)[0]
        embs = torch.Tensor(0, 2048)
        for enc_token in enc_tokens:
            embs = torch.cat([embs, enc_token.embedding.unsqueeze(0)], dim=0)
        return embs

    def encode(self, tokens):
        raise NotImplementedError("encode is currently not defined for combined model")


    def decode(self, ids):
        return self.roberta_tokenizer.decode(ids)


if __name__ == '__main__':
    from concat_ner.label_encoder import LabelEncoder
    label_encoder = LabelEncoder(["SCOPE", "O"])
    #encoder = CombinedEncoder(label_encoder, "/home/ole/src/Concat_ner/embeddings/domain_vocab.bin")
    #encoder = ClassicTokenEncoder(label_encoder, "/home/ole/src/Concat_ner/embeddings/domain_vocab.bin")
    #encoder = FlairTokenEncoder(label_encoder, "/home/ole/src/Concat_ner/embeddings/domain_vocab.bin")
    encoder = RobertaTokenEncoder(label_encoder)
    output = encoder.encode_with_labels({"tokens": [["encoding", "of", "waterfallmountains", "is", "not", "defined"],
                                                    ["the", "end"]],
                                          "labels": [["O", "O", "SCOPE", "O", "O", "O"],
                                                     ["O", "O"]]}
                                         )
    #output = encoder.encode(["encoding", "of", "waterfallmountains", "is", "not", "defined"])

    print(output)
