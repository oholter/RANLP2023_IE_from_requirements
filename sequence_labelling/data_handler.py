from tqdm import tqdm
import logging
import math
import json
import random
from pathlib import Path
import stanza
import spacy

nlp = spacy.load('en_core_web_sm')
STOPWORDS = nlp.Defaults.stop_words
PUNCTS = ['.', ',', ':', ';', '(', ')', ']', '[', '}', '{', '-']

random.seed(42)

class DataHandler:
    def __init__(self, path,
                 batch_size=1,
                 preprocess=False,
                 shuffle=False,
                 test_size=0.1):

        self.path = path
        self.remove_stopwords = False
        self.remove_punct = True
        self.lemmatize = False
        self.lowercase = True
        self.batch_size = batch_size
        #self.data = list(self.load_data(shuffle=shuffle))
        self.data = self.load_data()
        if shuffle:
            random.shuffle(self.data)

        self.train_data, self.test_data = self.get_train_test_split(test_size)
        self.train = list(self.data_batch(self.train_data))
        if self.test_data:
            self.test = list(self.data_batch(self.test_data))
        else:
            self.test = None

        if preprocess:
            self.apply_preprocessing()

    def get_all_batches(self):
        return list(self.data_batch(self.data))

    def apply_preprocessing(self):
        """lemmatization is done using stanfordnlp from
        stanza to be in accordance with the Farhad embed. paper

        - case normalized
        - lemmatized
        remove:
            - stopwords
            - punctuation
        """
        logging.info("preprocessing data, this may take some time ...")
        nlp = stanza.Pipeline(lang="en", processors="lemma,tokenize", tokenize_pretokenized=True)
        new_data = []
        for batch in tqdm(self.data):
            # typically we have only one 'sent'/label pair per batch
            new_sent = []
            new_labels = []
            for sent, labels in batch:
                doc = nlp([sent])
                #for sent_id, sent in enumerate(doc.sentences):
                #print(labels)
                #print(doc.sentences[0].text)
                for word in doc.sentences[0].words:
                    if word.text in STOPWORDS and self.remove_stopwords:
                        continue # do not add
                    #if word.pos == "PUNCT":
                    if word.text in PUNCTS and self.remove_punct:
                        continue # do not add
                    else:
                        if self.lemmatize:
                            new_sent.append(word.lemma)
                        elif self.lowercase:
                            new_sent.append(word.text.lower())
                        else:
                            new_sent.append(word)
                        # word.id starts with 1
                        new_labels.append(labels[word.id-1])
            new_data.append([(new_sent, new_labels)])

        self.data = new_data

    def get_dataset(self):
        return self.data

    def get_train_batch(self):
        return self.train

    def get_test_batch(self):
        return self.test

    def _get_scopes(self, data):
        scopes = []
        for doc in data:
            #print(doc)
            if 'answer' not in doc or doc['answer'] == 'accept':
                if 'scopes' in doc['meta']:
                    scopes.append(doc['meta']['scopes'])
                else:
                    scopes.append([])
        return scopes

    def _get_conditions(self, data):
        conditions = []
        for doc in data:
            #print(doc)
            if 'answer' not in doc or doc['answer'] == 'accept':
                if 'conditions' in doc['meta']:
                    conditions.append(doc['meta']['conditions'])
                else:
                    conditions.append([])
        return conditions

    def _get_demands(self, data):
        demands = []
        for doc in data:
            #print(doc)
            if 'answer' not in doc or doc['answer'] == 'accept':
                if 'demands' in doc['meta']:
                    demands.append(doc['meta']['demands'])
                else:
                    demands.append([])
        return demands

    def get_attributes(self, attribute="scope"):
        if attribute not in ['scope', 'condition', 'demand']:
            logging.warning("Unexpected attribute %s", attribute)
            return []

        p = Path(self.path)
        if p.suffix == ".json":
            with p.open(mode="r") as F:
                data = json.load(F)
        elif p.suffix == ".jsonl":
            with p.open(mode="r") as F:
                data = [json.loads(l) for l in list(F)]
        else:
            logging.error("%s has unknown suffix %s", p, p.suffix)

        if attribute == 'scope':
            return self._get_scopes(data)
        elif attribute == 'condition':
            return self._get_conditions(data)
        else:
            return self._get_demands(data)

    def get_original_documents(self):
        p = Path(self.path)
        if p.suffix == ".json":
            with p.open(mode="r") as F:
                data = json.load(F)
        elif p.suffix == ".jsonl":
            with p.open(mode="r") as F:
                data = [json.loads(l) for l in list(F)]
        else:
            logging.error("%s has unknown suffix %s", p, p.suffix)

        return data


    def get_train_test_split(self, test_size):
        if test_size > 1 or test_size < test_size:
            raise Exception("expected 0 < test_size < 1, got %f exiting", test_size)
        if test_size == 0:
            return self.data, None

        n_test = math.floor(len(self.data) * test_size)
        train_data = self.data[n_test:]
        test_data = self.data[:n_test]

        return train_data, test_data

    def load_data(self):
        logging.info("Loading dataset from %s", self.path)
        train_path = Path(self.path)
        if train_path.suffix == ".json":
            with train_path.open(mode="r") as F:
                data = json.load(F)
        elif train_path.suffix == ".jsonl":
            with train_path.open(mode="r") as F:
                data = [json.loads(l) for l in list(F)]
        else:
            logging.error("%s has unknown suffix %s", train_path, train_path.suffix)

        return data

    def data_batch(self, data):
        """
        creating a batch of sentence/label pairs
        """

        batch = {"tokens": [],
                 "labels": []}

        sample_num = 0
        for doc in data:
            # only include accepted annotations
            if 'answer' not in doc or doc['answer'] == 'accept':
                if 'tokens' in doc:
                    toks = [tok["text"] for tok in doc['tokens']]
                else: # untokenized document
                    if 'text' in doc:
                        s_doc = nlp(doc['text'])
                        toks = [tok.text for tok in s_doc]
                    else: # empty document -> ignore
                        continue
                if 'spans' in doc:
                    spans = [(span['token_start'], span['token_end'], span['label']) for span in doc['spans']]
                else:
                    spans = []
                labs = ['O']*len(toks)
                for start, end, label in spans:
                    for tok_idx in range(start, end+1): #prodigy uses inclusive end
                        try:
                            labs[tok_idx] = label.upper()
                        except IndexError:
                            print("Index error exiting...")
                            print("toks: {}".format(toks))
                            print("index: {}".format(tok_idx))
                            exit()

                #batch.append((toks, labs))
                batch['tokens'].append(toks)
                batch['labels'].append(labs)

                if ((sample_num + 1) % self.batch_size) == 0:
                    yield batch
                    batch = {"tokens": [],
                            "labels": []}

                sample_num += 1

        # any "leftovers"
        if batch['tokens']:
            yield batch



if __name__ == '__main__':
    data_handler = DataHandler("/home/ole/src/Req_annot/annotations/test/data22.jsonl",
                               shuffle=False,
                               preprocess=False,
                               batch_size=8)
    #data = data_handler.get_dataset()
    batch = data_handler.get_train_batch()
    print(batch[0])
    #print(data[-1])


