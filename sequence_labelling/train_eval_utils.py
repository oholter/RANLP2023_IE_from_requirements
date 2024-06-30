import torch
from collections.abc import Iterable

LABELS = ["SCOPE", "CONDITION", "DEMAND", "THEME", "O"]
#LABELS = ["SCOPE", "O"]
#LABELS = ["P641H", "P641T",
#         "P361H", "P361T",
#         "P47H", "P47T",
#         "P31H", "P31T",
#         "P17H", "P17T",
#         "P276H", "P276T",
#         "P279H", "P279T",
#         "P27H", "P27T",
#         "O"]
#LABELS = ["P20H", "P20T", "O"]
UNK_TAG = "UNK"
SPECIAL_TAG = "SPEC"

label2id = {}
id2label = {}
for i, label in enumerate(set(LABELS)):
    label2id[label] =  i
    id2label[i] = label

id2label[-100] = SPECIAL_TAG
label2id["SPEC"] = -100

def tokenize_and_align_labels(example, tokenizer):
    tokens, labels = example
    #print("tokens: {}".format(tokens))
    #print("labls: {}".format(labels))
    tokenized_input = tokenizer(tokens,
                                is_split_into_words=True,
                                truncation=True,
                                padding='max_length')

    #aligned_labels = []
    #for i, label in enumerate(labels):
        #print("tokenized_input: {}".format(tokenized_input))
        #print("batch_index: {}".format(i))
        #word_ids = tokenized_input.word_ids(batch_index=i)
        #previous_word_idx = None
        #label_ids = []
        #for word_idx in word_ids:
            #if word_idx is None:
                #label_ids.append(-100)
            #elif word_idx != previous_word_idx:
                #label_ids.append(labels[word_idx])
            #else:
                #label_ids.append(-100)
            #previous_word_idx = word_idx
        #aligned_labels.append(label_ids)

    aligned_labels = []
    previous_word_idx = None
    for word_idx in tokenized_input.word_ids(batch_index=0):
        if word_idx is None:
            aligned_labels.append(label2id[SPECIAL_TAG])
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id[labels[word_idx]])
        else:
            aligned_labels.append(label2id[SPECIAL_TAG])
        previous_word_idx = word_idx



    tokenized_input['labels'] = aligned_labels
    #print("tokenized_input: {}".format(tokenized_input))
    return tokenized_input







def create_encodings(example, tokenizer):
    sent, labels = example
    encoded_sent = tokenizer(sent, truncation=True, padding='max_length', return_tensors='pt', return_offset_mapping=True)
    encoded_labels = labels + ['O'] * (tokenizer.model_max_length - len(labels))

    #alignment = zip(encoded_sent, encoded_labels)
    #for tok, lab in alignment:
        #print(tok, lab)
    print(encoded_sent)

    exit()


    return {**encoded_sent, "labels": encoded_labels}


def idxs2tags(idx):
        """
        :param idx: one an iterable collection of indexes
        :return: a tag or list of tags
        >>> df = pd.DataFrame({'words': ["To", "read", "Alan", "Turing"], 'ner': ["O", "O", "B-PER", "I-PER"]})
        >>> dh = DataHandler(df)
        >>> dh.idx2tag(0)
        'O'
        >>> dh.idx2tag(1)
        'B-PER'
        >>> dh.idx2tag([1, 2])
        ['B-PER', 'I-PER']
        >>> dh.idx2tag(200)
        'O'
        >>> dh.idx2tag(torch.Tensor([0, 1, 12, 8, 4]))
        ['O', 'B-PER', 'I-PROD', 'I-GPE_LOC', 'I-ORG']
        """

        if isinstance(idx, Iterable):
            if isinstance(idx, torch.Tensor):
                tag = [id2label.get(i.item(), UNK_TAG) for i in idx]
            else:
                tag = [id2label.get(i, UNK_TAG) for i in idx]
        else:
            tag = id2label.get(idx, UNK_TAG)
        return tag
