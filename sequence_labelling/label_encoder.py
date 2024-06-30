SPECIAL_TAG = "SPEC"
SPECIAL_ID = -100


class LabelEncoder:
    def __init__(self, labels):
        self.labels = labels
        self.label2idx = {}
        self.idx2label = {}
        for i, label in enumerate(labels):
            self.label2idx[label] = i
            self.idx2label[i] = label

        self.idx2label[SPECIAL_ID] = SPECIAL_TAG
        self.label2idx[SPECIAL_TAG] = SPECIAL_ID

    def encode_label(self, label):
        if label in self.label2idx:
            return self.label2idx[label]
        else:
            raise Exception("label {} not found in label2idx".format(label))

    def decode_label(self, idx):
        if idx in self.idx2label:
            return self.idx2label[idx]
        else:
            raise Exception("idx {} not found in idx2label".format(idx))

    def encode_labels(self, labels):
        encoded_labels = []
        for l in labels:
            if l in self.label2idx:
                encoded_labels.append(self.label2idx[l])
            else:
                Exception("label {} not found in label2idx".format(l))
        return encoded_labels

    def decode_labels(self, batch, flatten=False):
        decoded = []
        for ids in batch:
            ids = ids.int()
            labels = []
            for i in ids:
                #print(i)
                i = i.item()
                if i in self.idx2label:
                    labels.append(self.idx2label[i])
                else:
                    print("idx2label: {}".format(self.idx2label))
                    raise Exception("idx {} not found in idx2label".format(i))
            decoded.append(labels)
        if flatten:
            return [item for sublist in decoded for item in sublist]
        else:
            return decoded
