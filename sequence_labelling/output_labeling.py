import logging
import json
import re
import torch
import spacy
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from spacy.training import Alignment

from concat_ner.model import RobertaClassifier, DomainSpecificClassifier, CombinedClassifier, FlairClassifier
from concat_ner.token_encoder import RobertaTokenEncoder, ClassicTokenEncoder, CombinedEncoder, FlairTokenEncoder
from concat_ner.label_encoder import LabelEncoder
from concat_ner.data_handler import DataHandler

SEED = 42
LABELS = ["SCOPE", "CONDITION", "DEMAND", "O"]

ROBERTA_TOKENS = ["<s>", "</s>"]

nlp = spacy.load("en_core_web_sm")

puncutation = "\.,:;_-"

def predict_labels(data_handler, model_obj, device=torch.device('cpu')):
    model = model_obj['model']
    token_encoder = model_obj['token_encoder']
    model.eval()
    predicted = []

    logging.info("predicting labels...")

    sent_with_predictions = []
    for batch in tqdm(data_handler):
        data = batch[0]
        sent, tags = data
        #print(sent)
        #print(tags)
        #exit()
        with torch.no_grad():
            encoded_input = token_encoder.encode_with_labels(batch)
            _, logits = model(encoded_input)  # model should only return logits if eval
            logits = logits.squeeze()

            pred_idxs = logits.argmax(dim=1).squeeze(0)
            pred_labels = token_encoder.label_encoder.decode_labels(pred_idxs)
            predicted.append(pred_labels)

            encoded_ids = encoded_input['input_ids']
            decoded_sent = token_encoder.decode(encoded_ids.squeeze(0))

        #sent_with_predictions.append({"sent" : sent, "pred": pred_labels})
        sent_with_predictions.append(list(zip(decoded_sent, pred_labels)))
    return sent_with_predictions


def collect_spans(docs):
    doc_spans = []
    for doc in docs:
        spans = []
        i = 0
        while i < len(doc):
            #print("i: {}".format(i))
            tok, tag = doc[i]
            j = i + 1
            while j < len(doc):
                #print("j: {}".format(j))
                next_tok, next_tag = doc[j]
                if next_tag != tag:  # end of span
                    break
                else:
                    j += 1

            if tag != 'O':
                text = " ".join([tok for tok,_ in doc[i:j]])
                spans.append({"text": text,
                              "token_start": i, # spacy starts counting on 0
                              "token_end": j-1,   # inclusive end tags
                              "label": tag})
            i = j
        #print(spans)
        doc_spans.append(spans)
    return doc_spans


def normalize_tags(tags):
    tagset = set(tags)

    if "O" in tagset:
        tagset.remove("O")

    if len(tagset) == 0:
        return "O"
    elif len(tagset) == 1:
        return tagset.pop()
    elif len(tagset) > 0:
        return tags[0] # return the first tag in the token


def normalize_tokens_and_align_labels(docs):
    doc_tokens = []
    for doc in docs:
        tokens = []
        i = 0
        while i < len(doc):
            #print("i: {}".format(i))
            tok, tag = doc[i]
            tags = [tag]
            j = i + 1

            # identify the complete token
            # add all tags from subtokens in tag
            if tok not in ROBERTA_TOKENS:
                tok = re.sub("<\\/s>", "", tok)
                tok = re.sub("âĢĵ", "–", tok)
                tok = re.sub("âĢĶ", "—", tok)
                tok = re.sub("¾", ";", tok)
                tok = re.sub("Í", "", tok)
                tok = re.sub("ÏĴ", "ϒ", tok)
                tok = re.sub("âĢĻ", "\'", tok)
                tok = re.sub("<s>", "", tok)
                if re.search("Ġ", tok):
                    tok = re.sub("Ġ", "", tok)
                    while j < len(doc):
                        #print("j: {}".format(j))
                        if re.search("Ġ", doc[j][0]): # new token
                            #print("new token: {}".format(tok))
                            break
                        else:
                            new_tok = doc[j][0]
                            new_tok = re.sub("<\\/s>", "", new_tok)
                            new_tok = re.sub("âĢĵ", "–", new_tok)
                            new_tok = re.sub("âĢĶ", "—", new_tok)
                            new_tok = re.sub("¾", ";", new_tok)
                            new_tok = re.sub("Í", "", new_tok)
                            new_tok = re.sub("ÏĴ", "ϒ", new_tok)
                            new_tok = re.sub("âĢĻ", "\'", new_tok)
                            tok += new_tok
                            #print("Extending token to: {}".format(tok))
                            tags.append(doc[j][1])
                            j += 1
                else:
                    print("Something wrong, should not happen")
                    print("tok: {}".format(tok))
                    print("tags: {}".format(tags))
                    print("i: {}".format(i))
                    exit()

                tokens.append((tok, normalize_tags(tags)))
                #print("adding token: {} and tag: {}".format(tok, tag))
                if "</s>" in tok:
                    print("something went wrong")
                    print("tok: {}".format(tok))
                    tok = re.sub("<\\/s>", "", tok)
                    print("tok: {}".format(tok))
                    exit()

            i = j
        doc_tokens.append(tokens)
    return doc_tokens



def save_data(data, path):
    with path.open(mode='w') as F:
        for d in data:
            json.dump(d, F)
            F.write("\n")

        print("Written {} sentences to {}".format(len(data), path))


def align_spans_with_document2(doc, spans, tokens):
    orig_tokens = [tok['text'] for tok in doc['tokens']]
    new_tokens = [tok[0] for tok in tokens]
    print(orig_tokens)
    print(new_tokens)
    #try:
        #align = Alignment.from_strings(orig_tokens, new_tokens)
        #print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
        #print(f"a -> b, mapping: {align.x2y.data}")  # array([0, 1, 2, 3, 4, 4, 5, 6]) : two tokens both refer to "'s"
        #print(f"b -> a, lengths: {align.y2x.lengths}")  # array([1, 1, 1, 1, 2, 1, 1])   : the token "'s" refers to two tokens
        #print(f"b -> a, mappings: {align.y2x.data}")  # array([0, 1, 2, 3, 4, 5, 6, 7])
    #except:
    for orig, new in zip(orig_tokens, new_tokens):
        print("{} {}", orig_tokens, new_tokens)
        if orig != new:
            print("difference: {} - {}".format(orig, new))
            exit()

def align_spans_with_document(doc, spans, tokens):
    doc['spans'] = spans
    new_tokens = []
    character_no = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i][0]
        if i < len(tokens) -1:
            next_token = tokens[i+1][0]
            if next_token in puncutation:
                ws = False
            else:
                ws = True
        else:
            ws = False

        if tok == '-':
            ws = False


        new_tokens.append({
            "text": tok,
            "start" : character_no,
            "end" : character_no + (len(tok)),
            "id" : i,
            "ws" : ws
        })
        character_no += len(tok) + 1
        i += 1
    doc['tokens'] = new_tokens

    for span in spans:
        start = new_tokens[span['token_start']]['start']
        end = new_tokens[span['token_end']]['end']
        span['start'] = start
        span['end'] = end


    return doc




def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("model_name", help="Model name (roberta|domain|flair|combined)")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("data", help="file with data")
    #parser.add_argument("--save", help="paths to save output of the model", action="store", default="/home/ole/src/Concat_ner/scope_output/output.txt")
    parser.add_argument("--flair", help="include flair embeddings in the combined model", default=False, action='store_true')
    #parser.add_argument("--emb_path", help="path to embeddings", default="/home/ole/src/Concat_ner/embeddings/domain_emb.bin")
    parser.add_argument("--vocab_path", help="path to embed. vocab", default="/home/ole/src/Concat_ner/vectors/dnv_vocab.pkl")
    parser.add_argument("--preprocess", help="case normalization, lemmatization etc.", default=False, action='store_true')

    args = parser.parse_args()
    logging.info("Running with arugments:")
    for arg in vars(args):
        logging.info("%s : %s", arg, getattr(args, arg))

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.model_path).exists():
        print("Model: {} not found... exiting".format(args.model_path))
        exit()


    # Load model and tools
    model_obj = {'model' : torch.load(args.model_path),
                 "action": "eval"}

    # INITIALIZE MODEL
    if args.model_name.lower() == 'roberta':
        model_obj['token_encoder'] = RobertaTokenEncoder(LabelEncoder(LABELS))
    elif args.model_name.lower() == 'domain':
        model_obj['token_encoder'] = ClassicTokenEncoder(LabelEncoder(LABELS), args.vocab_path)
    elif args.model_name.lower() == 'flair':
        model_obj['token_encoder'] = FlairClassifier(LabelEncoder(LABELS))
    elif args.model_name.lower() == 'combined':
        model_obj['token_encoder'] = CombinedEncoder(LabelEncoder(LABELS), args.vocab_path, args.flair)
    else:
        print("Unknown model: {}, try [roberta|domain|combined]".format(args.model_name))
        print("Exiting...")
        exit()

    data_handler = DataHandler(args.data)
    data = data_handler.get_dataset()

    #predictions = predict_labels['output']
    predictions = predict_labels(data, model_obj)
    normalized_tokens = normalize_tokens_and_align_labels(predictions)
    doc_spans = collect_spans(normalized_tokens)
    original_documents = data_handler.get_original_documents()

    for doc, spans, tokens in zip(original_documents, doc_spans, normalized_tokens):
        align_spans_with_document(doc, spans, tokens)
        #doc['spans'] = spans


    path = Path(args.data)
    out_path = path.parent / (path.stem + "_labeled.jsonl")
    save_data(original_documents, out_path)


if __name__ == '__main__':
    main()
