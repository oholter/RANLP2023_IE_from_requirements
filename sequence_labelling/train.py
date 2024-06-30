import torch
import copy
import logging
from tqdm import tqdm
from keras_preprocessing.sequence import pad_sequences
from transformers import RobertaTokenizer,AdamW, get_linear_schedule_with_warmup
#from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import classification_report, f1_score
from sequence_labelling.train_eval_utils import create_encodings
from sequence_labelling.eval import Evaluator

def calculate_acc(model_obj, logits, Y_gold):
        # code from:
        # https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
        logits_clean = logits[Y_gold != -100]
        label_clean = Y_gold[Y_gold != -100]

        filter_Y = torch.logical_and(Y_gold != -100, Y_gold != model_obj['token_encoder'].label_encoder.encode_label('O'))
        logits_without_O = logits[filter_Y]
        labels_without_O = Y_gold[filter_Y]

        predictions = logits_clean.argmax(dim=1)
        acc = (predictions == label_clean).float().mean()

        try:
            predictions_without_O = logits_without_O.argmax(dim=1)
        except RuntimeError:
            predictions_without_O = 1.0
        acc_without_O = (predictions_without_O == labels_without_O).float().mean()
        if torch.isnan(acc_without_O):
            acc_without_O = 0
        #print("acc: {}".format(acc))
        #print("acc_without_O: {}".format(acc_without_O))
        return acc, acc_without_O

def count_correct(model_obj, logits, Y_gold):
    logits_clean = logits[Y_gold != -100]
    label_clean = Y_gold[Y_gold != -100]

    predictions = logits_clean.argmax(dim=1)

    correct = predictions == label_clean
    num_correct = correct.sum()
    return num_correct, len(predictions)



def count_correct_without_O(model_obj, logits, Y_gold):
    filter_Y = torch.logical_and(Y_gold != -100, Y_gold != model_obj['token_encoder'].label_encoder.encode_label('O'))
    logits_without_O = logits[filter_Y]
    labels_without_O = Y_gold[filter_Y]

    predictions = logits_without_O.argmax(dim=1)

    correct = predictions == labels_without_O
    num_correct = correct.sum()
    return num_correct, len(predictions)



def train_loop(train_data,
               model_obj,
               device=torch.device('cpu'),
               val=None):

    model = model_obj['model']
    model = model.to(device)

    optimizer = model_obj['optim']
    token_encoder = model_obj['token_encoder']
    if 'scheduler' in model_obj:
        scheduler = model_obj['scheduler']
    else:
        scheduler = None
    tot_loss = 0
    #tot_acc = 0
    #tot_acc_without_O = 0
    tot_pred = 0
    tot_correct = 0
    tot_pred_without_O = 0
    tot_correct_without_O = 0
    for batch in tqdm(train_data):
        model.train()
        model.zero_grad()

        encoded_input = token_encoder.encode_with_labels(batch)
        encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
        encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
        #print("encoded input: {}".format(encoded_input))
        if 'labels' in encoded_input:
            encoded_input['labels'] = encoded_input['labels'].to(device)


def train_loop(train_data,
               model_obj,
               device=torch.device('cpu'),
               val=None):

    model = model_obj['model']
    model = model.to(device)

    optimizer = model_obj['optim']
    token_encoder = model_obj['token_encoder']
    if 'scheduler' in model_obj:
        scheduler = model_obj['scheduler']
    else:
        scheduler = None
    tot_loss = 0
    #tot_acc = 0
    #tot_acc_without_O = 0
    tot_pred = 0
    tot_correct = 0
    tot_pred_without_O = 0
    tot_correct_without_O = 0
    for batch in tqdm(train_data):
        model.train()
        model.zero_grad()

        encoded_input = token_encoder.encode_with_labels(batch)
        encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
        encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
        #print("encoded input: {}".format(encoded_input))
        if 'labels' in encoded_input:
            encoded_input['labels'] = encoded_input['labels'].to(device)

        loss, logits = model(**encoded_input, return_dict=False)

        Y_gold = encoded_input['labels']
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        num_correct, num_pred = count_correct(model_obj, logits, Y_gold)
        tot_pred += num_pred
        tot_correct += num_correct
        num_correct_without_O, num_pred_without_O = count_correct_without_O(model_obj, logits, Y_gold)
        tot_pred_without_O += num_pred_without_O
        tot_correct_without_O += num_correct_without_O

        #acc, acc_without_O = calculate_acc(model_obj, logits, Y_gold)
        #tot_acc += acc
        #tot_acc_without_O += acc_without_O

        tot_loss += loss.item()
        avg_loss = tot_loss / len(train_data)

    logging.info("Total loss: %f", tot_loss)
    logging.info("Average loss: %f", avg_loss)
    #logging.info("Acc: %f", tot_acc / len(train_data))
    logging.info("Acc: %f", tot_correct / tot_pred)
    #logging.info("Acc_without_O: %f", tot_acc_without_O / len(train_data))
    logging.info("Acc_without_O: %f", tot_correct_without_O / tot_pred_without_O)


    f1 = None
    if val is not None:
        res = eval(val, model_obj, device=device)
        #strict = res['strict']
        logging.info("f1: %f", res['tagtype_f1_score'])
        #print(res)
        print(res['tagtype_report'])
        f1 = res['tagtype_f1_score']
        print("Validation - Acc: {}.".format(res['acc']))
        print("Validation - Acc_without_O: {}".format(res['acc_without_O']))

    return avg_loss, f1


def eval(data_handler, model_obj, device=torch.device('cpu')):
    model = model_obj['model']
    model = model.to(device)

    token_encoder = model_obj['token_encoder']
    model.eval()

    predicted = []
    gold = []

    logging.info("Evaluating on val data...")

    #tot_acc = 0
    #tot_acc_without_O = 0
    tot_pred = 0
    tot_correct = 0
    tot_pred_without_O = 0
    tot_correct_without_O = 0

    sent_with_predictions = []
    for batch in tqdm(data_handler):
        #data = batch[0]
        #sent, tags = data
        encoded_input = token_encoder.encode_with_labels(batch)
        #encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
        #encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
        #if 'labels' in encoded_input:
        #    encoded_input['labels'] = encoded_input['labels'].to(device)
        for key in encoded_input.keys():
            encoded_input[key] = encoded_input[key].to(device)
        #print(sent)
        #print(tags)
        #exit()
        with torch.no_grad():
            #encoded_input = token_encoder.encode_with_labels(batch)
            #print(encoded_input)
            #exit()
            _, logits = model(**encoded_input, return_dict=False)  # model should only return logits if eval
            #logits = logits.squeeze()
            Y_gold = encoded_input['labels']

            # not working
            #acc, acc_without_O = calculate_acc(model_obj, logits, Y_gold)
            #tot_acc += acc
            #tot_acc_without_O += acc_without_O
            #print("logits: {}".format(logits))

            pred_idxs = logits.argmax(dim=2)
            #pred_idxs [ batch_size, sent_len + word_pieces ]
            #print("pred_idxs: {}".format(pred_idxs))
            pred_labels = token_encoder.label_encoder.decode_labels(pred_idxs, flatten=True)
            #print("pred_labels: {}".format(pred_labels))
            #exit()
            gold_labels = token_encoder.label_encoder.decode_labels(Y_gold, flatten=True)
            #print("gold_labels: {}".format(gold_labels))
            #gold.append(gold_labels)
            #predicted.append(pred_labels)
            for g, p in zip(gold_labels, pred_labels):
                if g not in ['SPEC']:
                    gold.append(g)
                    predicted.append(p)

            #gold += gold_labels
            #predicted += pred_labels

            encoded_ids = encoded_input['input_ids']
            #decoded_sent = token_encoder.decode(encoded_ids)#.squeeze(0))
            decoded_sents = [token_encoder.decode(line) for line in encoded_ids]
            decoded_sequence = []
            for token in decoded_sents:
                decoded_sequence += token

            #print(decoded_sent)
            #print("len(gold_labels): {}, len(pred_labels): {}, len(sent): {}".format(len(gold_labels), len(pred_labels), len(decoded_sent)))

            num_correct, num_pred = count_correct(model_obj, logits, Y_gold)
            tot_correct += num_correct
            tot_pred += num_pred

            num_correct_without_O, num_pred_without_O = count_correct_without_O(model_obj, logits, Y_gold)
            tot_pred_without_O += num_pred_without_O
            tot_correct_without_O += num_correct_without_O

            filter_Y = torch.logical_and(Y_gold != -100, Y_gold != model_obj['token_encoder'].label_encoder.encode_label('O'))
            logits_without_O = logits[filter_Y]
            labels_without_O = Y_gold[filter_Y]

            predictions_without_O = logits_without_O.argmax(dim=1)
            correct_without_O = predictions_without_O == labels_without_O
            num_correct_without_O = correct_without_O.sum()
            #assert len(gold_labels) == len(pred_labels) == len(decoded_sent)

        #sent_with_predictions.append({"sent" : sent, "pred": pred_labels})
        sent_with_predictions.append(list(zip(decoded_sequence, pred_labels, gold_labels)))

    tagset = copy.deepcopy(token_encoder.label_encoder.labels)
    tagset.remove("O")
    #tagtype_report = flat_classification_report(gold, predicted, labels=tagset)
    #tagtype_f1_score = flat_f1_score(gold, predicted, labels=tagset, average='micro')
    tagtype_report = classification_report(gold, predicted, labels=tagset)
    tagtype_report_dict = classification_report(gold, predicted, labels=tagset, output_dict=True)
    tagtype_f1_score = f1_score(gold, predicted, labels=tagset, average='micro')



    #ner_evaluator = Evaluator(gold, predicted, token_encoder.label_encoder.labels)
    #res, res_agg = ner_evaluator.evaluate()
    ret_dict = {#'res' : res,
                #'res_agg' : res_agg,
                'tagtype_report' : tagtype_report,
                'tagtype_report_dict' : tagtype_report_dict,
                'tagtype_f1_score' : tagtype_f1_score,
                'acc' : tot_correct / tot_pred,
                'acc_without_O' : tot_correct_without_O / tot_pred_without_O }
    if 'action' in model_obj and model_obj['action'] == 'eval':
        ret_dict['output'] = sent_with_predictions
    return ret_dict
