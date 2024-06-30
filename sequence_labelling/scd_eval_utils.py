import re
import torch
import spacy
from rouge import Rouge
from jiwer import wer
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk import word_tokenize
from sequence_labelling.XlmrCosineSim import XlmrCosineSim
from bert_score import score


ROBERTA_TOKENS = ["<s>", "</s>"]
nlp = spacy.load("en_core_web_sm")
BERT_TOKENS = ["[CLS]", "[SEP]"]


def roberta_tok_to_tok(tok):
    if re.search("Ġ", tok):
        tok = re.sub("Ġ", "", tok)
    return tok

def output_to_scopes(output):
    return output_to_scd(output, label="SCOPE")

def normalize_demands(demands):
    demand_texts = []

    for demand in demands:
        doc = nlp(demand)
        text = " ".join([token.text.lower() for token in doc])

        # post processing of the compiled string
        if re.search("\\s-\\s", text): #space around -
            text = re.sub("\\s-\\s", "-", text)
        if re.search("([\\[\\{\\(])\\s+", text): #space after [{(
            new_text = re.sub("([\\[\\{\\(])\\s+", "\\1", text)
            text = new_text
        if re.search("\\s+([\\]\\}\\)])", text): #space before ]})
            new_text = re.sub("\\s+([\\]\\}\\)])", "\\1", text)
            text = new_text
        if re.search("\\s+[.:;]", text):  # space before punctuation
            new_text = re.sub("\\s+([.:;])", "\\1", text)
            text = new_text
        if re.search("[—-]\\s+", text):  # space after item
            text = re.sub("([—-])\\s+", "\\1", text)
        if re.search(",", text): #cannot have commas in demand
            text = re.sub(",", "", text)
        if re.search("\\s\\s+", text): #extra spaces
            text = re.sub("\\s+", " ", text)
        if re.search("\\((?!.*\\))", text): #remove unmatched parenthesis
            text = re.sub("\\(.*", "", text)
        if re.search("\\.", text): # remove .
            text = re.sub("\\.", "", text)
        if re.search("\\(.\\s?\\)", text): # remove vessel(s )
            text = re.sub("\\(.\\s?\\)", "", text)

        if text.strip():
            demand_texts.append(text)

    demand_text_set = set(demand_texts)
    return demand_text_set

def output_to_demands(output):
    return output_to_scd(output, label="DEMAND")

def eval_demands(pred, gold):
    return eval_cd(pred, gold)

def normalize_conditions(conditions):
    condition_texts = []
    for condition in conditions:
        doc = nlp(condition)
        text = " ".join(token.text.lower() for token in doc)

        # post processing of the compiled string
        if re.search("\\s-\\s", text): #space around -
            text = re.sub("\\s-\\s", "-", text)
        if re.search("([\\[\\{\\(])\\s+", text): #space after [{(
            new_text = re.sub("([\\[\\{\\(])\\s+", "\\1", text)
            text = new_text
        if re.search("\\s+([\\]\\}\\)])", text): #space before ]})
            new_text = re.sub("\\s+([\\]\\}\\)])", "\\1", text)
            text = new_text
        if re.search("\\s+[.:;]", text):  # space before punctuation
            new_text = re.sub("\\s+([.:;])", "\\1", text)
            text = new_text
        if re.search("[—-]\\s+", text):  # space after item
            text = re.sub("([—-])\\s+", "\\1", text)
        if re.search(",", text): #cannot have commas
            text = re.sub(",", "", text)
        if re.search("\\s\\s+", text): #extra spaces
            text = re.sub("\\s+", " ", text)
        if re.search("\\((?!.*\\))", text): #remove unmatched parenthesis
            text = re.sub("\\(.*", "", text)
        if re.search("\\.", text): # remove .
            text = re.sub("\\.", "", text)
        if re.search("\\(.\\s?\\)", text): # remove vessel(s )
            text = re.sub("\\(.\\s?\\)", "", text)

        if text.strip(): # do not append empty strings
            condition_texts.append(text)

    condition_text_set = set(condition_texts)
    return condition_text_set

def output_to_conditions(output):
    return output_to_scd(output, label="CONDITION")

def eval_conditions(pred, gold):
    return eval_cd(pred, gold)



def normalize_scopes(scopes):
    """normalizes a set of scopes from one document"""
    scope_texts = []
    for scope in scopes:
        s_doc = nlp(scope)
        lemmatized_toks = [token.text.lower() for token in s_doc]
        if lemmatized_toks[0].lower() == "the":
            lemmatized_toks = lemmatized_toks[1:]
        lemmatized_toks[-1] = s_doc[-1].lemma_.lower()
        lemmatized = " ".join(lemmatized_toks)

        # post processing of the compiled string
        if re.search("\\s-\\s", lemmatized): #space around -
            lemmatized = re.sub("\\s-\\s", "-", lemmatized)
        if re.search(",", lemmatized): #cannot have commas in scope
            lemmatized = re.sub(",", "", lemmatized)
        if re.search("\\s\\s+", lemmatized): #extra spaces
            lemmatized = re.sub("\\s+", " ", lemmatized)
        if re.search("\\((?!.*\\))", lemmatized): #remove unmatched parenthesis
            lemmatized = re.sub("\\(.*", "", lemmatized)
        if re.search("\\.", lemmatized): # remove .
            lemmatized = re.sub("\\.", "", lemmatized)
        if re.search("\\(.\\s?\\)", lemmatized): # remove vessel(s )
            lemmatized = re.sub("\\(.\\s?\\)", "", lemmatized)


        #print("lemmatized: {}".format(lemmatized))
        if lemmatized.strip():
            scope_texts.append(lemmatized)

    scope_text_set = set(scope_texts)
    return scope_text_set


# todo: change variable names
def output_to_scd(output, label):
    """ Version for BERT!!

    this outputs the tokens that are of one label,
    it divides them into a list of strings"""
    doc_scopes = []
    for doc in output:
        scopes = []
        scope_string = ""
        i = 0
        #for i, (tok, tag, gold) in enumerate(doc):
        #print(doc)
        while i < len(doc):
            tok, tag, gold = doc[i]
            tags = [tag]
            j = i + 1
            if tok not in BERT_TOKENS: # not a bert specific token
                if not re.search("##", tok): # a part

                    # append any word parts
                    while j < len(doc):
                        if not re.search("##", doc[j][0]): # next tok is not a part
                            #print("not part tok: {}".format(tok))
                            break
                        else: # add part to token
                            #print("tok: {}".format(tok))
                            next_tok = doc[j][0]
                            #print("next tok before: {}".format(next_tok))
                            next_tok = re.sub("##", "", next_tok)
                            #print("next tok after: {}".format(next_tok))
                            tok += next_tok
                            #print("updated token: {}".format(tok))
                            tags.append(doc[j][1])
                            j += 1
                else:
                    print("Something wrong, should not happen")
                    print("tok: {}".format(tok))
                    print("tags: {}".format(tags))
                    print("i: {}".format(i))
                    exit()

                if label.upper() in tags:
                    #print("scope in: {}".format(tok))
                    #if i > 0 and doc[i-1][1] != "SCOPE" and scope_string:  # new span
                    if i > 0 and doc[i-1][1] != label.upper() and scope_string:  # new span
                        #print("breaking: {}".format(tok))
                        scopes.append(scope_string)
                        scope_string = tok
                    else:
                        #if gold == 'SPEC' and tok not in ROBERTA_TOKENS:
                            #scope_string += tok
                        if scope_string:
                            scope_string += " " + tok
                        else:
                            scope_string = tok

            i = j
        if scope_string: # adding the last one (if any)
            #print(scope_string)
            scopes.append(scope_string)

        doc_scopes.append(scopes)

    return doc_scopes


# todo: change variable names
def output_to_scd_roberta(output, label):
    """
    Version for RoBERTa!!

    this outputs the tokens that are of one label,
    it divides them into a list of strings"""
    doc_scopes = []
    for doc in output:
        scopes = []
        scope_string = ""
        i = 0
        #for i, (tok, tag, gold) in enumerate(doc):
        while i < len(doc):
            tok, tag, gold = doc[i]
            tags = [tag]
            j = i + 1
            if tok not in ROBERTA_TOKENS: # not a roberta specific token
                if re.search("Ġ", tok):
                    tok = re.sub("Ġ", "", tok)

                    # append any word parts
                    while j < len(doc):
                        if re.search("Ġ", doc[j][0]): # not a part
                            break
                        else: # add part to token
                            tok += doc[j][0]
                            tags.append(doc[j][1])
                            j += 1
                else:
                    print("Something wrong, should not happen")
                    print("tok: {}".format(tok))
                    print("tags: {}".format(tags))
                    print("i: {}".format(i))
                    exit()

                if label.upper() in tags:
                    #print("scope in: {}".format(tok))
                    #if i > 0 and doc[i-1][1] != "SCOPE" and scope_string:  # new span
                    if i > 0 and doc[i-1][1] != label.upper() and scope_string:  # new span
                        #print("breaking: {}".format(tok))
                        scopes.append(scope_string)
                        scope_string = tok
                    else:
                        #if gold == 'SPEC' and tok not in ROBERTA_TOKENS:
                            #scope_string += tok
                        if scope_string:
                            scope_string += " " + tok
                        else:
                            scope_string = tok

            i = j
        if scope_string: # adding the last one (if any)
            scopes.append(scope_string)

        doc_scopes.append(scopes)

    return doc_scopes


def b_score(preds, golds):
    """
    @param preds/golds: list of texts

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p, r, f = score(preds,
                    golds,
                    lang="en",
                    batch_size=16,
                    verbose=False,
                    device=device)

    return p, r, f


def eval_cd(pred, gold):
    #print("pred:\n{}".format(pred))
    #print("gold:\n{}".format(gold))
    rouge = Rouge()
    tp = fp = fn = 0
    c = XlmrCosineSim(bs=8)  # condition/demands are long, need smaller batch size on NREC

    r_scores = []
    tot_bleu = 0
    met = 0
    tot_word_error_rate = 0
    tot_xml = 0
    tot_bert_score_p = 0
    tot_bert_score_r = 0
    tot_bert_score_f = 0
    all_scores = {
            "pred": [],
            "gold": [],
            "rouge_1-p": [],
            "rouge_1-r": [],
            "rouge_1-f": [],
            "rouge_l-p": [],
            "rouge_l-r": [],
            "rouge_l-f": [],
            "bleu": [],
            "meteor": [],
            "wer": [],
            "wacc": [],
            "xml-cosine": [],
            "bert-score-p": [],
            "bert-score-r": [],
            "bert-score-f": []}

    # batching what can be batched
    # - rouge
    # - xlm cosine sim
    # - bertscore

    text_preds = [" and ".join(p) if p else "EMPTY" for p in pred]
    text_golds = [" and ".join(g) if g else "EMPTY" for g in gold]

    #print("len(text_preds): {}".format(len(text_preds)))
    #print("len(text_golds): {}".format(len(text_golds)))

    # debugging empty hypothesis
    #print("Printing hypothesises:")
    #for i, h in enumerate(text_preds):
        #print("Pred {}: {}".format(i, h))

    #print("text_preds:\n{}".format(text_preds))
    #print("text_golds:\n{}".format(text_golds))
    r_scores = rouge.get_scores(text_preds, text_golds)
    xml_scores = c(text_golds, text_preds)
    bert_p, bert_r, bert_f = b_score(text_preds, text_golds)


    # now, what cannot be batched
    for p, g in zip(pred, gold):
        p_set = set(p)
        g_set = set(g)
        tp += len(p_set.intersection(g_set))
        fp += len(p_set.difference(g_set))
        fn += len(g_set.difference(p_set))

        # create one string of possibly multiple spans
        p_text = " and ".join(p)
        g_text = " and ".join(g)

        if not p_text:
            p_text = "EMPTY"
        if not g_text:
            g_text = "EMPTY"

        #r_score = rouge.get_scores(p_text, g_text)[0]
        #r_scores.append(r_score)
        p_tok = word_tokenize(p_text)
        g_tok = word_tokenize(g_text)
        met = single_meteor_score(g_tok, p_tok)

        bleu = sentence_bleu([g_tok], p_tok, weights=(1, 0, 0, 0))
        tot_bleu += bleu
        word_error_rate = wer(g_text, p_text)
        tot_word_error_rate += word_error_rate
        #xml_score = c([g_text], [p_text])[0]
        #tot_xml += xml_score

        #[bert_score_P], [bert_score_R], [bert_score_F] = score([p_text], [g_text], lang="en", verbose=False)
        #bert_score_P, bert_score_R, bert_score_F = b_score(p_text, g_text)
        #tot_bert_score_p += bert_score_P
        #tot_bert_score_r += bert_score_R
        #tot_bert_score_f += bert_score_F

        all_scores['pred'].append(p_text)
        all_scores['gold'].append(g_text)
        #all_scores['rouge_1-p'].append(r_score['rouge-1']['p'])
        #all_scores['rouge_1-r'].append(r_score['rouge-1']['r'])
        #all_scores['rouge_1-f'].append(r_score['rouge-1']['f'])
        #all_scores['rouge_l-p'].append(r_score['rouge-l']['p'])
        #all_scores['rouge_l-r'].append(r_score['rouge-l']['r'])
        #all_scores['rouge_l-f'].append(r_score['rouge-l']['f'])
        all_scores['bleu'].append(bleu)
        all_scores['meteor'].append(met)
        all_scores['wer'].append(word_error_rate)
        all_scores['wacc'].append(1 - word_error_rate)
        #all_scores['xml-cosine'].append(xml_score)
        #all_scores['bert-score-p'].append(bert_score_P)
        #all_scores['bert-score-r'].append(bert_score_R)
        #all_scores['bert-score-f'].append(bert_score_F)

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    r1_p_ave = 0
    r1_r_ave = 0
    r1_f_ave = 0
    rl_p_ave = 0
    rl_r_ave = 0
    rl_f_ave = 0
    for s in r_scores:
        all_scores['rouge_1-p'].append(s['rouge-1']['p'])
        all_scores['rouge_1-r'].append(s['rouge-1']['r'])
        all_scores['rouge_1-f'].append(s['rouge-1']['f'])
        all_scores['rouge_l-p'].append(s['rouge-l']['p'])
        all_scores['rouge_l-r'].append(s['rouge-l']['r'])
        all_scores['rouge_l-f'].append(s['rouge-l']['f'])
        r1_p_ave += s['rouge-1']['p']
        r1_r_ave += s['rouge-1']['r']
        r1_f_ave += s['rouge-1']['f']
        rl_p_ave += s['rouge-l']['p']
        rl_r_ave += s['rouge-l']['r']
        rl_f_ave += s['rouge-l']['f']
    r1_p_ave /= len(r_scores)
    r1_r_ave /= len(r_scores)
    r1_f_ave /= len(r_scores)
    rl_p_ave /= len(r_scores)
    rl_r_ave /= len(r_scores)
    rl_f_ave /= len(r_scores)

    bert_score_p_ave = 0
    bert_score_r_ave = 0
    bert_score_f_ave = 0
    for p, r, f in zip(bert_p, bert_r, bert_f):
        all_scores['bert-score-p'].append(p.item())
        all_scores['bert-score-r'].append(r.item())
        all_scores['bert-score-f'].append(f.item())
        bert_score_p_ave += p
        bert_score_r_ave += r
        bert_score_f_ave += f
    bert_score_p_ave /= len(bert_p)
    bert_score_r_ave /= len(bert_r)
    bert_score_f_ave /= len(bert_f)

    all_scores['xml-cosine'] = xml_scores
    tot_xml = 0
    for x in xml_scores:
        tot_xml += x
    tot_xml /= len(xml_scores)

    met /= len(pred)
    tot_bleu /= len(pred)
    word_error_rate = tot_word_error_rate / len(pred)

    return {"precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge_1-p": r1_p_ave,
            "rouge_1-r": r1_r_ave,
            "rouge_1-f": r1_f_ave,
            "rouge_l-p": rl_p_ave,
            "rouge_l-r": rl_r_ave,
            "rouge_l-f": rl_f_ave,
            "bleu-1": tot_bleu,
            "meteor": met,
            "wer": word_error_rate,
            "xml-cosine": tot_xml,
            "bert-score-p": bert_score_p_ave.item(),
            "bert-score-r": bert_score_r_ave.item(),
            "bert-score-f": bert_score_f_ave.item()}, all_scores


def eval_scopes(pred, gold):
    rouge = Rouge()
    tp = fp = fn = 0
    c = XlmrCosineSim()

    r_scores = []
    tot_bleu = 0
    met = 0
    tot_word_error_rate = 0
    tot_xml = 0
    tot_bert_score_p = 0
    tot_bert_score_r = 0
    tot_bert_score_f = 0
    all_scores = {
            "pred": [],
            "gold": [],
            "rouge_1-p": [],
            "rouge_1-r": [],
            "rouge_1-f": [],
            "rouge_l-p": [],
            "rouge_l-r": [],
            "rouge_l-f": [],
            "bleu": [],
            "meteor": [],
            "wer": [],
            "wacc": [],
            "xml-cosine": [],
            "bert-score-p": [],
            "bert-score-r": [],
            "bert-score-f": []}


    # batching what can be batched
    # - rouge
    # - xlm cosine sim
    # - bertscore

    text_preds = [" and ".join(p) if p else "EMPTY" for p in pred]
    text_golds = [" and ".join(g) if g else "EMPTY" for g in gold]

    r_scores = rouge.get_scores(text_preds, text_golds)
    xml_scores = c(text_golds, text_preds)
    bert_p, bert_r, bert_f = b_score(text_preds, text_golds)


    # getting what can not be batched
    for p, g in zip(pred, gold):
        p_set = set(p)
        g_set = set(g)
        tp += len(p_set.intersection(g_set))
        fp += len(p_set.difference(g_set))
        fn += len(g_set.difference(p_set))

        # create one string of possibly multiple spans
        p_text = " and ".join(p)
        g_text = " and ".join(g)

        if not p_text:
            p_text = "EMPTY"
        if not g_text:
            g_text = "EMPTY"

        #r_score = rouge.get_scores(p_text, g_text)[0]
        #r_scores.append(r_score)
        p_tok = word_tokenize(p_text)
        g_tok = word_tokenize(g_text)
        met = single_meteor_score(g_tok, p_tok)

        bleu = sentence_bleu([g_tok], p_tok, weights=(1, 0, 0, 0))
        tot_bleu += bleu
        word_error_rate = wer(g_text, p_text)
        tot_word_error_rate += word_error_rate
        #xml_score = c([g_text], [p_text])[0]
        #tot_xml += xml_score

        #[bert_score_P], [bert_score_R], [bert_score_F] = score([p_text], [g_text], lang="en", verbose=False)
        #tot_bert_score_p += bert_score_P
        #tot_bert_score_r += bert_score_R
        #tot_bert_score_f += bert_score_F

        all_scores['pred'].append(p_text)
        all_scores['gold'].append(g_text)
        #all_scores['rouge_1-p'].append(r_score['rouge-1']['p'])
        #all_scores['rouge_1-r'].append(r_score['rouge-1']['r'])
        #all_scores['rouge_1-f'].append(r_score['rouge-1']['f'])
        #all_scores['rouge_l-p'].append(r_score['rouge-l']['p'])
        #all_scores['rouge_l-r'].append(r_score['rouge-l']['r'])
        #all_scores['rouge_l-f'].append(r_score['rouge-l']['f'])
        all_scores['bleu'].append(bleu)
        all_scores['meteor'].append(met)
        all_scores['wer'].append(word_error_rate)
        all_scores['wacc'].append(1 - word_error_rate)
        #all_scores['xml-cosine'].append(xml_score)
        #all_scores['bert-score-p'].append(bert_score_P)
        #all_scores['bert-score-r'].append(bert_score_R)
        #all_scores['bert-score-f'].append(bert_score_F)

    # calculating totals and averages
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    r1_p_ave = 0
    r1_r_ave = 0
    r1_f_ave = 0
    rl_p_ave = 0
    rl_r_ave = 0
    rl_f_ave = 0
    for s in r_scores:
        all_scores['rouge_1-p'].append(s['rouge-1']['p'])
        all_scores['rouge_1-r'].append(s['rouge-1']['r'])
        all_scores['rouge_1-f'].append(s['rouge-1']['f'])
        all_scores['rouge_l-p'].append(s['rouge-l']['p'])
        all_scores['rouge_l-r'].append(s['rouge-l']['r'])
        all_scores['rouge_l-f'].append(s['rouge-l']['f'])
        r1_p_ave += s['rouge-1']['p']
        r1_r_ave += s['rouge-1']['r']
        r1_f_ave += s['rouge-1']['f']
        rl_p_ave += s['rouge-l']['p']
        rl_r_ave += s['rouge-l']['r']
        rl_f_ave += s['rouge-l']['f']
    r1_p_ave /= len(r_scores)
    r1_r_ave /= len(r_scores)
    r1_f_ave /= len(r_scores)
    rl_p_ave /= len(r_scores)
    rl_r_ave /= len(r_scores)
    rl_f_ave /= len(r_scores)

    bert_score_p_ave = 0
    bert_score_r_ave = 0
    bert_score_f_ave = 0
    for p, r, f in zip(bert_p, bert_r, bert_f):
        all_scores['bert-score-p'].append(p.item())
        all_scores['bert-score-r'].append(r.item())
        all_scores['bert-score-f'].append(f.item())
        bert_score_p_ave += p
        bert_score_r_ave += r
        bert_score_f_ave += f
    bert_score_p_ave /= len(bert_p)
    bert_score_r_ave /= len(bert_r)
    bert_score_f_ave /= len(bert_f)

    all_scores['xml-cosine'] = xml_scores
    tot_xml = 0
    for x in xml_scores:
        tot_xml += x
    tot_xml /= len(xml_scores)

    met /= len(pred)
    tot_bleu /= len(pred)
    word_error_rate = tot_word_error_rate / len(pred)

    return {"precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge_1-p": r1_p_ave,
            "rouge_1-r": r1_r_ave,
            "rouge_1-f": r1_f_ave,
            "rouge_l-p": rl_p_ave,
            "rouge_l-r": rl_r_ave,
            "rouge_l-f": rl_f_ave,
            "bleu-1": tot_bleu,
            "meteor": met,
            "wer": word_error_rate,
            "xml-cosine": tot_xml,
            "bert-score-p": bert_score_p_ave.item(),
            "bert-score-r": bert_score_r_ave.item(),
            "bert-score-f": bert_score_f_ave.item()}, all_scores


def save_output(all_scores, path):
    with open(path, 'w') as out:
        out.write("idx\tpred\tgold\trouge-lf\tbleu\twacc\tLMB\tBert-f\n")
        for i in range(len(all_scores['pred'])):
            #out.write("{}\t{}\n".format(list(p), g))
            out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i,
                                          all_scores['pred'][i],
                                          all_scores['gold'][i],
                                          all_scores['rouge_l-f'][i],
                                          all_scores['bleu'][i],
                                          all_scores['wacc'][i],
                                          all_scores['xml-cosine'][i],
                                          all_scores['bert-score-f']))
        out.write("\n\n")
    print("Written output data to {}".format(path))




def create_latex_report(d, path, prefix=None):
    if not prefix:
        prefix = ""

    s = ""
    s += "\\newcommand*\\{}ExactP{{{}}}".format(prefix,d['precision'])
    s += "\\newcommand*\\{}ExactR{{{}}}".format(prefix,d['recall'])
    s += "\\newcommand*\\{}ExactF{{{}}}".format(prefix,d['f1'])
    s += "\\newcommand*\\{}RougeOnep{{{}}}".format(prefix,d['rouge_1-p'])
    s += "\\newcommand*\\{}RougeOner{{{}}}".format(prefix,d['rouge_1-r'])
    s += "\\newcommand*\\{}RougeOnef{{{}}}".format(prefix,d['rouge_1-f'])
    s += "\\newcommand*\\{}Rougelp{{{}}}".format(prefix,d['rouge_l-p'])
    s += "\\newcommand*\\{}Rougelr{{{}}}".format(prefix,d['rouge_l-r'])
    s += "\\newcommand*\\{}Rougelf{{{}}}".format(prefix,d['rouge_l-f'])
    s += "\\newcommand*\\{}Bleu{{{}}}".format(prefix,d['bleu-1'])
    s += "\\newcommand*\\{}Meteor{{{}}}".format(prefix,d['meteor'])
    s += "\\newcommand*\\{}Wer{{{}}}".format(prefix,d['wer'])
    s += "\\newcommand*\\{}Wacc{{{}}}".format(prefix,1 - d['wer'])
    s += "\\newcommand*\\{}XmlCosine{{{}}}".format(prefix,d['xml-cosine'])
    s += "\\newcommand*\\{}BertScorep{{{}}}".format(prefix,d['bert-score-p'])
    s += "\\newcommand*\\{}BertScorer{{{}}}".format(prefix,d['bert-score-r'])
    s += "\\newcommand*\\{}BertScoref{{{}}}".format(prefix,d['bert-score-f'])
    s += "\n"
    print(s)

    with open(path, 'w') as F:
        F.write(s)
        print("written latex variables to {}".format(path))
