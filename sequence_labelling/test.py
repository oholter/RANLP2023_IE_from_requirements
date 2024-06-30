import logging
import torch
import json
from torch.optim import Adam, AdamW, SGD
from argparse import ArgumentParser
from pathlib import Path
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch import nn
import matplotlib.pyplot as plt

from concat_ner.model import RobertaClassifier, DomainSpecificClassifier, CombinedClassifier, FlairClassifier
from concat_ner.data_handler import DataHandler
from concat_ner.train import train_loop, eval
from concat_ner.token_encoder import RobertaTokenEncoder, ClassicTokenEncoder, CombinedEncoder, FlairTokenEncoder
from concat_ner.label_encoder import LabelEncoder


LABELS = ["SCOPE", "CONDITION", "DEMAND", "O"]
#LABELS = ["O", "DEMAND", "CONDITION", "SCOPE"]
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

SEED = 42


def create_latex_report(eval_d, prefix=None):
    if not prefix:
        prefix = ""

    d = eval_d['tagtype_report_dict']
    s = ""
    s += "\\newcommand*\\{}ScopeP{{{}}}".format(prefix,d['SCOPE']['precision'])
    s += "\\newcommand*\\{}ScopeR{{{}}}".format(prefix,d['SCOPE']['recall'])
    s += "\\newcommand*\\{}ScopeF{{{}}}".format(prefix,d['SCOPE']['f1-score'])
    s += "\\newcommand*\\{}ScopeSup{{{}}}".format(prefix,d['SCOPE']['support'])
    s += "\\newcommand*\\{}CondP{{{}}}".format(prefix,d['CONDITION']['precision'])
    s += "\\newcommand*\\{}CondR{{{}}}".format(prefix,d['CONDITION']['recall'])
    s += "\\newcommand*\\{}CondF{{{}}}".format(prefix,d['CONDITION']['f1-score'])
    s += "\\newcommand*\\{}CondSup{{{}}}".format(prefix,d['CONDITION']['support'])
    s += "\\newcommand*\\{}DemandP{{{}}}".format(prefix,d['DEMAND']['precision'])
    s += "\\newcommand*\\{}DemandR{{{}}}".format(prefix,d['DEMAND']['recall'])
    s += "\\newcommand*\\{}DemandF{{{}}}".format(prefix,d['DEMAND']['f1-score'])
    s += "\\newcommand*\\{}DemandSup{{{}}}".format(prefix,d['DEMAND']['support'])
    s += "\\newcommand*\\{}MicAvgP{{{}}}".format(prefix,d['micro avg']['precision'])
    s += "\\newcommand*\\{}MicAvgR{{{}}}".format(prefix,d['micro avg']['recall'])
    s += "\\newcommand*\\{}MicAvgF{{{}}}".format(prefix,d['micro avg']['f1-score'])
    s += "\\newcommand*\\{}TotSup{{{}}}".format(prefix,d['micro avg']['support'])
    s += "\\newcommand*\\{}MacAvgP{{{}}}".format(prefix,d['macro avg']['precision'])
    s += "\\newcommand*\\{}MacAvgR{{{}}}".format(prefix,d['macro avg']['recall'])
    s += "\\newcommand*\\{}MacAvgF{{{}}}".format(prefix,d['macro avg']['f1-score'])
    s += "\\newcommand*\\{}WeiAvgP{{{}}}".format(prefix,d['weighted avg']['precision'])
    s += "\\newcommand*\\{}WeiAvgR{{{}}}".format(prefix,d['weighted avg']['recall'])
    s += "\\newcommand*\\{}WeiAvgF{{{}}}".format(prefix,d['weighted avg']['f1-score'])
    s += "\n"
    print(s)
    return s



def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("test", help="file with test data")
    parser.add_argument("--prefix", help="prefix to variables")
    parser.add_argument("--save", help="paths to save output of the model")
    parser.add_argument("--latex", help="path to save latex file")
    parser.add_argument("--result", "-r", help="path to save all the results as a .json file")

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
    model_obj = {'model' : torch.load(args.model_path, map_location=device),
                 "action": "eval"}

    # INITIALIZE MODEL
    model_obj['token_encoder'] = RobertaTokenEncoder(LabelEncoder(LABELS))

    # run test cycle
    test_handler = DataHandler(args.test, batch_size=8)
    test_data = test_handler.get_all_batches()
    eval_dict = eval(test_data, model_obj, device=device)
    print(eval_dict['tagtype_report'])
    print("acc: {}".format(eval_dict['acc']))
    print("acc without O: {}".format(eval_dict['acc_without_O']))
    output = eval_dict['output']
    if args.save:
        with open(args.save, 'w') as out:
            out.write("token\tpred\tgold\n")
            for doc in output:
                for token, tag, gold in doc:
                    out.write("{}\t{}\t{}\n".format(token, tag, gold))
                out.write("\n\n")
        print("Written output data to {}".format(args.save))
    if args.latex:
        s = create_latex_report(eval_dict, prefix=args.prefix)
        with open(args.latex, 'w') as out:
            out.write(s)
    if args.result:
        with open(args.result, 'w') as F:
            F.write(json.dumps(eval_dict['tagtype_report_dict'], indent=4))
            logging.info("Written scores to %s", args.result)

if __name__ == '__main__':
    main()
