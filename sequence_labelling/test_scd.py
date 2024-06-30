import logging
import json
import torch
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

from sequence_labelling.model import RobertaClassifier, DomainSpecificClassifier, CombinedClassifier, FlairClassifier
from sequence_labelling.token_encoder import RobertaTokenEncoder, ClassicTokenEncoder, CombinedEncoder, FlairTokenEncoder
from sequence_labelling.train import eval
from sequence_labelling.label_encoder import LabelEncoder
from sequence_labelling.data_handler import DataHandler

from sequence_labelling.scd_eval_utils import normalize_scopes, normalize_conditions, normalize_demands, save_output, eval_scopes, eval_conditions, eval_demands, output_to_scopes, output_to_conditions, output_to_demands, create_latex_report

LABELS = ["SCOPE", "CONDITION", "DEMAND", "O"]

SEED = 42

def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("test", help="file with test data")
    parser.add_argument("attribute", help='scope/condition/demand')
    parser.add_argument("--save", help="paths to save output of the model", action="store")
    parser.add_argument("--latex", help="path to latex output of similarity scores")
    parser.add_argument("--prefix", help="prefix to distinguish variable names")
    parser.add_argument("--figure", help="figure to plot all measures")
    parser.add_argument("--result", "-r", help="path to save all the results as a .json file")

    args = parser.parse_args()
    logging.info("Running with arugments:")
    for arg in vars(args):
        logging.info("%s : %s", arg, getattr(args, arg))

    if args.attribute.lower() not in ['scope', 'condition', 'demand']:
        print("Unexpected attribute {}, please use scope,condition ord demand. exiting".format(args.attribute))
        exit()

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
    test_handler = DataHandler(args.test)
    test_data = test_handler.get_all_batches()
    eval_dict = eval(test_data, model_obj, device=device)
    print(eval_dict['tagtype_report'])
    output = eval_dict['output']

    if args.attribute.lower() == "scope":
        gold = test_handler.get_attributes(attribute="scope")
        output_scopes = output_to_scopes(output)
        pred = [normalize_scopes(scopes) for scopes in output_scopes]
        score, all_scores = eval_scopes(pred, gold)

    elif args.attribute.lower() == 'condition':
        gold = test_handler.get_attributes(attribute="condition")
        output_conditions = output_to_conditions(output)
        pred = [normalize_conditions(cond) for cond in output_conditions]
        score, all_scores = eval_conditions(pred, gold)

    elif args.attribute.lower() == "demand":
        logging.info("Evaluating demand")
        gold = test_handler.get_attributes(attribute="demand")
        output_demands = output_to_demands(output)
        pred = [normalize_demands(demand) for demand in output_demands]
        score, all_scores = eval_demands(pred, gold)
    else:
        logging.error("something went wrong")
        exit()


    if args.save:
        save_output(all_scores, args.save)

    for key, value in score.items():
        print("{}: {}".format(key, value))


    if args.latex:
        create_latex_report(score, args.latex, args.prefix)

    if args.figure:
        dpi = 300
        x = [x for x in range(len(all_scores['bleu']))]
        plt.figure(dpi=dpi)
        plt.scatter(x, all_scores['rouge_1-f'], s=1, label='rouge_1-f')
        plt.scatter(x, all_scores['rouge_l-f'], s=1, label='rouge_l-f')
        plt.scatter(x, all_scores['bleu'], s=1, label='bleu')
        plt.scatter(x, all_scores['meteor'], s=1, label='meteor')
        #plt.plot(all_scores['wer'], label='wer')
        plt.scatter(x, all_scores['wacc'], s=1, label='wacc')
        plt.scatter(x, all_scores['xml-cosine'], s=1, label='xml')
        plt.legend()
        plt.savefig(args.figure)

    if args.result:
        #print("score: {}".format(score))
        with open(args.result, 'w') as F:
            F.write(json.dumps(score, indent=4))
            logging.info("Written scores to %s", args.result)


if __name__ == '__main__':
    main()
