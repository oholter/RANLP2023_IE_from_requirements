import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from concat_ner.data_handler import DataHandler
#from concat_ner.scope_eval_utils import normalize_scopes, save_output, eval_scopes, create_latex_report
from concat_ner.scd_eval_utils import normalize_scopes, normalize_conditions, normalize_demands, save_output, eval_scopes, eval_conditions, eval_demands, output_to_scopes, output_to_conditions, output_to_demands, create_latex_report


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("pred", help="file with predicted scopes")
    parser.add_argument("gold", help="file with gold scopes")
    parser.add_argument("attribute", help='scope/condition/demand')
    parser.add_argument("--save", help="paths to save output of the model", action="store")
    parser.add_argument("--latex", help="path to latex output of similarity scores")
    parser.add_argument("--prefix", help="prefix to distinguish variable names")
    parser.add_argument("--figure", help="fire to plot all measures")

    args = parser.parse_args()
    logging.info("Running with arugments:")
    for arg in vars(args):
        logging.info("%s : %s", arg, getattr(args, arg))

    if args.attribute.lower() not in ['scope', 'condition', 'demand']:
        print("Unexpected attribute {}, please use scope,condition ord demand. exiting".format(args.attribute))
        exit()

    # run test cycle
    pred_handler = DataHandler(args.pred)
    gold_handler = DataHandler(args.gold)

    if args.attribute.lower() == "scope":
        logging.info("Evaluating scope")
        pred = pred_handler.get_attributes(attribute="scope")
        #print(pred)
        pred = [normalize_scopes(scopes) for scopes in pred]

        gold = gold_handler.get_attributes(attribute="scope")
        gold = [normalize_scopes(scopes) for scopes in gold]

        score, all_scores = eval_scopes(pred, gold)

    elif args.attribute.lower() == 'condition':
        logging.info("Evaluating condition")
        pred = pred_handler.get_attributes(attribute="condition")
        pred = [normalize_conditions(cond) for cond in pred]

        gold = gold_handler.get_attributes(attribute="condition")
        gold = [normalize_conditions(cond) for cond in gold]


        score, all_scores = eval_conditions(pred, gold)

    elif args.attribute.lower() == "demand":
        logging.info("Evaluating demand")
        pred = pred_handler.get_attributes(attribute="demand")
        pred = [normalize_demands(demand) for demand in pred]

        gold = gold_handler.get_attributes(attribute="demand")
        gold = [normalize_demands(demand) for demand in gold]

        score, all_scores = eval_demands(pred, gold)
    else:
        logging.error("something went wrong")
        exit()


    if args.save:
        save_output(pred, gold, args.save)

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

if __name__ == '__main__':
    main()
