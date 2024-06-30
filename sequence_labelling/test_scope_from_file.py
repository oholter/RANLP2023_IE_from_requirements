import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sequence_labelling.data_handler import DataHandler
from sequence_labelling.scope_eval_utils import normalize_scopes, save_output, eval_scopes, create_latex_report


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("pred", help="file with predicted scopes")
    parser.add_argument("gold", help="file with gold scopes")
    parser.add_argument("--save", help="paths to save output of the model", action="store", default="/home/ole/src/Concat_ner/scope_output/output.txt")
    parser.add_argument("--latex", help="path to latex output of similarity scores")
    parser.add_argument("--prefix", help="prefix to distinguish variable names")
    parser.add_argument("--figure", help="fire to plot all measures")

    args = parser.parse_args()
    logging.info("Running with arugments:")
    for arg in vars(args):
        logging.info("%s : %s", arg, getattr(args, arg))

    # run test cycle
    pred_handler = DataHandler(args.pred)
    pred_scopes = pred_handler.get_scopes()

    gold_handler = DataHandler(args.gold)
    gold_scopes = gold_handler.get_scopes()

    # the same normalizing that is done on gold data
    norm_pred_scopes = [normalize_scopes(scopes) for scopes in pred_scopes]

    if args.save:
        save_output(norm_pred_scopes, gold_scopes, args.save)


    score, all_scores = eval_scopes(norm_pred_scopes, gold_scopes)
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
