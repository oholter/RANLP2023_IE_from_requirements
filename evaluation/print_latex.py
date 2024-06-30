import logging
import numpy as np
import json
from argparse import ArgumentParser
from pathlib import Path


def read_data(path):
    path = Path(path)
    if not path.exists():
        print("{} does not exist".format(path))
        print("Exiting")
        exit()

    with path.open() as F:
        data = json.load(F)
        logging.info("Read info from %s", path)

    return data

def create_latex_report(d, prefix=None, z=1.96):
    """
    Generate latex macros with means and confidence intervals (z*stddev)
    Need:
        RougeLF
        Bleu
        LBM
        WAcc
    """
    if not prefix:
        prefix = ""

    s = ""
    #s += "\\newcommand*\\{}ExactPMean{{{}}}".format(prefix,d['precision']['mean'])
    #s += "\\newcommand*\\{}ExactPInterval{{{}}}".format(prefix,z*d['precision']['stdev'])
    #s += "\\newcommand*\\{}ExactRMean{{{}}}".format(prefix,d['recall']['mean'])
    #s += "\\newcommand*\\{}ExactRInteval{{{}}}".format(prefix,z*d['recall']['stdev'])
    #s += "\\newcommand*\\{}ExactFMean{{{}}}".format(prefix,d['f1']['mean'])
    #s += "\\newcommand*\\{}ExactFInterval{{{}}}".format(prefix,z*d['f1']['stdev'])
    #s += "\\newcommand*\\{}RougeOnepMean{{{}}}".format(prefix,d['rouge_1-p']['mean'])
    #s += "\\newcommand*\\{}RougeOnepInterval{{{}}}".format(prefix,z*d['rouge_1-p']['stdev'])
    #s += "\\newcommand*\\{}RougeOnerMean{{{}}}".format(prefix,d['rouge_1-r']['mean'])
    #s += "\\newcommand*\\{}RougeOnerInterval{{{}}}".format(prefix,z*d['rouge_1-r']['stdev'])
    #s += "\\newcommand*\\{}RougeOnefMean{{{}}}".format(prefix,d['rouge_1-f']['mean'])
    #s += "\\newcommand*\\{}RougeOnefInterval{{{}}}".format(prefix,z*d['rouge_1-f']['stdev'])
    #s += "\\newcommand*\\{}RougelpMean{{{}}}".format(prefix,d['rouge_l-p']['mean'])
    #s += "\\newcommand*\\{}RougelpInterval{{{}}}".format(prefix,z*d['rouge_l-p']['stdev'])
    #s += "\\newcommand*\\{}RougelrMean{{{}}}".format(prefix,d['rouge_l-r']['mean'])
    #s += "\\newcommand*\\{}RougelrInterval{{{}}}".format(prefix,z*d['rouge_l-r']['stdev'])
    s += "\\newcommand*\\{}RougelfMean{{{}}}".format(prefix,d['rouge_l-f']['mean'])
    s += "\\newcommand*\\{}RougelfInterval{{{}}}".format(prefix,z*d['rouge_l-f']['stdev'])
    s += "\\newcommand*\\{}BleuMean{{{}}}".format(prefix,d['bleu-1']['mean'])
    s += "\\newcommand*\\{}BleuInterval{{{}}}".format(prefix,z*d['bleu-1']['stdev'])
    #s += "\\newcommand*\\{}MeteorMean{{{}}}".format(prefix,d['meteor']['mean'])
    #s += "\\newcommand*\\{}MeteorInterval{{{}}}".format(prefix,z*d['meteor']['stdev'])
    #s += "\\newcommand*\\{}WerMean{{{}}}".format(prefix,d['wer']['mean'])
    #s += "\\newcommand*\\{}WerInterval{{{}}}".format(prefix,z*d['wer']['stdev'])
    #s += "\\newcommand*\\{}WaccMean{{{}}}".format(prefix,1 - d['wer']['mean'])
    #s += "\\newcommand*\\{}WaccInterval{{{}}}".format(prefix,1 - z*d['wer']['stdev'])
    s += "\\newcommand*\\{}XmlCosineMean{{{}}}".format(prefix,d['xml-cosine']['mean'])
    s += "\\newcommand*\\{}XmlCosineInterval{{{}}}".format(prefix,z*d['xml-cosine']['stdev'])
    s += "\n"
    #print(s)

    return s

def create_diff_report(dw, do, prefix=None, z=1.96):
    """
    Generate latex macros with the mean differences and z*standard deviation
    Only need:
        RougeLF
        Bleu
        LBM
        WAcc
    """
    s = ""
    #precision_diffs = np.array(dw['precision']['results']) - np.array(do['precision']['results'])
    #s += "\\newcommand*\\{}DiffExactPMean{{{}}}".format(prefix,precision_diffs.mean())
    #s += "\\newcommand*\\{}DiffExactPInterval{{{}}}".format(prefix,z*precision_diffs.std())

    #recall_diffs = np.array(dw['recall']['results']) - np.array(do['recall']['results'])
    #s += "\\newcommand*\\{}DiffExactRMean{{{}}}".format(prefix, recall_diffs.mean())
    #s += "\\newcommand*\\{}DiffExactRInteval{{{}}}".format(prefix,z*recall_diffs.std())

    #f1_diffs = np.array(dw['f1']['results']) - np.array(do['recall']['results'])
    #s += "\\newcommand*\\{}DiffExactFMean{{{}}}".format(prefix,f1_diffs.mean())
    #s += "\\newcommand*\\{}DiffExactFInterval{{{}}}".format(prefix,z*f1_diffs.std())

    #s += "\\newcommand*\\{}DiffRougeOnepMean{{{}}}".format(prefix,dw['rouge_1-p']['mean'] - do['rouge_1-p']['mean'])
    #s += "\\newcommand*\\{}DiffRougeOnepInterval{{{}}}".format(prefix,z*dw['rouge_1-p']['stdev'])
    #s += "\\newcommand*\\{}DiffRougeOnerMean{{{}}}".format(prefix,dw['rouge_1-r']['mean'] - do['rouge_1-r']['mean'])
    #s += "\\newcommand*\\{}DiffRougeOnerInterval{{{}}}".format(prefix,z*dw['rouge_1-r']['stdev'])
    #s += "\\newcommand*\\{}DiffRougeOnefMean{{{}}}".format(prefix,dw['rouge_1-f']['mean'] - do['rouge_1-f']['mean'])
    #s += "\\newcommand*\\{}DiffRougeOnefInterval{{{}}}".format(prefix,z*dw['rouge_1-f']['stdev'])
    #s += "\\newcommand*\\{}DiffRougelpMean{{{}}}".format(prefix,dw['rouge_l-p']['mean'] - do['rouge_l-p']['mean'])
    #s += "\\newcommand*\\{}DiffRougelpInterval{{{}}}".format(prefix,z*dw['rouge_l-p']['stdev'])
    #s += "\\newcommand*\\{}DiffRougelrMean{{{}}}".format(prefix,dw['rouge_l-r']['mean'] - do['rouge_l-r']['mean'])
    #s += "\\newcommand*\\{}DiffRougelrInterval{{{}}}".format(prefix,z*dw['rouge_l-r']['stdev'])

    rouge_diffs = np.array(dw['rouge_l-f']['results']) - np.array(do['rouge_l-f']['results'])
    s += "\\newcommand*\\{}DiffRougelfMean{{{}}}".format(prefix, rouge_diffs.mean())
    s += "\\newcommand*\\{}DiffRougelfInterval{{{}}}".format(prefix,z*rouge_diffs.std())

    bleu_diffs = np.array(dw['bleu-1']['results']) - np.array(do['bleu-1']['results'])
    s += "\\newcommand*\\{}DiffBleuMean{{{}}}".format(prefix, bleu_diffs.mean())
    s += "\\newcommand*\\{}DiffBleuInterval{{{}}}".format(prefix,z*bleu_diffs.std())

    #s += "\\newcommand*\\{}DiffMeteorMean{{{}}}".format(prefix,dw['meteor']['mean'] - do['meteor']['mean'])
    #s += "\\newcommand*\\{}DiffMeteorInterval{{{}}}".format(prefix,z*dw['meteor']['stdev'])
    #s += "\\newcommand*\\{}DiffWerMean{{{}}}".format(prefix,dw['wer']['mean'] - do['wer']['mean'])
    #s += "\\newcommand*\\{}DiffWerInterval{{{}}}".format(prefix,z*dw['wer']['stdev'])

    wacc_diffs = (1 - np.array(dw['wer']['results'])) - (1 - np.array(do['wer']['results']))
    s += "\\newcommand*\\{}DiffWaccMean{{{}}}".format(prefix, wacc_diffs.mean())
    s += "\\newcommand*\\{}DiffWaccInterval{{{}}}".format(prefix,1 - z*wacc_diffs.std())

    lbm_diff = np.array(dw['xml-cosine']['results']) - np.array(do['xml-cosine']['results'])
    s += "\\newcommand*\\{}DiffXmlCosineMean{{{}}}".format(prefix, lbm_diff.mean())
    s += "\\newcommand*\\{}DiffXmlCosineInterval{{{}}}".format(prefix,z*lbm_diff.std())
    s += "\n"
    #print(s)

    return s



def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("input", help="experiments folder")
    parser.add_argument("--latex", help="save to disk as .tex file")
    args = parser.parse_args()

    #data_context = read_data(args.with_context)
    #data_noc = read_data(args.without_context)


    path = Path(args.input)

    if not path.exists():
        print("The path: {} does not exist. Exiting".format(path))
        exit()

    # find all .json-files "with context"
    #files_with_context = [f for f in path.iterdir() if f.is_file() and f.suffix == '.json' and '_with_' in f.stem]
    #files_with_context.sort()
    files = [f for f in path.iterdir() if f.is_file() and f.suffix == '.json']

    s = ""
    for file in files:
        file_name_components = file.stem.split("_")
        if len(file_name_components) < 3:
            print("Not using: {}".format(file))
            continue

        prefix = file_name_components[1]
        if prefix == "with":
            prefix = file_name_components[2]

        if prefix[-1] == "5":
            prefix = prefix[:-1] + "five"
        elif prefix[-1] == "6":
            prefix = prefix[:-1] + "six"

        if len(file_name_components) == 4:
            prefix += file_name_components[3].capitalize()
        if len(file_name_components) == 3 and file_name_components[1] != "with":
            prefix += file_name_components[2].capitalize()


        #noc_prefix = prefix
        #prefix = "with" + prefix.capitalize()

        if file_name_components[1] == "with":
            prefix = "with" + prefix.capitalize()

        #file_without_context = path / file.name.replace("with_", "")
        #if not file_without_context.exists():
            #logging.warning("file: %s was not found, skipping", file_without_context)
            #continue

        data = read_data(file)
        #data_noc = read_data(file_without_context)

        s += create_latex_report(data, prefix=prefix)
        #s += create_latex_report(data_noc, prefix=noc_prefix)
        #s += create_diff_report(data_context, data_noc, prefix=noc_prefix)

        s += "\n"


    if args.latex:
        with open(args.latex, 'w') as F:
            F.write(s)
            logging.info("written latex variables to {}".format(args.latex))

    print(s)




if __name__ == '__main__':
    main()
