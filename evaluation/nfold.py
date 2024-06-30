import logging
import random
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# read file
# split into n
# combine into n train/test splits
# optionally: write to files


def read_data(path):
    """ Reading the data file, can handle both .json and .jsonl """
    logging.info("Loading dataset from %s", path)
    data_path = Path(path)
    if data_path.suffix == ".json":
        with data_path.open(mode="r") as F:
            data = json.load(F)
    elif data_path.suffix == ".jsonl":
        with data_path.open(mode="r") as F:
            data = [json.loads(l) for l in list(F)]
    else:
        logging.error("%s has unknown suffix %s", data_path, data_path.suffix)
        data = None

    return data

def write_file(data, path):
    """ writes one file as .jsonl"""
    with path.open(mode='w') as F:
        for obj in data:
            F.write(json.dumps(obj, ensure_ascii=False) + "\n")

        logging.info("Written %d documents to %s", len(data), path)


def split_data(data, n):
    """ partition data into n partitions """

    # n-fold must first shuffle the data
    random.shuffle(data)
    return np.array_split(data, n)


def compile_train_test(splits):
    for fidx in range(len(splits)):
        test = splits[fidx]
        before = splits[:fidx]
        after = splits[fidx+1:]
        train = before + after
        # flatten the list of splits (a list of lists)
        train = [item for s in train for item in s]
        yield fidx, train, test


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--input",
                        help="the file to split",
                        default="/home/ole/src/Req_annot/annotations_with_scd/all.jsonl")
    parser.add_argument("-n",
                        help="the number of folds",
                        type=int,
                        default=10)
    parser.add_argument("--output", "-o",
                        help="the folder to write the train/test split",
                        default="./experiments/")

    args = parser.parse_args()

    data = read_data(args.input)
    splits = split_data(data, args.n)
    parts = compile_train_test(splits)

    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    for idx, train, test in parts:
        # create one folder for each of the experiments (idx)
        path = Path(args.output) / str(idx)
        if not path.exists():
            path.mkdir()

        train_path = path / "train.jsonl"
        test_path = path / "test.jsonl"
        write_file(train, train_path)
        write_file(test, test_path)




if __name__ == '__main__':
    main()
