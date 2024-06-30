import json
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    path = Path(args.input)

    if not path.exists():
        print("path %s does not exists, exiting", path)
        exit()

    print("Trying to parse")
    with path.open(mode='r') as F:
        for i, line in enumerate(F):
            print("Parsing line {}".format(i+1))
            data = json.loads(line)
            print("data: {}".format(data))






if __name__ == '__main__':
    main()
