import logging
import json
import random as rand
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from argparse import ArgumentParser
from gpt.vectorizer import SentenceVectorizer


vec = None

def sample_requirements(G, n=0, ids=None):
    if ids:
        samples = {}
        for id in ids:
            if id in G:
                if 'F-prime' in G[id] and not \
                        ('ignore' in G[id] and G[id]['ignore'].lower()
                            in ['yes', '1', 'true']):
                    samples[id] = G[id]
                else:
                    logging.warning("id: %d must be ignored", id)
            else:
                logging.warning("id: %d not found in G", id)
        return samples

    if len(G) > n:
        return dict(rand.sample(G.items(), n))
    else:
        logging.warning("Sample size smaller than len(G)")
        return G


def find_similar_requirements(examples, target, n, sorting=None, context=False):
    """ 1) embed all examples + r
        2) calculate dist from each emb to r.emb
        3) sort r according to dist
        4) return n with shortest dist

        optional top n desc

    returns [list of r dicts]
    """

    # avoid recreating vec because this takes a lot of time !
    global vec

    if len(examples) == 0:
        return []
    if len(examples) == 1:
        return examples

    # cannot have more samples than actual items in examples
    n = min(len(examples)-1, n)

    # to remove the ones without F-prime and ignored
    #print("len examples: {}".format(len(examples)))

    if vec == None:
        vec = SentenceVectorizer()

        logging.info("examples sentence vectors")
        for e in tqdm(examples):
            if 'vec' not in e:
                #if context:
                    #e['vec'] = vec(e['text'].replace(".", " "))
                #else:
                e['vec'] = vec(e['meta']['org'])

    #logging.info("vectorizing target")
    # Not using context for semantic similarity, would have to deal with
    # sentence tokenizing
    #if context:
    #    target['vec'] = vec(target['text'])
    #else:
    target['vec'] = vec(target['meta']['org'])

    e_vector_arr = np.array([e['vec'] for e in examples])
    distances = np.linalg.norm(e_vector_arr - target['vec'], axis=1)

    id_with_dist = list(zip(list(range(len(distances))), distances))
    id_with_dist.sort(key=lambda x: x[1])

    # keeping only the top n
    id_with_dist = id_with_dist[:n]

    if sorting == "desc":
        id_with_dist.sort(key=lambda x: x[1], reverse=True)
        # print("desc: {}".format(id_with_dist))
    if sorting == "random":
        rand.shuffle(id_with_dist)

    closest = []
    for id, _ in id_with_dist:
        closest.append(examples[id])

    return closest


class PromptGenerator(ABC):
    """
    generates a prompt for a language model
    """
    @abstractmethod
    def __init__(self, r, G, **kwargs):
        pass

    @abstractmethod
    def generate_prompt(self):
        pass


class PromptGeneratorGPT3(PromptGenerator):
    """
    generates a prompt for a language model from dictionaries
    """
    def __init__(self, targets, examples, id, limit, **kwargs):
        self.targets = targets
        self.examples = examples
        self.strategy = kwargs['strategy'].lower().strip()
        self.n_samples = kwargs['n_samples']
        self.sorting = kwargs['sort']
        self.id = id
        self.limit = limit
        if self.sorting not in ['asc', 'desc', 'random']:
            logging.warning("sorting is {} will be treated as asc".format(self.sorting))
            self.sorting = None

        self.context = kwargs['context']
        self.prefix = "Below are some inputs and the outputs of an information \
extraction tool of industry standards. It always extracts the scope, the \
condition and the demand from a textual requirement. The scope is a \
physical artefact the requirement is about, the condition a condition that applies \
to the scope or a process the scope is part of and the demand is whatever \
is demanded of the scope to conform to the document. The output is in .json \
format.\n\n"

    def generate_prompts(self):
        prompts = []
        for i, target in enumerate(self.targets):
            if i < self.id:
                logging.info("skipping: i:{}".format(i))
                continue
            if (i - self.id) >= self.limit:
                logging.info("break: i:{}".format(i))
                break
            prompt = self.generate_prompt(target)
            prompts.append(prompt)

        return prompts


    def generate_prompt(self, target):
        if self.strategy == "most similar":
            samples = find_similar_requirements(self.examples, target, self.n_samples, sorting=self.sorting, context=self.context)
        elif self.strategy == "random":
            samples = sample_requirements(self.examples, n=self.n_samples)

        p = self.prefix
        for s in samples:
            if self.strategy == "most similar":
                if self.context:
                    p += "Input: {}\n".format(s['text'])
                else:
                    if self.context:
                        p += "Input: {}\n".format(s['text'])
                    else:
                        p += "Input: {}\n".format(s['meta']['org'])

                if 'scopes' in s['meta']:
                    scopes = s['meta']['scopes']
                else:
                    scopes = []
                if 'conditions' in s['meta']:
                    conditions = s['meta']['conditions']
                else:
                    conditions = []
                if 'demands' in s['meta']:
                    demands = s['meta']['demands']
                else:
                    demands = []
                p +="Output: {{\"meta\" : {{\"scopes\": {}, \"conditions\": {}, \"demands\": {}\"}}}}\n\n".format(scopes, conditions, demands)
        if self.context:
            p += "Input: {}\n".format(target['text'])
        else:
            p += "Input: {}\n".format(target['meta']['org'])
        p += "Output: "

        return p


def read_documents(path):
    with path.open(mode='r') as F:
        data = [json.loads(l) for l in F]
        logging.info("Read %d docs from %s", len(data), path)

    return data


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--id", help="which req to start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--cfg",
                        help="config file",
                        default="gpt/config.json")
    args = parser.parse_args()

    config_file = args.cfg
    with open(config_file, 'r') as F:
        config = json.load(F)
    io = config['io']

    examples_path = Path(io['examples'])
    targets_path = Path(io['targets'])
    if not examples_path.exists():
        print("Cannot find examples file: {}".format(examples_path))
        exit()
    if not targets_path.exists():
        print("Cannot find targets file: {}".format(targets_path))
        exit()

    examples = read_documents(examples_path)
    targets = read_documents(targets_path)

    examples = examples[:20]
    targets = targets[:2]

    #for i, t in enumerate(targets):
        #if i < args.id:
            #print("skipping: i:{}".format(i))
            #continue
        #if (i - args.id) > args.limit:
            #print("break: i:{}".format(i))
            #break

    #generator = PromptGeneratorChatGPT(t, examples, **config['prompt'])
    generator = PromptGeneratorGPT3(targets, examples, args.id, args.limit, **config['prompt'])
    prompt = generator.generate_prompts()
    print(prompt)


if __name__ == '__main__':
    main()
