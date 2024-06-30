import logging
import traceback
import time
import json
import openai
import random as rnd
from argparse import ArgumentParser
from pathlib import Path

from gpt.prompt_generator import PromptGeneratorGPT3



PROMPT_FILE = "prompt.txt"
CONFIG_FILE = "gpt/config.json"


def query_openai(prompt, **kwargs):
    model = kwargs['model']
    temperature = kwargs['temperature']
    stop = kwargs["stop"]
    top_p = kwargs['top_p']
    n = kwargs['n']
    max_tokens = kwargs['max_tokens']

    response = openai.Completion.create(model=model,
                                        prompt=prompt,
                                        temperature=temperature,
                                        top_p=top_p,
                                        n=n,
                                        #stop=stop,
                                        max_tokens=max_tokens)
    return response


def read_file(path):
    with path.open(mode='r') as F:
        data = [json.loads(l) for l in F]
    return data


def main():
    """
    1) reads in file with examples
    2) reads in file with targets
    3) creates a prompt > prompt.txt
    4) inserts the next requirement into gold-file
    """
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="config file")
    parser.add_argument("--id", "-i", help="comma separated ids")
    args = parser.parse_args()


    if args.cfg:
        config_file = args.cfg
    else:
        config_file = CONFIG_FILE

    with open(config_file, 'r') as F:
        config = json.load(F)
    io = config['io']
    examples_file = io['examples']
    targets_file = io['targets']
    output_file = io['output']
    openai.api_key = config['openai']["api_key"]

    examples_path = Path(examples_file)
    if not examples_path.exists():
        print("Cannot find {}. exiting...".format(examples_path))
        exit()

    targets_path = Path(targets_file)
    if not targets_path.exists():
        print("Cannot find {}. exiting...".format(targets_path))
        exit()

    if not args.id:
        print("no ids provided, exiting")
        exit()



    output_path = Path(output_file)

    E = read_file(examples_path)
    #E = E[:20]
    T = read_file(targets_path)
    #T = T[:2]

    ids = args.id
    ids = ids.split(",")
    ids = [int(i) for i in ids]
    print(ids)

    p_gen = PromptGeneratorGPT3(T, E, args.id, 100000, **config['prompt'])
    #prompts = p_gen.generate_prompts()
    prompts = [p_gen.generate_prompt(T[i]) for i in ids]
    logging.info("number of prompts: {}".format(len(prompts)))
    #exit()

    with output_path.open(mode='a') as O:
        for prompt_id, prompt in zip(ids, prompts):
            try:
                response = query_openai(prompt, **config['openai'])
                #logging.info("Got response: {}".format(response))

                response_text = ""
                for choice in response['choices']:
                    response_text += choice['text']

                logging.info("Response text: {}".format(response_text))
                O.write("{} - {}\n".format(prompt_id, response_text))
            except Exception as e:
                O.write("{} not prompted correctly\n".format(prompt_id))
                logging.error(traceback.format_exc())
                time.sleep(5)


if __name__ == '__main__':
    main()
