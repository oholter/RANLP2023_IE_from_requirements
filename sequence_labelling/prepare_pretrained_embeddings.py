"""
Preparing the embeddings for use with the concat-ner project
adding a vocab element and embeddings for
unk, stop, pad, punct

and storing the embeddings and vocabulary for shorter load times
"""
import torch
from pathlib import Path
import logging
import numpy as np
from argparse import ArgumentParser

from concat_ner.token_encoder import UNK_IDX, UNK_TOK, PUNCT_IDX, PUNCT_TOK, PAD_IDX, PAD_TOK, STOP_IDX, STOP_TOK



def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", help=".txt file with embeddings")
    parser.add_argument("--out_emb", "-o", help="file to save the embeddings")
    parser.add_argument("--out_vocab", "-v", help="file to save the out vocab")
    args = parser.parse_args()


    if not args.input:
        print("expecting argument --input")
        exit()
    if not args.out_emb:
        print("expecting argument --out_emb")
        exit()
    if not args.out_vocab:
        print("expecting argument --out_vocab")
        exit()

    embeddings_path = Path(args.input)
    if not embeddings_path.exists():
        print("file %s not found, exiting".format(embeddings_path))
        exit()

    with embeddings_path.open(mode='r') as F:
        full_content = F.read().strip().split('\n')

    # Loading and preparing the embeddings file
    logging.info("Loading and preparing the embeddings file")
    #dim = full_content[0].split()[0]
    #num_toks = full_content[0].split()[1]

    # gensim word2vec adds dim/vocab size on the first line
    full_content = full_content[1:]

    vocab, embeddings = [], []
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_emb = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_emb)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    # Addding extra embeddings and vocab items
    logging.info("Adding the extra embeddings and vocab items")
    vocab_npa = np.insert(vocab_npa, UNK_IDX, UNK_TOK)
    vocab_npa = np.insert(vocab_npa, PUNCT_IDX, PUNCT_TOK)
    vocab_npa = np.insert(vocab_npa, PAD_IDX, PAD_TOK)
    vocab_npa = np.insert(vocab_npa, STOP_IDX, STOP_TOK)
    logging.info("vocab_npa[:10]: {}".format(vocab_npa[:10]))

    unk_emb = np.mean(embs_npa, axis=0, keepdims=True)
    punct_emb = np.zeros((1, embs_npa.shape[1]))
    pad_emb = np.zeros((1, embs_npa.shape[1]))
    stop_emb = np.mean(embs_npa, axis=0, keepdims=True)

    embs_npa = np.vstack((unk_emb, punct_emb, pad_emb, stop_emb, embs_npa))
    logging.info("embs_npa.shape: {}".format(embs_npa.shape))

    # Saving the embeddings and vocab
    with open(args.out_vocab, 'wb') as F:
        np.save(F, vocab_npa)
        logging.info("Saved vocab to: %s", args.out_vocab)
    with open(args.out_emb, 'wb') as F:
        np.save(F, embs_npa)
        logging.info("Saved embeddings to: %s", args.out_emb)

if __name__ == '__main__':
    main()
