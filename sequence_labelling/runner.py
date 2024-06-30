import logging
import torch
from torch.optim import Adam, AdamW, SGD
from argparse import ArgumentParser
from transformers import RobertaForTokenClassification, get_linear_schedule_with_warmup, AutoTokenizer
from torch import nn
import matplotlib.pyplot as plt

from concat_ner.model import RobertaClassifier, DomainSpecificClassifier, CombinedClassifier, FlairClassifier, RobertaLargeClassifier
from concat_ner.data_handler import DataHandler
from concat_ner.train import train_loop, eval
from concat_ner.token_encoder import RobertaTokenEncoder, ClassicTokenEncoder, CombinedEncoder, FlairTokenEncoder
from concat_ner.label_encoder import LabelEncoder


LABELS = ["SCOPE", "CONDITION", "DEMAND", "O"]
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

#ROBERTA_MODEL = 'roberta-base'
ROBERTA_MODEL = 'roberta-large'

def initialize_combined_model(args, data):
    label_encoder = LabelEncoder(LABELS)
    token_encoder = CombinedEncoder(label_encoder, args.vocab_path, args.flair)
    #token_encoder.build_vocab(embedding_lines, vocab_size)

    model = CombinedClassifier(len(LABELS),
                               vocab_size=token_encoder.vocab_size,
                               emb_path=args.emb_path,
                               hidden_size=args.hidden,
                               full_finetuning=args.full_finetuning,
                               flair=args.flair)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]

    optim = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.eps
    )

    total_steps = len(data) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )


    model_obj = {"model" : model,
                 "token_encoder" : token_encoder,
                 "optim" : optim,
                 "scheduler" : scheduler}
    return model_obj

def initialize_domain_specific_model(args, data):
    label_encoder = LabelEncoder(LABELS)
    token_encoder = ClassicTokenEncoder(label_encoder, args.vocab_path)
    model = DomainSpecificClassifier(len(LABELS),
                                     vocab_size=token_encoder.vocab_size,
                                     emb_path=args.emb_path,
                                     hidden_size=args.hidden,
                                     full_finetuning=args.full_finetuning)

    optim = Adam(model.parameters(), lr=args.lr, eps=args.eps)
    #optim = SGD(model.parameters(), lr=args.lr)

    model_obj = {"model" : model,
                 "token_encoder" : token_encoder,
                 "optim" : optim}
    return model_obj

def initialize_flair_model(args, data):
    label_encoder = LabelEncoder(LABELS)
    token_encoder = FlairTokenEncoder(label_encoder)
    model = FlairClassifier(len(LABELS))

    optim = Adam(model.parameters(), lr=args.lr, eps=args.eps)
    model_obj = {"model" : model,
                 "token_encoder" : token_encoder,
                 "optim" : optim}
    return model_obj

def initialize_roberta_model(args, data):
    #model = RobertaClassifier(len(LABELS), hidden_size=args.hidden, full_finetuning=args.full_finetuning)
    #model = RobertaLargeClassifier(len(LABELS), hidden_size=args.hidden, full_finetuning=args.full_finetuning)
    model = RobertaForTokenClassification.from_pretrained(ROBERTA_MODEL,
                                                          num_labels=len(LABELS),
                                                          cache_dir='./cache',
                                                          classifier_dropout=0.5)
    label_encoder = LabelEncoder(LABELS)
    token_encoder = RobertaTokenEncoder(label_encoder)

    if args.full_finetuning:
        param_optimizer = list(model.named_parameters())  # all the model's parameters
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optim = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.eps
    )
    #optim = Adam(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)

    total_steps = len(data) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    model_obj = {"model" : model,
                 "token_encoder" : token_encoder,
                 "optim" : optim,
                 "scheduler" : scheduler}
    return model_obj




def main():
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", default=4, type=int)
    parser.add_argument("--save", help="paths to save the model", action="store", default="/home/ole/src/Concat_ner/models/model.pt")
    parser.add_argument("--train", help="file with train data", default="/home/ole/src/Req_annot/annotations_with_scope/train.jsonl")
    parser.add_argument("--test", help="file with test data")
    parser.add_argument("--lr", help="learning rate", default=0.001, type=float)  # 1e-5, or 3e-5 (6e-5)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--full_finetuning", default=False, action='store_true')
    parser.add_argument("--flair", help="include flair embeddings in the combined model", default=False, action='store_true')
    parser.add_argument("--emb_path", help="path to embeddings", default="/home/ole/src/Concat_ner/vectors/dnv_embeddings.pkl")
    parser.add_argument("--vocab_path", help="path to embed. vocab", default="/home/ole/src/Concat_ner/vectors/dnv_vocab.pkl")
    parser.add_argument("--test_size", help="train/test-split", default=0.1, type=float)
    parser.add_argument("--model", help="which model (roberta/domain/combined)", default='roberta')
    parser.add_argument("--hidden", help="size of hidden layer", default=200, type=int)
    parser.add_argument("--graph", "-g", help="show loss/f-score", default="last.png")
    parser.add_argument("--preprocess", help="case normalization, lemmatization etc.", default=False, action='store_true')
    parser.add_argument("--batch_size", "-b", help="batch size", default=1, type=int)

    args = parser.parse_args()
    logging.info("Running with arugments:")
    for arg in vars(args):
        logging.info("%s : %s", arg, getattr(args, arg))

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PREPARE DATA
    #random.seed(SEED)
    data_handler = DataHandler(args.train, preprocess=args.preprocess,
                               shuffle=True,
                               batch_size=args.batch_size,
                               test_size=args.test_size)

    #train_data, dev_data = data_handler.get_train_test_split(args.test_size)
    train_data = data_handler.get_train_batch()
    dev_data = data_handler.get_test_batch()

    # for testing
    #train_data = list(load_data(args.train))[:1]
    #dev_data = list(load_data(args.train))[:1]


    # INITIALIZE MODEL
    if args.model.lower() == 'roberta':
        model_obj = initialize_roberta_model(args, train_data)
    elif args.model.lower() == 'domain':
        model_obj = initialize_domain_specific_model(args, train_data)
    elif args.model.lower() == 'flair':
        model_obj = initialize_flair_model(args, train_data)
    elif args.model.lower() == 'combined':
        model_obj = initialize_combined_model(args, train_data)
    else:
        print("Unknown model: {}, try [roberta|domain|combined]".format(args.model))
        print("Exiting...")
        exit()

    #print("model: {}".format(model))

    # TRAIN MODEL
    loss_vals = []
    f1_vals = []
    logging.info("Starting to train")
    for epoch in range(args.epochs):
        logging.info("Epoch no: %d", epoch + 1)
        avg_loss, f1 = train_loop(train_data, model_obj, val=dev_data, device=device)
        loss_vals.append(avg_loss)
        f1_vals.append(f1)
    logging.info("loss vals: {}, f1_vals: {}".format(loss_vals, f1_vals))

    # save model to disk
    if args.save:
        logging.info("Saving model to: {}".format(args.save))
        torch.save(model_obj['model'], args.save)

    # save the graph with the f-scores
    if args.graph:
        plt.plot(loss_vals, label="avg loss")
        plt.plot(f1_vals, label="f1")
        plt.axis([0, args.epochs, 0, None])
        plt.xlabel("epoch no.")
        plt.legend()
        plt.savefig(args.graph)
        logging.info("Saved graph as %s", args.graph)
        #plt.show()

    # run test cycle
    if args.test:
        test_handler = DataHandler(args.test, batch_size=args.batch_size)
        test_data = test_handler.get_all_batches()
        res = eval(test_data, model_obj, device=device)
        print(res['tagtype_report'])
        print("Accuracy: {}".format(res['acc']))
        print("Accuracy without O: {}".format(res['acc_without_O']))
        print("F-score: {}".format(res['tagtype_f1_score']))

if __name__ == '__main__':
    main()
