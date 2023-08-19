#!/usr/bin/env python

import argparse
import sys
import hashlib

from train import loop
from params import hyperparams

LEARNING_RATE = hyperparams.LEARNING_RATE
BATCH_SIZE = hyperparams.BATCH_SIZE
EPOCHS = hyperparams.EPOCHS
FRACTION = hyperparams.FRACTION

parser = argparse.ArgumentParser('autosuggest-train',description='train the autosuggest program with a text corpus')
parser.add_argument('input', nargs='+', help='path of the text file(s) to use')
parser.add_argument('-r', '--lr', type=float,
                    help=f'learning rate (default {LEARNING_RATE})', default=LEARNING_RATE)
parser.add_argument('-b', '--batch', type=int,
                    help=f'batch size (default {BATCH_SIZE})', default=BATCH_SIZE)
parser.add_argument('-e', '--epochs', type=int,
                    help=f'number of epochs (default {EPOCHS})', default=EPOCHS)
parser.add_argument('-f', '--fraction', type=float,
                    help=f'fraction of data used for validation (default {FRACTION})', default=FRACTION)
parser.add_argument('-s', '--save', help='save the state of the model in the given file')
parser.add_argument('-l', '--load', help='load the state of the model from the given file')


if __name__ == '__main__':

    args = parser.parse_args()
    try:
        corpus = ''
        for file in args.input:
            with open(file, 'r') as f:
                corpus += '\n' + f.read()
    except FileNotFoundError as e:
        print('Error: ' + str(e))
        sys.exit(1)

    learning_rate = args.lr if args.lr else LEARNING_RATE
    batch_size = args.batch if args.batch else BATCH_SIZE
    epochs = args.epochs if args.epochs else EPOCHS
    fraction = args.fraction if args.fraction else FRACTION

    # We hash the filenames so that we can use that as a name for the token pickle file
    # Careful: if the file changes while keeping the same name the wrong pickle will be loaded!
    corpus_hash = hashlib.md5(' '.join(args.input).encode('utf-8')).hexdigest()

    loop.train_model(
        corpus,
        corpus_hash,
        learning_rate,
        batch_size,
        epochs,
        fraction,
        args.save,
        args.load
    )
