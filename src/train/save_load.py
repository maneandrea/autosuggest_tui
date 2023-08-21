import os
import re

import torch
import pickle
import zipfile

from transformer import embedding, transformer


def unique_temp(path: str) -> str:
    """Returns a path for a temp file which does not overwrite existing files"""
    new_path = path + '_temp0'
    while os.path.isfile(new_path):
        new_path = re.sub(r'_temp(\d+)$', lambda m: f'_temp{int(m.group(1))+1}', new_path)
    return new_path


def save_model(model: transformer.Transformer, path: str, overwrite=False):
    """Saves a model in a given path"""

    temp_path = unique_temp(path)
    temp_emb = unique_temp(path + '_emb')

    if not os.path.isdir(os.path.dirname(os.path.realpath(path))):
        print(f"Error: directory '{os.path.dirname(path)}' does not exist")
        return False

    if not overwrite and os.path.isfile(path):
        choice = None
        while choice not in ['n', 'N', 'no', 'No', 'y', 'Y', 'yes', 'Yes']:
            choice = input(f"Warning: file '{path}' exists, overwriting? [y/n] ")
            if choice in ['n', 'N', 'no', 'No']:
                print('Ok, bye')
                return True
            elif choice not in ['y', 'Y', 'yes', 'Yes']:
                print('Choose yes (y) or no (n).')

    try:
        torch.save(model.state_dict(), temp_path)
        with open(temp_emb, 'wb') as f:
            # We don't dump the last token ('') because it's the placeholder for out-of-vocabulary
            pickle.dump(model.embeddings.decode[:-1], f)
        with zipfile.ZipFile(path, 'w') as z:
            z.write(temp_path, 'model.pt')
            z.write(temp_emb, 'vocabulary.pickle')
        os.remove(temp_path)
        os.remove(temp_emb)
    except (IOError, RuntimeError) as e:
        print(e)
        return False
    else:
        return True


def load_model(path: str, context_size: int) -> transformer.Transformer | None:
    """Loads a model into the given reference"""

    if not os.path.isfile(path):
        print(f"Error: file '{path}' does not exist")
        return None

    try:
        temp_model = unique_temp(path)
        temp_token = unique_temp(path + '_tok')
        with zipfile.ZipFile(path, 'r') as z:
            with open(temp_model, 'wb') as tm:
                tm.write(z.read('model.pt'))
            with open(temp_token, 'wb') as tt:
                tt.write(z.read('vocabulary.pickle'))

        emb = embedding.WordEmbedding(load_emb(temp_token))
        model = transformer.Transformer(context_size, emb)
        model.load_state_dict(torch.load(temp_model))

    except (IOError, RuntimeError) as e:
        print('Error: ' + str(e))
        return None
    else:
        os.remove(temp_token)
        os.remove(temp_model)
        return model


def load_emb(path: str) -> list | None:
    """Loads the list of tokens to be given to a WordEmbedding object"""

    try:
        with open(path, 'rb') as e:
            return pickle.load(e)
    except FileNotFoundError:
        print(f'File {path} not available, re-tokenizing text')
        return None
    except (IOError, RuntimeError) as e:
        print('Error: ' + str(e))
        return None


def save_emb(emb: embedding.WordEmbedding, path: str):
    """Saves the list of tokens in a given file"""

    if not os.path.isdir(os.path.dirname(os.path.realpath(path))):
        print(f"Error: directory '{os.path.dirname(path)}' does not exist")
        return False

    if os.path.isfile(path):
        choice = None
        while choice not in ['n', 'N', 'no', 'No', 'y', 'Y', 'yes', 'Yes']:
            choice = input(f"Warning: file '{path}' exists, overwriting? [y/n] ")
            if choice in ['n', 'N', 'no', 'No']:
                print('Ok, bye')
                return True
            elif choice not in ['y', 'Y', 'yes', 'Yes']:
                print('Choose yes (y) or no (n).')

    try:
        with open(path, 'wb') as f:
            pickle.dump(emb.tokens, f)
    except (IOError, RuntimeError) as e:
        print(e)
        return False
    else:
        return True
