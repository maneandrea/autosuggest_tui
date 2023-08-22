import re
import math


## For the moment we use a letter-based tokenizer, so this is left for future usage
def tokenize_future(text: str, max_tokens=math.inf) -> [str]:
    """Splits the given text into a list of tokens"""
    gen = re.finditer(r"[a-zA-Z']+|[,.;?!]", text[::-1], re.UNICODE)
    to_return = []
    for i, a in enumerate(gen):
        if i < max_tokens:
            to_return.append(a.group(0).lower()[::-1])
        else:
            break
    return reversed(to_return)


def tokenize(text: str, max_tokens=None) -> [str]:
    """Splits the given text into a list of tokens"""
    m = -max_tokens if max_tokens else None
    return list(text[m:])

def end_word(vocabulary: dict[str, int]) -> [int]:
    """Returns the positions of the tokens that can end a word"""

    end_word_regex = r'[\s.,;:!?]+$'

    return [
        i for w, i in vocabulary.items() if re.match(end_word_regex, w)
    ]