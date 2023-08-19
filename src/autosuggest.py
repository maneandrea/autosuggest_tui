#!/usr/bin/env python

import sys
import os

from tui import app
from transformer import suggest

if __name__ == '__main__':

    if sys.argv[1:]:
        file = sys.argv[1]
    else:
        file = os.path.join(os.path.dirname(__file__), '..', 'data', 'gpt.pt.zip')

    S = suggest.Suggester(file)

    App = app.AutosuggestApp(S)
    App.run()
