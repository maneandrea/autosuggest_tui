Autosuggest
---

Simple text field that gives suggestions for continuing a word after a <Tab> press.
Based on the [Transformer](http://arxiv.org/abs/1706.03762v7) architecture and developed in PyTorch.
It is a curses-based TUI.

####Training
To train download one or more sources of text and run
```bash
train.py SOURCE [SOURCES..] --save path/to/model.pt.zip
```
See `train.py --help` for more options. Edit `src/params/hyperparams.py` to edit hyperparameters.

####Execution
To execute run
```bash
autosuggest.py path/to/model.pt.zip
```

####Dependencies
 - [PyTorch](https://pytorch.org/)
 - [npyscreen](https://pypi.org/project/npyscreen/)