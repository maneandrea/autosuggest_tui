import os

small_ver = os.environ.get('SMALL_GPT', None)

CONTEXT_SIZE = 16 if small_ver else 256      # Context length on which the model predicts
N_SUGGESTIONS = 4                            # Number of suggestions shown
MAX_DEPTH = 100                              # Depth of the beam search
N_HEADS = 4 if small_ver else 6              # Number of heads in the self-attention (must divide EMBEDDING_DIM)
VALUE_DIM = None                             # Dimension of the value vector (if none defaults to head_dim)
DROPOUT = 0.2                                # Dropout probability
N_ITER = 100                                 # Number of iterations in the validation evaluation
EVAL_ITER = 10                               # Number of steps every which to trigger an evaluation
LEARNING_RATE = 3e-4                         # Step of the parameters in the optimizer
BATCH_SIZE = 64                              # Numbers of examples to evaluate in parallel
EPOCHS = 4                                   # Number of epochs to run
FRACTION = 0.2                               # Fraction of evaluation vs train data
N_BLOCKS = 4 if small_ver else 6             # Number of transformer blocks
EMBEDDING_DIM = 100 if small_ver else 384    # Dimension of the embedding space and model internal dimension
HIDDEN_FACTOR = 4                            # Factor of the dimension of hidden layer in the feed forward network
