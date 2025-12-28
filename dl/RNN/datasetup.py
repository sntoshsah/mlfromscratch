import numpy as np


# Data Setup
chars = ['h', 'e', 'l', 'o']
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
vocab_size = len(chars)

# Input: "hell", Target: "ello"
inputs_str = "hell"
targets_str = "ello"
inputs = [np.eye(4)[char_to_ix[ch]] for ch in inputs_str]
target_indices = [char_to_ix[ch] for ch in targets_str]
