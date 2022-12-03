REGISTRY = {}

from .rnn_agent import RNNAgent
from .linda_agent import LINDAAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["linda"] = LINDAAgent