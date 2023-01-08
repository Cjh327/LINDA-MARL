REGISTRY = {}

from .rnn_agent import RNNAgent
from .linda_agent import LINDAAgent
from .linda_atten_agent import LINDAAttenAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["linda"] = LINDAAgent
REGISTRY["linda_atten"] = LINDAAttenAgent
