REGISTRY = {}

from .basic_controller import BasicMAC
from .linda_controller import LINDAMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["linda_mac"] = LINDAMAC