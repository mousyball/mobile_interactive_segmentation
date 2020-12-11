from fvcore.common.config import CfgNode as CN

_C = CN()

_C.EVAL = CN(new_allowed=True)
_C.EVAL.MAX_CLICKS = 100
_C.EVAL.OUTPUT_PROBABILITY_THRESH = 0.5


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

