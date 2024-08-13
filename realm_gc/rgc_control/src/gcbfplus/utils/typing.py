from flax import core, struct
# from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, TypeVar, Any, List
from numpy import ndarray


# jax types
PRNGKey = float

BoolScalar = bool
ABool = bool

# environment types
Action = float
Reward = float
Cost = float
Done = BoolScalar
Info = dict
EdgeIndex = float
AgentState = float
State = float
Node = float
EdgeAttr = float
Pos2d = float
Pos3d = float
Pos = Pos2d
Radius = float
Array = float
Bool = bool
Float = float
Int = int
# neural network types
Params = TypeVar("Params", bound=core.FrozenDict[str, Any])

# obstacles
ObsType = int
ObsWidth = float
ObsHeight = float
ObsLength = float
ObsTheta = float
ObsQuaternion = float
