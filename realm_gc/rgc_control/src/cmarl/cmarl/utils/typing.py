from flax import core, struct
from flax.core import FrozenDict
from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, TypeVar, Any, List
from numpy import ndarray
from typing import NamedTuple


# jax types
PRNGKey = Float[Array, '2']

BoolScalar = Bool[Array, ""]
ABool = Bool[Array, "num_agents"]
Shape = tuple[int, ...]

BFloat = Float[Array, "b"]
BInt = Int[Array, "b"]
FloatScalar = float
IntScalar = int 
TFloat = Float[Array, "T"]

# environment types
Action = Float[Array, 'num_agents action_dim']
Reward = Float[Array, '']
Cost = Float[Array, 'nh']
Done = BoolScalar
Info = Dict[str, Shaped[Array, '']]
EdgeIndex = Float[Array, '2 n_edge']
AgentState = Float[Array, 'num_agents agent_state_dim']
State = Float[Array, 'num_states state_dim'] 
Node = Float[Array, 'num_nodes node_dim']
EdgeAttr = Float[Array, 'num_edges edge_dim']
Pos2d = Float[Array, '2'] 
Pos3d = Float[Array, '3'] 
Pos = Pos2d 
Radius = Float[Array, ''] 


# neural network types
Params = dict[str, Any] 

# obstacles
ObsType = Int[Array, '']
ObsWidth = Float[Array, '']
ObsHeight = Float[Array, '']
ObsLength = Float[Array, '']
ObsTheta = Float[Array, '']
ObsQuaternion = Float[Array, '4']
