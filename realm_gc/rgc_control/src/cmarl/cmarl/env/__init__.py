from typing import Optional

from .base import MultiAgentEnv
from .single_integrator_cover import SingleIntegratorCover
from .single_integrator_cover_1d import SingleIntegratorCover1D
from .memorize_state import MemorizeState
from .single_int_nav import SingleIntNav
from .mpe_nav import MPENav
from .mpe_spread import MPESpread
from .mpe_line import MPELine
from .mpe_formation import MPEFormation
from .mpe_corridor import MPECorridor
from .mpe_connect_spread import MPEConnectSpread
from .lidar_spread import LidarSpread
from .lidar_nav import LidarNav
from .point_lidar_nav import PointLidarNav
from .lidar_line import LidarLine
from .lidar_f1tenth_target import LidarF1TenthTarget


ENV = {
    'SingleIntCover1D': SingleIntegratorCover1D,
    'SingleIntCover': SingleIntegratorCover,
    'MemorizeState': MemorizeState,
    'SingleIntNav': SingleIntNav,
    'MPENav': MPENav,
    'MPESpread': MPESpread,
    'MPELine': MPELine,
    'MPEFormation': MPEFormation,
    'MPECorridor': MPECorridor,
    'MPEConnectSpread': MPEConnectSpread,
    'LidarSpread': LidarSpread,
    'PointLidarNav': PointLidarNav,
    'LidarNav': LidarNav,
    'LidarLine': LidarLine,
    'LidarF1TenthTarget': LidarF1TenthTarget,
}


DEFAULT_MAX_STEP = 128


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        full_observation: bool = False,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        n_mov_obs: Optional[int] = None,
        delta_scale: Optional[float] = None,
        goal_reward_scale: Optional[float] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    if full_observation:
        area_size = params['default_area_size'] if area_size is None else area_size
        params['comm_radius'] = area_size * 10
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.03,
        params=params,
        n_mov_obs=n_mov_obs,
        delta_scale=delta_scale,
        goal_reward_scale=goal_reward_scale
    )
