from typing import Optional

from .base import MultiAgentEnv
from .single_integrator import SingleIntegrator
from .single_integrator_adapt import SingleIntegratorAdapt
from .single_integrator_adapt_edge import SingleIntegratorAdaptEdge
from .double_integrator import DoubleIntegrator
from .linear_drone import LinearDrone
from .dubins_car import DubinsCar
from .dubins_car_adapt import DubinsCarAdapt
from .crazyflie import CrazyFlie


ENV = {
    'SingleIntegrator': SingleIntegrator,
    'SingleIntegratorAdapt': SingleIntegratorAdapt,
    'SingleIntegratorAdaptEdge': SingleIntegratorAdaptEdge,
    'DoubleIntegrator': DoubleIntegrator,
    'LinearDrone': LinearDrone,
    'DubinsCar': DubinsCar,
    'DubinsCarAdapt': DubinsCarAdapt,
    'CrazyFlie': CrazyFlie,
}

DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        num_mov_obs: Optional[int] = None,
        mov_obs_speed: Optional[float] = None,
        n_rays: Optional[int] = None,
        node_feat: Optional[int] = None,
        mov_obs_at_infty: Optional[bool] = False,
        station_obs_at_infty: Optional[bool] = False,
        use_stop_mask: Optional[bool] = False,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    if num_mov_obs is not None:
        params['n_mov_obs'] = num_mov_obs
    if mov_obs_speed is not None:
        params['mov_obs_speed'] = mov_obs_speed
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.03,
        node_feat=node_feat,
        mov_obs_speed=mov_obs_speed,
        mov_obs_at_infty=mov_obs_at_infty,
        station_obs_at_infty=station_obs_at_infty,
        params=params,
        use_stop_mask=use_stop_mask,
    )
