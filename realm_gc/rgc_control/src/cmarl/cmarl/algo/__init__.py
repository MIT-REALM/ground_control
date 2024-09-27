from .base import Algorithm
from .informarl import InforMARL
from .efinformarl import EFInforMARL
from .informarl_crpo import InformarlCRPO
from .informarl_lagr import InforMARLLagr
from .informarl_cpo import InforMARLCPO
from .efmarl import EFMARL
from .gcbfmarl import GCBFMARL
from .gcbfcrpo import GCBFCRPO


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        return InforMARL(**kwargs)
    elif algo == 'efinformarl':
        return EFInforMARL(**kwargs)
    elif algo == 'efmarl':
        return EFMARL(**kwargs)
    elif algo == 'gcbfmarl':
        return GCBFMARL(**kwargs)
    elif algo == 'informarl_crpo':
        return InformarlCRPO(**kwargs)
    elif algo == 'informarl_lagr':
        return InforMARLLagr(**kwargs)
    elif algo == 'gcbfcrpo':
        return GCBFCRPO(**kwargs)
    elif algo == 'informarl_cpo':
        return InforMARLCPO(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
