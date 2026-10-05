from imprl.agents.DDQN import DDQNAgent as DDQN
from imprl.agents.JAC import JointActorCritic as JAC
from imprl.agents.DCMAC import DeepCentralisedMultiAgentActorCritic as DCMAC
from imprl.agents.DDMAC import DeepDecentralisedMultiAgentActorCritic as DDMAC
from imprl.agents.IACC import IndependentActorCentralisedCritic as IACC
from imprl.agents.IAC import IndependentActorCritic as IAC
from imprl.agents.IAC_PS import IndependentActorCriticParameterSharing as IAC_PS
from imprl.agents.IACC_PS import (
    IndependentActorCentralisedCriticParameterSharing as IACC_PS,
)
from imprl.agents.VDN_PS import ValueDecompositionNetworkParameterSharing as VDN_PS
from imprl.agents.QMIX_PS import QMIXParameterSharing as QMIX_PS
from imprl.agents.PPO import ProximalPolicyOptimization as PPO
from imprl.agents.IPPO_PS import (
    IndependentProximalPolicyOptimizationParameterSharing as IPPO_PS,
)
from imprl.agents.MAPPO_PS import (
    MultiAgentProximalPolicyOptimizationParameterSharing as MAPPO_PS,
)
from imprl.agents.SARSOP import SARSOPAgent as SARSOP


_AGENT_CLASSES = {
    "SARSOP": SARSOP,
    "DDQN": DDQN,
    "JAC": JAC,
    "PPO": PPO,
    "DCMAC": DCMAC,
    "DDMAC": DDMAC,
    "IACC": IACC,
    "IACC_PS": IACC_PS,
    "VDN_PS": VDN_PS,
    "QMIX_PS": QMIX_PS,
    "MAPPO_PS": MAPPO_PS,
    "IPPO_PS": IPPO_PS,
    "IAC": IAC,
    "IAC_PS": IAC_PS,
}

REGISTRY = {
    name: {
        "agent_class": cls,
        **{
            attr_name: attr_value
            for attr_name, attr_value in vars(cls).items()
            if not attr_name.startswith("_") and not callable(attr_value)
        },
    }
    for name, cls in _AGENT_CLASSES.items()
}


def make(algorithm, *agent_args, **agent_kwargs):
    """Instantiate an agent by algorithm name."""
    if algorithm not in REGISTRY:
        raise NotImplementedError(f"The algorithm '{algorithm}' is not implemented.")

    return REGISTRY[algorithm]["agent_class"](*agent_args, **agent_kwargs)
