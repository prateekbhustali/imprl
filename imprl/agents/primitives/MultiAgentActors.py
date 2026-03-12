import torch

from imprl.agents.primitives.ActorNetwork import ActorNetwork


class MultiAgentActors:

    def __init__(self, num_agents, num_actions, config, device) -> None:

        self.networks = []
        actor_params = []
        for _ in range(num_agents):
            actor_network = ActorNetwork(
                config["architecture"], initialization="orthogonal"
            ).to(device)
            self.networks.append(actor_network)
            actor_params.extend(actor_network.parameters())

        # common optimizer
        self.optimizer = getattr(torch.optim, config["optimizer"])(
            actor_params, lr=config["lr"]
        )

        # common learning rate scheduler
        lr_scheduler = config["lr_scheduler"]
        self.lr_scheduler = getattr(
            torch.optim.lr_scheduler, lr_scheduler["scheduler"]
        )(self.optimizer, **lr_scheduler["kwargs"])

    def forward(self, t_observation, training, ind_obs=False, parallel=False):
        """
        Parallel forward is beneficial when the number of agents is large.

        """

        #################### Parallel ########################

        if parallel:

            futures = []

            if training:
                for a, actor_network in enumerate(self.networks):
                    if ind_obs:
                        _output = torch.jit.fork(
                            actor_network._get_sample_action,
                            t_observation[:, a, :],
                            log_prob=True,
                        )
                    else:
                        _output = torch.jit.fork(
                            actor_network._get_sample_action,
                            t_observation,
                            log_prob=True,
                        )
                    futures.append(_output)

                # list of tuples of tensors
                results = [torch.jit.wait(f) for f in futures]

                # Transpose the list of tuples into a tuple of lists
                _results = tuple(zip(*results))

                t_action, log_prob = tuple(torch.stack(t) for t in _results)

                # Combine actions of all agents
                action_prob = torch.prod(torch.exp(log_prob), dim=0)

                return t_action.cpu().numpy(), t_action, action_prob

            else:
                for a, actor_network in enumerate(self.networks):
                    if ind_obs:
                        _output = torch.jit.fork(
                            actor_network._get_sample_action,
                            t_observation[:, a, :],
                            log_prob=False,
                        )
                    else:
                        _output = torch.jit.fork(
                            actor_network._get_sample_action,
                            t_observation,
                            log_prob=False,
                        )
                    futures.append(_output)

                # list of tensors
                results = [torch.jit.wait(f) for f in futures]

                return torch.hstack(results).detach().numpy()

        #################### Series ########################
        else:

            if training:

                _list_t_action = []
                _list_log_prob = []

                for a, actor_network in enumerate(self.networks):

                    if ind_obs:
                        _t_action, _log_prob = actor_network._get_sample_action(
                            t_observation[:, a, :], log_prob=True
                        )
                    else:
                        _t_action, _log_prob = actor_network._get_sample_action(
                            t_observation, log_prob=True
                        )

                    _list_t_action.append(_t_action)
                    _list_log_prob.append(_log_prob)

                t_action = torch.stack(_list_t_action)
                log_prob = torch.stack(_list_log_prob)

                # Combine actions of all agents
                action_prob = torch.prod(torch.exp(log_prob), dim=0)

                return t_action.cpu().numpy(), t_action, action_prob

            else:

                _list_t_action = []

                for a, actor_network in enumerate(self.networks):

                    if ind_obs:
                        _t_action = actor_network._get_sample_action(
                            t_observation[:, a, :], log_prob=False
                        )
                    else:
                        _t_action = actor_network._get_sample_action(
                            t_observation, log_prob=False
                        )

                    _list_t_action.append(_t_action)

                return torch.stack(_list_t_action).detach().numpy()
