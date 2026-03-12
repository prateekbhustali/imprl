import torch

from imprl.agents.primitives.MLP import NeuralNetwork


class MultiAgentCritics:
    """
    Critic networks for individual agents.

    """

    def __init__(self, num_agents, config, device) -> None:

        self.networks = []
        critic_params = []
        for _ in range(num_agents):
            critic_network = NeuralNetwork(
                config["architecture"], initialization="orthogonal"
            ).to(device)

            self.networks.append(critic_network)
            critic_params.extend(critic_network.parameters())

        # common optimizer
        self.optimizer = getattr(torch.optim, config["optimizer"])(
            critic_params, lr=config["lr"]
        )

        # common learning rate scheduler
        lr_scheduler = config["lr_scheduler"]
        self.lr_scheduler = getattr(
            torch.optim.lr_scheduler, lr_scheduler["scheduler"]
        )(self.optimizer, **lr_scheduler["kwargs"])

    def forward(self, t_observation, training, parallel=False) -> torch.Tensor:
        """
        Forward pass of the critic networks.

        Parameters
        ----------
        t_observation : torch.Tensor
            shape: (batch_size, n_damage_states + 1, n_components)

        Returns
        -------
        Value : torch.Tensor
            shape: (batch_size, n_components)

        """

        if parallel:
            futures = []
            for c, critic_network in enumerate(self.networks):
                # fork the forward pass of each critic network
                _output = torch.jit.fork(
                    critic_network.forward, t_observation[:, c, :], training
                )
                futures.append(_output)

            results = [torch.jit.wait(f) for f in futures]

            return torch.hstack(results)

        else:

            outputs = []
            for c, critic_network in enumerate(self.networks):
                _output = critic_network.forward(t_observation[:, c, :], training)
                outputs.append(_output)

            return torch.hstack(outputs)
