import torch

from imprl.agents.primitives.MLP import NeuralNetwork


class ActorNetwork(NeuralNetwork):
    """
    A class to create an actor network for policy gradients.
    It simply adds a softmax head to the NeuralNetwork class.
    """

    def __init__(
        self,
        architecture,
        reshaping=None,
        activation="relu",
        initialization="orthogonal",
        optimizer=None,
        learning_rate=None,
        lr_scheduler=None,
    ) -> None:
        super().__init__(
            architecture,
            activation=activation,
            initialization=initialization,
            loss="cross_entropy",
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
        )

        # shape: (n_components, n_actions)
        # shape: (4, 3)
        # probs = np.array([[0.1, 0.2, 0.8],
        #                   [0.5, 0.3, 0.2],
        #                   [0.8, 0.1, 0.1],
        #                   [0.0, 0.0, 1.0]])
        # dist = Categorical(torch.tensor(probs))
        # dist.sample()
        # >>> tensor([2, 0, 0, 2])
        # self.new_shape = (-1, n_components, n_actions)
        self.reshaping = reshaping

        self.Categorical = torch.distributions.categorical.Categorical

    def forward(self, x, training=True):
        """
        Parameters
        ----------
        x : tensor
            shape: (n_samples, n_inputs)
        Returns
        -------
        dist: torch dist object
              to sample use: sample()
                             shape: (n_samples, n_components)
        """

        # call forward of parent (NeuralNetwork) class
        logits = super().forward(x, training=training)

        if self.reshaping is not None:
            dist = self.Categorical(logits=logits.view(-1, *self.reshaping))
        else:
            dist = self.Categorical(logits=logits.squeeze())

        return dist

    def _get_sample_action(self, x, log_prob=False, training=True):

        dist = self.forward(x, training=training)
        action = dist.sample().squeeze()

        if log_prob:
            return action, dist.log_prob(action)

        return action
