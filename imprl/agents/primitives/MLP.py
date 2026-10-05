import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A class to create a fully connected neural network
    """

    def __init__(
        self,
        architecture,
        activation="relu",
        initialization="orthogonal",
        loss="mse",
        optimizer=None,
        learning_rate=None,
        lr_scheduler=None,
        dropout_prob=0.0,
    ):
        super().__init__()

        self.initialization = initialization
        self.activation_name = activation
        self.dropout = torch.nn.Dropout(dropout_prob)

        # activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

        # neural network
        self.linears = nn.ModuleList()
        for i in range(len(architecture) - 1):
            layer = nn.Linear(architecture[i], architecture[i + 1])
            self.linears.append(layer)

        # initialization
        for layer in self.linears:

            # weights
            gain = torch.nn.init.calculate_gain(self.activation_name)
            if self.initialization == "xavier":
                nn.init.xavier_normal_(layer.weight.data, gain=gain)
            elif self.initialization == "orthogonal":
                nn.init.orthogonal_(layer.weight.data, gain=gain)

            # biases
            nn.init.zeros_(layer.bias.data)

        # loss function
        if loss == "mse":
            self.loss_function = nn.MSELoss(reduction="mean")
        elif loss == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="mean")

        # optimizer
        if optimizer is not None:
            self.optimizer = getattr(torch.optim, optimizer)(
                self.parameters(), lr=learning_rate
            )

        # learning rate scheduler
        if lr_scheduler is not None:

            # create scheduler class
            self.lr_scheduler = getattr(
                torch.optim.lr_scheduler, lr_scheduler["scheduler"]
            )(self.optimizer, **lr_scheduler["kwargs"])

    def forward(self, x, training=True):

        a = x
        with torch.inference_mode(not training):
            for i in range(len(self.linears) - 1):
                z = self.linears[i](a)
                a = self.activation(z)
                a = self.dropout(a) if training else a
            a = self.linears[-1](a)

        return a
