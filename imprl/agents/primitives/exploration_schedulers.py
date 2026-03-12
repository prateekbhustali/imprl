import math


class LinearExplorationScheduler:

    def __init__(self, final_eps, num_episodes=None, rate=None, initial_eps=1) -> None:

        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.eps = initial_eps

        if rate is not None and num_episodes is None:
            self.rate = rate
        elif num_episodes is not None and rate is None:
            self.rate = (self.initial_eps - self.final_eps) / num_episodes
        elif rate is None and num_episodes is None:
            print("Neither rate nor num_episodes provided!")
        else:
            print("Only rate or num_episodes must be provided not both!")

    def step(self):
        self.eps -= self.rate

        return max(self.final_eps, self.eps)


class ExponentialExplorationScheduler:

    def __init__(self, final_eps, num_episodes=None, gamma=None, initial_eps=1) -> None:

        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.eps = initial_eps

        if gamma is not None and num_episodes is None:
            self.gamma = gamma
        elif num_episodes is not None and gamma is None:
            ln_gamma = (
                math.log(self.final_eps) - math.log(self.initial_eps)
            ) / num_episodes
            self.gamma = math.exp(ln_gamma)
        elif gamma is None and num_episodes is None:
            print("Neither rate nor num_episodes provided!")
        else:
            print("Only rate or num_episodes must be provided not both!")

    def step(self):
        self.eps *= self.gamma

        assert self.eps >= self.final_eps, "exploration param lower than min value!"
