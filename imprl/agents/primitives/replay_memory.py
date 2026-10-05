import random
from collections import deque


class AbstractReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    def store_experience(self, *args):
        self.memory.append(args)

    def sample_batch(self, batch_size):
        """Randomly sample batch_size experiences from the memory"""

        experiences = random.sample(self.memory, batch_size)

        return self._create_lists(experiences)

    def _create_lists(self, experiences):

        num_elements = len(experiences[0])

        # create an empty list for each element of an experience
        # for example: obs, action, reward, next_obs, done
        lists = [[] for _ in range(num_elements)]

        # loop over all experiences
        for experience in experiences:
            # loop over element of an experience
            for element in range(num_elements):
                # store element in corresponding list
                lists[element].append(experience[element])

        return lists


class EpisodicReplayMemory(AbstractReplayMemory):

    def __init__(self, size):
        super().__init__(size)

        self.experience = []

    def store_experience(self, *args):

        self.experience.append(args)

        # check if the episode is complete
        if args[-1]:  # done
            self.memory.append(self.experience)
            self.experience = []

    def sample_batch(self, batch_size):
        episodes = random.sample(self.memory, batch_size)
        return self.create_batch(episodes)

    def create_batch(self, episodes):
        experiences = [experience for episode in episodes for experience in episode]
        return self._create_lists(experiences)

    def create_dataset(self, size, ratios=[1]):

        all_episodes = random.sample(self.memory, size)

        split_size = [int(size * ratio) for ratio in ratios]
        split_episodes = [[] for _ in range(len(ratios))]

        for i, split in enumerate(split_size):
            start = sum(split_size[:i])
            end = start + split
            for episode in all_episodes[start:end]:
                split_episodes[i].append(episode)

        return split_episodes
