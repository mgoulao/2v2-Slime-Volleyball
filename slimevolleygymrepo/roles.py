from abc import abstractmethod, ABC
from math import sqrt


class Role(ABC):
    @abstractmethod
    # Available actions to role
    def actions(self):
        pass

    @abstractmethod
    # Calculate reward based on role
    def reward(self, *args):
        pass

    @abstractmethod
    # Potential function
    def potential(self, *args):
        pass

    @abstractmethod
    # In case agents need to switch roles
    def switch(self, agent):
        pass

    @abstractmethod
    # Info that can be communicated to teammate
    def info(self):
        pass


class Attacker(Role):
    def __init__(self):
        self.mirror = Defender()

    def actions(self):
        pass

    def reward(self, *args):
        px = args[0]
        py = args[1]
        nx = args[2]
        ny = args[3]
        bpx = args[4]
        bpy = args[5]
        bnx = args[6]
        bny = args[7]
        reward = args[8]

        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            return reward * -1
        else:
            return reward * 1.1

    # Distance to the ball
    def potential(self, *args):
        x = args[0]
        y = args[1]
        bx = args[2]
        by = args[3]

        return sqrt((x - bx) * (x - bx) + (y - by) * (y - by))

    def switch(self, agent):
        agent.role = self.mirror

    def info(self):
        pass


class Defender(Role):
    def __init__(self):
        self.mirror = Attacker()

    def actions(self):
        pass

    def reward(self, *args):
        px = args[0]
        py = args[1]
        nx = args[2]
        ny = args[3]
        bpx = args[4]
        bpy = args[5]
        bnx = args[6]
        bny = args[7]
        reward = args[8]

        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            return reward * 1.1
        else:
            return reward * -1

    # Distance to the ball
    def potential(self, *args):
        x = args[0]
        y = args[1]
        bx = args[2]
        by = args[3]

        return sqrt((x - bx) * (x - bx) + (y - by) * (y - by))

    def switch(self, agent):
        agent.role = self.mirror

    def info(self):
        pass
