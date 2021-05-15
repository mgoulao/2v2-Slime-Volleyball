from abc import abstractmethod, ABC
from math import sqrt
from slimevolleygym.slimevolley import REF_W

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


# Same as no role
class Vanilla(Role):
    def actions(self):
        pass

    def reward(self, *args):
        pass

    def potential(self, *args):
        pass

    def switch(self, agent):
        pass

    def info(self):
        pass


class Attacker(Role):
    def actions(self):
        pass

    def reward(self, *args):
        # previous state
        px = args[0]
        py = args[1]
        # next state
        nx = args[2]
        ny = args[3]
        # ball previous state
        bpx = args[4]
        bpy = args[5]
        # ball next state
        bnx = args[6]
        bny = args[7]
        # ppo reward
        reward = args[8]
        # teammates next state
        tx = args[9]
        ty = args[10]

        # Reward agents if the actions are according to role
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            reward = reward * -1
        else:
            reward = reward * 1.1

        # Reward agents if theyre not close together
        if self.potential(nx, ny, tx, ty) > REF_W/6:
            return reward
        else:
            step = (REF_W/6 - self.potential(nx, ny, tx, ty)) / (REF_W/6)
            if reward < 0:
                step = 1 + step
            return reward * step

    # Distance to the ball
    def potential(self, *args):
        x = args[0]
        y = args[1]
        bx = args[2]
        by = args[3]

        return sqrt((x - bx) * (x - bx) + (y - by) * (y - by))

    def switch(self, agent):
        agent.role = Defender()

    def info(self):
        pass


class Defender(Role):
    def actions(self):
        pass

    def reward(self, *args):
        # previous state
        px = args[0]
        py = args[1]
        # next state
        nx = args[2]
        ny = args[3]
        # ball previous state
        bpx = args[4]
        bpy = args[5]
        # ball next state
        bnx = args[6]
        bny = args[7]
        # ppo reward
        reward = args[8]
        # teammates next state
        px = args[9]
        py = args[10]
        # teammates next state
        tx = args[9]
        ty = args[10]

        # Reward agents if the actions are according to role
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            reward = reward * 1.1
        else:
            reward = reward * -1

        # Reward agents if theyre not close together
        if self.potential(nx, ny, tx, ty) > REF_W / 6:
            return reward
        else:
            step = (REF_W / 6 - self.potential(nx, ny, tx, ty)) / (REF_W / 6)
            if reward < 0:
                step = 1 + step
            return reward * step

    # Distance to the ball
    def potential(self, *args):
        x = args[0]
        y = args[1]
        bx = args[2]
        by = args[3]

        return sqrt((x - bx) * (x - bx) + (y - by) * (y - by))

    def switch(self, agent):
        agent.role = Attacker()

    def info(self):
        pass
