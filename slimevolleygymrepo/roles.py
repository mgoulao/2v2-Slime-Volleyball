from abc import abstractmethod, ABC
from math import sqrt
from slimevolleygym.slimevolley import REF_W

class Role(ABC):
    @abstractmethod
    # Available actions to role
    def actions(self):
        pass

    @abstractmethod
    # Calculate reward based on role, should only be used during training
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
    def info(self, *args):
        pass

    @abstractmethod
    # Based on teammate info, decide role
    def decide(self, *args):
        pass


# Same as no role
class Vanilla(Role):
    def actions(self):
        pass

    def reward(self, *args):
        return args[8]

    def potential(self, *args):
        return 0

    def switch(self, agent):
        pass

    def info(self, *args):
        return 0

    def decide(self, *args):
        pass


class AD(Role):
    @abstractmethod
    def actions(self):
        pass

    @abstractmethod
    def reward(self, *args):
        pass

    # Distance to object
    def potential(*args):
        x = args[0]
        y = args[1]
        bx = args[2]
        by = args[3]

        return sqrt((x - bx) * (x - bx) + (y - by) * (y - by))

    @abstractmethod
    def switch(self, agent):
        pass

    # How far away the agent is from the ball
    def info(self, *args):
        agent = args[0]
        ball = args[1]

        return self.potential(agent.x, agent.y, ball.x, ball.y)

    @abstractmethod
    def decide(self, *args):
        pass


class Attacker(AD):
    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return (threshold - self.potential(x1, y1, x2, y2)) / threshold

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
        # teammates past state
        tpx = args[9]
        tpy = args[10]
        # teammates past state
        tnx = args[11]
        tny = args[12]

        # Reward agents if the actions are according to role
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            # If ball is getting closer to teammate, reward should be higher for standing still
            d1 = self.__step(REF_W/2, tpx, tpy, bnx, bny)
            d2 = self.__step(REF_W/2, tnx, tny, bnx, bny)
            if d2 < d1 and d2 < 0.2:
                if px == nx and py == ny:
                    reward = reward * 1.2
                else:
                    reward = reward * 0.5
            else:
                reward = reward
        else:
            reward = reward * 1.5

        # Reward agents if theyre not close together
        if self.potential(nx, ny, tpx, tpy) > REF_W/6:
            return reward
        else:
            step = self.__step(REF_W/6, nx, ny, tpx, tpy)
            return reward * step

    def switch(self, agent):
        agent.role = Defender()

    def decide(self, *args):
        agent = args[0]
        ball = args[1]
        info = args[2]

        if self.info(agent, ball) >= info:
            self.switch(agent)


class Defender(AD):
    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return (threshold - self.potential(x1, y1, x2, y2)) / threshold

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
        tpx = args[9]
        tpy = args[10]
        # teammates next state
        tnx = args[11]
        tny = args[12]

        # Reward agents if the actions are according to role
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            reward = reward * 1.5
        else:
            # If ball is moving away from teammate, reward should be higher for standing still
            d1 = self.__step(REF_W / 2, tpx, tpy, bnx, bny)
            d2 = self.__step(REF_W / 2, tnx, tny, bnx, bny)
            if d2 > d1 and d2 > 0.2:
                if px == nx and py == ny:
                    reward = reward * 1.2
                else:
                    reward = reward * 0.5
            else:
                reward = reward

        # Reward agents if theyre not close together
        if self.potential(nx, ny, tpx, tpy) > REF_W / 6:
            return reward
        else:
            step = self.__step(REF_W/6, nx, ny, tpx, tpy)
            return reward * step

    def switch(self, agent):
        agent.role = Attacker()

    def decide(self, *args):
        agent = args[0]
        ball = args[1]
        info = args[2]

        if self.info(agent, ball) <= info:
            self.switch(agent)
