from abc import abstractmethod, ABC
from math import sqrt
from slimevolleygym.game_settings import REF_W

SCALED_REF_W = REF_W / 10

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
    def potential(self, *args):
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
        agent_x = args[0][0]
        agent_y = args[0][1]
        ball_x = args[1][0]
        ball_y = args[1][1]

        return self.potential(agent_x, agent_y, ball_x, ball_y)

    @abstractmethod
    def decide(self, *args):
        pass


class Attacker(AD):
    def name(self):
        return "attacker"

    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return self.potential(x1, y1, x2, y2) / threshold

    def reward(self, *args):
        prev_state = args[0]
        state = args[1]
        reward = args[2]
        # previous state
        px = prev_state[0]
        py = prev_state[1]
        # # next state
        nx = state[0]
        ny = state[1]
        # # ball previous state
        bpx = prev_state[8]
        bpy = prev_state[9]
        # # ball next state
        bnx = state[8]
        bny = state[9]
        # # ppo reward
        # # teammates previous state
        tpx = prev_state[4]
        tpy = prev_state[5]
        # # teammates next state
        tnx = state[4]
        tny = state[5]

        # Reward agents if the actions are according to role TODO: Check github
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            reward = (reward + 0.3) * 3

        # If ball is getting closer to teammate, reward should be higher for going away from the ball
        dt = self.__step(sqrt(2) * SCALED_REF_W/4, tnx, tny, bnx, bny)
        db = self.__step(SCALED_REF_W, tnx, 0, bnx, 0)
        if dt < 0.5 and db <= 0.49:
            if self.potential(px, py, bpx, bpy) < self.potential(nx, py, bnx, bny):
                reward = (reward + 0.1) * 1.5
            else:
                reward = reward * -0.1
        else:
            reward = reward

        # Reward agents if theyre not close together
        ad = self.__step(SCALED_REF_W/2, nx, 0, tpx, 0)
        if 1/4 < ad < 3/4:
            return reward * 1.2
        else:
            if ad <= 0.5:
                step = ad + 0.5
            else:
                step = 1 - ad + 0.5
            return reward * max(0.3, step)

    def switch(self, agent):
        agent.role = Defender()

    def decide(self, *args):
        agent = args[0]
        state = args[1]
        teammate = args[2]
        agent_pos = [state[0], state[1]]
        ball_pos = [state[8], state[9]]
        teammate_pos = [state[4], state[5]]
        teammate_info = teammate.role.info(teammate_pos, ball_pos)

        if self.info(agent_pos, ball_pos) > teammate_info:
            self.switch(agent)


class Defender(AD):
    def name(self):
        return "defender"
    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return self.potential(x1, y1, x2, y2) / threshold

    def reward(self, *args):
        prev_state = args[0]
        state = args[1]
        reward = args[2]
        # previous state
        px = prev_state[0]
        py = prev_state[1]
        # # next state
        nx = state[0]
        ny = state[1]
        # # ball previous state
        bpx = prev_state[8]
        bpy = prev_state[9]
        # # ball next state
        bnx = state[8]
        bny = state[9]
        # # ppo reward
        # # teammates previous state
        tpx = prev_state[4]
        tpy = prev_state[5]
        # # teammates next state
        tnx = state[4]
        tny = state[5]

        # Reward agents if the actions are according to role
        if self.potential(px, py, bpx, bpy) < self.potential(nx, ny, bnx, bny):
            reward = reward + 0.01

        # If ball is moving away from teammate, reward should be higher for moving closer to ball
        dt = self.__step(sqrt(2) * SCALED_REF_W/4, tnx, tny, bnx, bny)
        db = self.__step(SCALED_REF_W, tnx, 0, bnx, 0)
        if dt >= 0.5 and db <= 0.49:
            if self.potential(px, py, bpx, bpy) > self.potential(nx, py, bnx, bny):
                reward = (reward + 0.1) * 1.5
            else:
                reward = reward * -0.1
        else:
            reward = reward

        # Reward agents if theyre not close together
        ad = self.__step(SCALED_REF_W/2, nx, 0, tpx, 0)
        if 1/4 < ad < 3/4:
            return reward * 1.2
        else:
            if ad <= 0.5:
                step = ad + 0.5
            else:
                step = 1 - ad + 0.5
            return reward * max(0.3, step)

    def switch(self, agent):
        agent.role = Attacker()

    def decide(self, *args):
        agent = args[0]
        state = args[1]
        teammate = args[2]
        agent_pos = [state[0], state[1]]
        ball_pos = [state[8], state[9]]
        teammate_pos = [state[4], state[5]]
        teammate_info = teammate.role.info(teammate_pos, ball_pos)

        if self.info(agent_pos, ball_pos) <= teammate_info:
            self.switch(agent)

# -------- Floor is Lava -------- #
class TB(Role):
    @abstractmethod
    def actions(self):
        pass

    @abstractmethod
    def reward(self, *args):
        pass

    def potential(self, x, y, ox, oy):
        dx = ox - x
        dy = oy - y
        return dx*dx+dy*dy

    @abstractmethod
    def switch(self, agent):
        pass

    #@abstractmethod
    def info(self, *args):
        pass

    #@abstractmethod
    def decide(self, *args):
        pass

class Bottom(TB):
    def actions(self):
        pass

    def reward(self, *args):
        prev_state = args[0]
        state = args[1]
        reward = args[2] + 0.3

        # previous state
        px = prev_state[0]
        py = prev_state[1]
        # next state
        nx = state[0]
        ny = state[1]
        # ball previous state
        bpx = prev_state[8]
        bpy = prev_state[9]
        # ball next state
        bnx = state[8]
        bny = state[9]

        # Distance between Bottom and Ball
        pdist2 = self.potential(px, py, bpx, bpy)
        ndist2 = self.potential(nx, ny, bnx, bny)

        # Reward agent if it moved closer to the ball
        if pdist2 > ndist2:
            return reward * 1.5
        else:
           return reward * (-0.5)

        # Punish if agent jumps
        # if py < ny:
        #     return reward * (-0.8)
        # else:
        #     return reward

    def switch(self, agent):
        agent.role = Top()


class Top(TB):
    def actions(self):
        pass

    def reward(self, *args):
        prev_state = args[0]
        state = args[1]
        reward = args[2] + 0.3

        # previous state
        px = prev_state[0]
        py = prev_state[1]
        # next state
        nx = state[0]
        ny = state[1]
        # ball previous state
        bpx = prev_state[8]
        bpy = prev_state[9]
        # ball next state
        bnx = state[8]
        bny = state[9]
        # teammates previous state
        tpx = prev_state[4]
        tpy = prev_state[5]
        # teammates next state
        tnx = state[4]
        tny = state[5]

        # distance between Top and Bottom
        pdist2_a = self.potential(px, py, tpx, tpy)
        ndist2_a = self.potential(nx, ny, tnx, tny)
        # distance between Top and ball
        pdist2_b = self.potential(px, py, bpx, bpy)
        ndist2_b = self.potential(nx, ny, bnx, bny)

        # Agent was already on top of teammate (px=tpx)
        if px == tpx:
            if nx == tnx: # Reward + for remaining
                return reward * 1.8
            else:
                return reward * (-0.5)
        
        # Agent was not on top of teammate
        else:         
            if ndist2_b == 4: # Agent touches ball (distance = 1.5 + 0.5), Punish +
                reward = reward * (-0.8)
               
            if pdist2_a > ndist2_a: # Agent is getting closer to teammate
                return reward * 1.2
            
            # Punish if agent is getting further away from teammate
            else:
                return reward * (-0.5)

    def switch(self, agent):
        del agent.role
        agent.role = Bottom()