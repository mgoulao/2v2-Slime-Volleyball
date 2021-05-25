from abc import abstractmethod, ABC
from math import sqrt
from game_settings import REF_W

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
    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return (threshold - self.potential(x1, y1, x2, y2)) / threshold

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
            # If ball is getting closer to teammate, reward should be higher for standing still
            d1 = self.__step(REF_W/2, tpx, tpy, bnx, bny)
            d2 = self.__step(REF_W/2, tnx, tny, bnx, bny)
            if d1 < d2 < 0.2:
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
            return reward * (1 - step)

    def switch(self, agent):
        agent.role = Defender()

    def decide(self, *args):
        agent = args[0]
        state = args[1]
        teammate = args[2]
        agent_pos = [state[0], state[1]]
        ball_pos = [state[8], state[9]]
        teammate_pos = [state[4], state[5]]
        teammate_info  = teammate.role.info(teammate_pos, ball_pos)

        if self.info(agent_pos, ball_pos) > teammate_info:
            self.switch(agent)


class Defender(AD):
    def actions(self):
        pass

    def __step(self, threshold, x1, y1, x2, y2):
        return (threshold - self.potential(x1, y1, x2, y2)) / threshold

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
        if self.potential(px, py, bpx, bpy) > self.potential(nx, ny, bnx, bny):
            reward = reward * 1.5
        else:
            # If ball is moving away from teammate, reward should be higher for standing still
            d1 = self.__step(REF_W / 2, tpx, tpy, bnx, bny)
            d2 = self.__step(REF_W / 2, tnx, tny, bnx, bny)
            if d1 > d2 >= 0.2:
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
            return reward * (1 - step)

    def switch(self, agent):
        del agent.role
        agent.role = Attacker()

    def decide(self, *args):
        agent = args[0]
        state = args[1]
        teammate = args[2]
        agent_pos = [state[0], state[1]]
        ball_pos = [state[8], state[9]]
        teammate_pos = [state[4], state[5]]
        teammate_info  = teammate.role.info(teammate_pos, ball_pos)

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
        reward = args[2]

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
            reward = reward * 1.5
        else:
            reward = reward * 0.5

        # Punish agent if it touches the ball
        if ndist2 < 4:
            return reward * 0.2 
        else:
            return reward

    def switch(self, agent):
        agent.role = Top()


class Top(TB):
    def actions(self):
        pass

    def reward(self, *args):
        prev_state = args[0]
        state = args[1]
        reward = args[2]

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
                return reward * 1.5
        
        # Agent was not on top of teammate
        else:         
            if ndist2_b == 4: # Agent touches ball (distance = 1.5 + 0.5), Punish +
                return reward*0.3
            
            if pdist2_a > ndist2_a: # Agent is getting closer to teammate
                if nx == tnx: # Agent achieved desired position
                    return reward*1.5
                return reward*1.2
            
            # Punish if agent is getting further away from teammate
            else:
                return reward * 0.5

    def switch(self, agent):
        del agent.role
        agent.role = Bottom()