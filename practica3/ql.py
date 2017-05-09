import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3
}

actions_rev = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

actions_vectors = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1)
}

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)

def getActionsNumber(state):
    y, x = getStateCoord(state)
    actions = []
    for a in getActions(state):
        actions.append(actions_list[a])
    return actions

def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

def getRndAction(state):
    return random.choice(getActions(state))

def getRndState():
    return random.randint(0, height * width - 1)

Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

#print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

def move(state, action):
    if type(state) != tuple:
        state = getStateCoord(state)

    y = state[0] + actions_vectors[action][0]
    x = state[1] + actions_vectors[action][1]
    return getState(y, x)

def explore(state):
    action = getRndAction(state)
    new_state = move(state, action)
    qlearning(state, actions_list[action], new_state)
    return new_state

def exploit(state):
    possible_actions = getActionsNumber(state)
    action = possible_actions[Q[state, possible_actions].argmax()]
    action = actions_rev[action]
    new_state = move(state, action)
    return new_state

class Policy:
    runs = 0
    steps = 0

    def _move(self, state):
        raise NotImplementedError('Policy is an abstract class that must be extended')

    def next_state(self, state):
        self.steps += 1
        return self._move(state)

    def new_run(self):
        self.runs += 1

    def get_average(self):
        return float(self.steps) / self.runs

    def reset(self):
        self.steps = 0
        self.runs = 0

    def __str__(self):
        s = str(self.__class__.__name__) + '\n'
        if hasattr(self, 'eps'):
            s += 'Eps: ' + str(self.eps) + '\n'
        s += 'Runs: ' + str(self.runs) + '\n'
        s += 'Avg:  ' + str(self.get_average())
        return s


class AlwaysExploitPolicy(Policy):
    def _move(self, state):
        return exploit(state)

class AlwaysExplorePolicy(Policy):
    def _move(self, state):
        return explore(state)

class GreedyPolicy(Policy):
    def _move(self, state):
        if np.max(Q[state]) <= 0:
            return explore(state)
        else:
            return exploit(state)


class EpsilonGreedyPolicy(Policy):

    eps = 0

    def __init__(self, eps):
        if eps < 0 or eps > 1:
            raise ValueError("Epsilon must be in range [0, 1]")

        self.eps = eps

    def _move(self, state):
        rnd = random.uniform(0, 1)
        if np.max(Q[state]) <= 0 or rnd > self.eps:
            return explore(state)
        else:
            return exploit(state)


def evaluate(strategy, generations=100):
    global Q
    Q = np.zeros((height * width, num_actions))  # Reset Q matrix
    for i in xrange(generations):
        strategy.new_run()
        state = getRndState()
        while state != final_state:
            state = strategy.next_state(state)

    print strategy
    print ''
    strategy.reset()




to_evaluate = [
    GreedyPolicy(),
    EpsilonGreedyPolicy(0.8),
    EpsilonGreedyPolicy(0.5),
    EpsilonGreedyPolicy(0.2),
    AlwaysExplorePolicy(),
]

for x in xrange(100, 1100, 200):
    for p in to_evaluate:
        evaluate(p, x)

#evaluate(GreedyPolicy())
# Q matrix plot

#print Q

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()
