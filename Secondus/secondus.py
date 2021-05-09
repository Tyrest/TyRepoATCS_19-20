import numpy as np
import time
import random
import os
from collections                    import deque

from PIL                            import Image
import cv2

from Helpers                        import get_screen, isalive, reshape_screen
from Input                          import press_LB, release_LB, restart

import gym
from tensorflow.keras.models        import Sequential
from tensorflow.keras.layers        import Dense, Conv2D, Flatten, MaxPooling2D, Activation
from tensorflow.keras.optimizers    import Adam, RMSprop

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "gd_weights.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=1000000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95  #0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.999
        self.brain              = self._build_model()

    # Change this to a CNN to better handle the screen shots for Atari games
    # Starting with CNN described by DeepMind's first paper
    def _build_model(self):
        act = 'elu'
        opt = RMSprop()

        model = Sequential()
        model.add(Conv2D(32,(9,9),strides=2, input_shape=(410,470,1)))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64,(5,5),strides=2))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128,(3,3),strides=2))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256,(3,3)))
        model.add(Activation(act))

        model.add(Conv2D(512,(3,3)))
        model.add(Activation(act))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(act))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=opt)
        model.summary()

        if os.path.isfile(self.weight_backup):
            print("Loading weights...")
            model.load_weights(self.weight_backup)
<<<<<<< HEAD
            self.exploration_rate = self.exploration_decay ** (96 + 736 + 160 + 1328 + 128)
=======
            self.exploration_rate = self.exploration_decay ** (96 + 736 + 160 + 1328)
>>>>>>> 4ffd06f83db619662036c5641a7625408eae2a23
            print(self.exploration_rate)
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class GDAI:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.state_size        = get_screen().shape
        self.action_size       = 2
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for episode in range(self.episodes):
                state = get_screen()
                restart()
                next_state = get_screen()
                self.state_size = state.shape

                alive = True
                rounds = 0
                time_start = time.time()
                frame_queue = [next_state]
                while alive:
                    # cv2.imshow("GDB View",state)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     cv2.destroyAllWindows()
                    #     break

                    action = self.agent.act(reshape_screen(state))

                    if action == 1:
                        press_LB()
                    else:
                        release_LB()

                    next_state = get_screen()
                    frame_queue.insert(0, next_state)
                    if rounds > 8:
                        alive = isalive(state, frame_queue.pop())

                    reward = time.time() - time_start

                    if not alive:
                        reward = -10

                    self.agent.remember(reshape_screen(state), action, reward, reshape_screen(next_state), alive)
                    rounds += 1

                    state = next_state
                print("Episode {} Time Alive: {} Rounds: {}".format(episode, time.time() - time_start, rounds))
                self.agent.replay(self.sample_batch_size)
                if episode % 16 == 0:
                    print("Saving model into gd_weights...")
                    self.agent.save_model()
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    GDAI = GDAI()
    GDAI.run()
