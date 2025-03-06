import argparse
import numpy as np
import os
import tensorflow as tf

import gym
import model

from tensorflow import keras

from gym_go import gogame, govars

np.set_printoptions(suppress=True, linewidth=200, precision=5)

parser = argparse.ArgumentParser(description='Go REINFORCE')
parser.add_argument('--boardsize', '-b', type=int, default=7)
parser.add_argument('--komi', '-k', type=float, default=0)
args = parser.parse_args()

defaults = {
    "input_dim": 6,
    "board": args.boardsize,
    "filters_per_layer": 43,
    "layers": 6,
    "filter_width_1": 3
}


def channels_last(state):
    if state.ndim == 3:
        state = state.transpose(1, 2, 0)
        state = state[np.newaxis, :]
    else:
        state = state.transpose(0, 2, 3, 1)
    return state


class Game:

    def __init__(self, bsize, komi):
        self.bsize = bsize
        self.komi = komi
        self.go_env = gym.make('gym_go:go-v0', size=bsize, komi=komi)
        self.states = [] 
        self.turns = []
        self.actions = []
        self.done = False
        self.winner = None

    def reset(self):
        self.go_env.reset()
        del self.states[:]
        del self.turns[:]
        del self.actions[:]
        self.done = False
        self.winner = None

    def state(self):
        return self.go_env.state()

    def step(self, state, action):
        turn = self.turn()

        _, _, done, _ = self.go_env.step(action)

        if action:
            self.states.append(state)
            self.turns.append(turn)

            onehot_action = np.zeros(gogame.action_size(state), np.float32)
            onehot_action[action] = 1
            self.actions.append(onehot_action)

        if done:
            self.done = True
            w = self.go_env.winner()
            if w == 1:
                self.winner = 0  # b
            elif w == -1:
                self.winner = 1  # w
            else:
                self.winner = None # draw

        return done

    def reward(self, color):
        if self.winner is None:
            return 0
        elif self.winner == color:
            return 1
        else:
            return -1

    def turn(self):
        return self.go_env.turn()

    def done(self):
        return self.done

    def render(self, mode):
        self.go_env.render(mode=mode)


class Trainer:

    def __init__(self,
                 learner,
                 optimizer,
                 clip_norm=None,
                 clip_value_min=None,
                 clip_value_max=None,
                 global_clip_norm=None):
        self.learner = learner
        self.optimizer = optimizer
        self.clip_norm=clip_norm
        self.clip_value_min=clip_value_min
        self.clip_value_max=clip_value_max
        self.global_clip_norm=global_clip_norm
        self.games_states = []
        self.games_actions = []
        self.games_reward = []

    def clear(self):
        del self.games_states[:]
        del self.games_actions[:]
        del self.games_reward[:]

    def add_game(self, game):
        reward = game.reward(self.learner.color)

        learner_states = [game.states[i] for i, c in enumerate(game.turns) if c == self.learner.color]
        self.games_states.append(learner_states)

        learner_actions = [game.actions[i] for i, c in enumerate(game.turns) if c == self.learner.color]
        self.games_actions.append(learner_actions)

        self.games_reward.append(reward)

    def update(self):
        losses = []
        grads = None
        game_batch = len(self.games_reward)
        for states, actions, reward in zip(self.games_states, self.games_actions, self.games_reward):
            states = np.stack(states)
            states = channels_last(states)
            states = tf.cast(tf.constant(states), tf.float32)
            actions = np.stack(actions)
            actions = tf.cast(tf.constant(actions), tf.float32)
            z  = tf.cast(tf.constant(reward), tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(states)
                probs = self.learner.model(states)
                masked_probs = probs * actions
                safe_probs = tf.where(tf.equal(masked_probs, 0.), tf.ones_like(masked_probs), masked_probs)
                log_probs = tf.math.log(safe_probs)
                # take the average of step loss (-log(p(a|s))) since the number of moves varies from game to game
                game_loss = -tf.reduce_mean(tf.reduce_sum(log_probs, axis=1)) * z
                game_grads = tape.gradient(game_loss, self.learner.model.trainable_variables)

                losses.append(game_loss.numpy())

            if grads:
                for i, g in enumerate(game_grads):
                    grads[i] += g/game_batch
            else:
                grads = [g/game_batch for g in game_grads]

        # clip grads
        if self.global_clip_norm is not None:
            scaled_grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.global_clip_norm)
        elif self.clip_norm is not None:
            scaled_grads = [tf.clip_by_norm(grad, self.clip_norm) for grad in grads]
        elif not (self.clip_value_min is None or self.clip_value_max is None):
            scaled_grads = [tf.clip_by_value(grad, self.clip_value_min, self.clip_value_max) for grad in grads]
        else:
            scaled_grads

        self.learner.model.optimizer.apply_gradients(zip(scaled_grads, self.learner.model.trainable_variables))

        return losses, grads, scaled_grads


class Player:

    def __init__(self, model):
        self.model = model
        self.color = None

    def generate_action(self, state):
        mask = 1.0 - state[govars.INVD_CHNL].ravel()
        mask = np.append(mask, 1.0)  # add PASS option
        state = channels_last(state)
        output = self.model.predict(state, verbose=0).ravel()
        prob = output * mask
        if any(prob):
            prob /= np.sum(prob)
            action = np.random.choice(len(prob), p=prob)
        else:
            action = len(prob) - 1 # PASS
        return action


learner_model = model.cnn_policy(**defaults)
opponent_model = model.cnn_policy(**defaults)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Instantiate the learning rate schedule
#initial_learning_rate = 0.001
#alpha = 0.01
#warmup_target = 1e-3
#cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
#    initial_learning_rate=warmup_target,
#    decay_steps=1000,
#    alpha=alpha
#)
#optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)

learner_model.compile(optimizer=optimizer)
learner_model.summary()

bsize = args.boardsize
komi = args.komi

game_batch = 32 
n_batches = 320

game = Game(bsize, komi)
learner = Player(learner_model)
opponent = Player(opponent_model)

trainer = Trainer(learner, optimizer, global_clip_norm=1.0)

for i in range(n_batches):
    rewards, move_lens = [], []
    for j in range(game_batch):
        game.reset()

        if j % 2 == 0:
            current, other = opponent, learner
        else:
            current, other = learner, opponent

        current.color, other.color = 0, 1

        k = 0
        while True: 
            state = game.state()
            turn = game.turn()

            action = current.generate_action(state)
            done = game.step(state, action)
            
            if done:
                reward = game.reward(learner.color)
                rewards.append(reward)
                move_len = len(game.actions)
                move_lens.append(move_len)

                trainer.add_game(game)

                break

            current, other = other, current

            k += 1
    
    losses, grads, scaled_grads = trainer.update()

    # Check grads global norm
    grads_norm = tf.linalg.global_norm(grads)
    scaled_grads_norm = tf.linalg.global_norm(scaled_grads)
    
    print(f"Batch {i:03}. Avg. Moves: {np.mean(move_lens):.1f} Total Reward: {np.sum(rewards)} Winning Ratio: {np.count_nonzero(np.array(rewards)==1)/len(rewards):.3f} Loss: mean. {np.mean(losses):.5f} std. {np.std(losses):.5f} var. {np.var(losses):.5f} Grads Norm: {grads_norm:.5f} Scaled Grads Norm. {scaled_grads_norm:.5f} lr: {learner_model.optimizer.learning_rate.numpy():.10f}")

    trainer.clear()

    # Save intermediate models.
    if i % 10 == 0:
        os.makedirs("./logs/weights", exist_ok=True)
        learner_weights = "{:05d}.weights.h5".format(i)
        learner_model.save_weights(os.path.join('./logs/weights', learner_weights))


    learner.color = 1 - learner.color

