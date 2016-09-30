from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import gym
import gym_vgdl
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class Agent():
    def __init__(self, session, args):

        self.n_input = args.input_size     # Number of features in each observation
        self.n_actions = args.num_actions  # Number of output q_values
        self.discount = args.discount      # Discount factor
        self.epsilon = 0.25                # Epsilon
        self.learning_rate = args.learning_rate

	self.layer_sizes = [self.n_input] + args.layer_sizes + [self.n_actions]

        self.session = session

        self.memory = ReplayMemory(args)

        # Tensorflow variables
        self.state = tf.placeholder("float", [None, self.n_input])
        self.pred_q = self.network(self.state, self.layer_sizes)
        self.pred_action = tf.argmax(self.pred_q, dimension=1)
        self.target_q = tf.placeholder("float", [None])

        self.action = tf.placeholder('int64', [None])
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_q * action_one_hot, reduction_indices=1)

        delta = self.target_q - q_acted
        loss = tf.reduce_mean(tf.square(delta))
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.step = tf.Variable(0, name='global_step', trainable=False)

    def predict(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = self.pred_action.eval({self.state: [state]})[0]
        return action

    def tdUpdate(self, s_t0, a_t0, r_t0, s_t1, t_t1):
        q_t1 = self.pred_q.eval({self.state: s_t1})
        V_t1 = np.max(q_t1, axis=1)
        V_t1 = np.multiply(np.ones(shape=np.shape(t_t1)) - t_t1, V_t1)
        target_q_t = self.discount * V_t1 + r_t0
        self.session.run(self.optim, feed_dict={self.state: s_t0, self.target_q: target_q_t, self.action: a_t0})
        return True

    def network(self, state, d):
        hidden_dim = len(d)-1
        weights = [None]*hidden_dim
        biases = [None]*hidden_dim
    
        # Create params
        for i in range(hidden_dim):
            weights[i] = tf.Variable(tf.random_normal((d[i],d[i+1])), name='weights'+str(i+1))
            biases[i] = tf.Variable(tf.zeros(d[i+1], name='biases'+str(i+1)))
        
        # Build graph
        fc = state
        for i in range(hidden_dim-1):
            temp = tf.tanh(tf.matmul(fc, weights[i])) 
            fc = tf.nn.bias_add(temp, biases[i])
    
        Qs = tf.nn.bias_add(tf.matmul(fc, weights[-1]), biases[-1])

        # Returns the output Q-values
        return Qs

# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, args):
    self.memory_size = args.memory_size
    self.batch_size = args.batch_size
    self.n_inputs = args.input_size

    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.float16)
    self.states = np.empty((self.memory_size, self.n_inputs), dtype=np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.n_inputs), dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.n_inputs), dtype = np.float16)

  def add(self, action, reward, state, terminal):
    # NB! state is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.states[self.current] = state
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    return self.states[index]

  def sample(self):
    # sample random indexes
    indexes = []
    watchdog = 0
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = np.random.randint(1, self.count - 1)
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - 1):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes)] = self.getState(index-1)
      self.poststates[len(indexes)] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    return self.prestates, actions, rewards, self.poststates, terminals



def main(_):

    # Launch the graph
    with tf.Session() as sess:

        training_iters = args.training_iters
        display_step = args.display_step
        save_step = display_step*5
        training_start = args.memory_size
        batch_size = args.batch_size

        env = gym.make(args.env)
        state = env.reset()

        args.input_size = env.observation_space.shape[0]
        args.num_actions = env.action_space.n

        agent = Agent(sess, args)

        # Load saver after agent tf variables initialised
        saver = tf.train.Saver()

        # Variables for keeping track of agent performance
        rewards = []
        ep_r = 0
        r = 0


        if args.play_from is None:
          # Training, act and learn

          # Load or initialise variables
          if args.resume_from is not None:
            # Load from file
            ckpt = tf.train.get_checkpoint_state(args.resume_from)
            print("Loading model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = agent.step.eval()
            print start_step
          else:
            # Initialize the variables
            sess.run(tf.initialize_all_variables())
            start_step = 0

          # Keep training until reach max iterations
          for step in tqdm(range(start_step,training_iters), ncols=70):

            # Ideally this would be part of the agent, but updating the tf parameter is expensive
            per = min(step / args.epsilon_anneal, 1.)
            epsilon = args.epsilon * (1. - per) + args.epsilon_final * per
            agent.epsilon = epsilon

            # Act, and add 
            act = agent.predict(state)
            state, reward, terminal, _ = env.step(act)
            agent.memory.add(act, reward, state, terminal)

            # keep track of total reward
            r += reward
            ep_r += reward

            if terminal:
                #Reset environment and add episode reward to list
                state = env.reset()
                rewards.append(ep_r); ep_r = 0


            # Train 
            if (agent.memory.count > training_start): #& ((step) % batch_size == 0): ,_ for some reason this causes divergence
                # Get transition sample from memory
                s_t0, a_t0, r_t1, s_t1, t_t1 = agent.memory.sample()
                # Run optimization op (backprop)
                agent.tdUpdate(s_t0, a_t0, r_t1, s_t1, t_t1)


            # Display Statistics
            if (step) % display_step == 0:
                 r = r/display_step # get average reward
                 if rewards != []:
                     max_ep_r = np.amax(rewards); avr_ep_r = np.mean(rewards)
                 else:
                     max_ep_r = avr_ep_r = 0
                 tqdm.write("{}, {:>7}/{}it | avg_r: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, num_eps: {}, epsilon: {:4.3f}".format(time.strftime("%H:%M:%S"), step, \
                            training_iters, r, avr_ep_r, max_ep_r, len(rewards), epsilon))
                 r=0; max_ep_r = 0;
                 rewards = []

            # Save model
            if ((step+1) % save_step == 0) & (args.chk_dir is not None):
                 sess.run(agent.step.assign(step))
                 checkpoint_path = os.path.join(args.chk_dir, args.chk_name + '.ckpt')
                 tqdm.write("Saving model to {}".format(checkpoint_path))
                 saver.save(sess, checkpoint_path, global_step = step)


        else:
          # Playing from file, just act in the environment normally

          # Load from file
          ckpt = tf.train.get_checkpoint_state(args.play_from)
          print("Loading model from {}".format(ckpt.model_checkpoint_path))
          saver.restore(sess, ckpt.model_checkpoint_path)
          start_step = agent.step.eval()
          print start_step

          agent.epsilon = args.epsilon_final

          while True:
            act = agent.predict(state)
            state, reward, terminal, _ = env.step(act)

            ep_r += reward

            env.render()

            if terminal:
                state = env.reset()
                print "{}: Episode finished with reward {}".format(time.strftime("%H:%M:%S"), ep_r)
                ep_r = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                       help='Name of Gym environment')

    parser.add_argument('--training_iters', type=int, default=500000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=50000,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--memory_size', type=int, default=1000,
                       help='Time to start training from')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--discount', type=float, default=0.97,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.25,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')

    parser.add_argument('--learning_rate', type=float, default=0.0025,
                       help='Learning rate for TD updates')

    parser.add_argument('--layer_sizes', type=str, default='',
                       help='Hidden layer sizes for network, separate with comma')

    parser.add_argument('--chk_dir', type=str, default=None,
                       help='data directory to save checkpoints')
    parser.add_argument('--chk_name', type=str, default='model',
                       help='Name to save checkpoints as')

    parser.add_argument('--resume_from', type=str, default=None,
                       help='Location of checkpoint to resume from')

    parser.add_argument('--play_from', type=str, default=None,
                       help='Location of checkpoint to play game from (remember, you need the same layer sizes!)')

    args = parser.parse_args()

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters

    args.layer_sizes = [lambda: [int(i) for i in args.layer_sizes.split(',')], lambda: []][args.layer_sizes == '']() #Best expression ever?

    print args

    tf.app.run()

