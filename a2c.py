from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf


class Agent():
    def __init__(self, session, args):
        self.n_input = args.input_size     # Number of features in each observation
        self.n_actions = args.num_actions  # Number of output q_values
        self.discount = args.discount      # Discount factor

        self.learning_rate = args.learning_rate

	self.layer_sizes = [self.n_input] + args.layer_sizes + [self.n_actions]

        self.session = session

        self.memory = ReplayMemory(args)


        # Tensorflow variables:

        # Model
        self.state = tf.placeholder("float", [None, self.n_input])
        self.pi, self.V, self.net_weights = self.network(self.state, self.layer_sizes)
 
        # Graph for loss functions
        self.critic_target = tf.placeholder("float32", [None], name="critic_ph")
        self.actor_target = tf.placeholder("float32", [None], name="actor_ph")
        self.selected_action = tf.placeholder("int32", [None], name="action_ph")
        
        # Actor objective
        log_pi = tf.log(tf.add(self.pi,tf.constant(1e-30)))
        pi_entropy = tf.reduce_sum(tf.mul(self.pi, log_pi), reduction_indices = 1)
        action_one_hot = tf.one_hot(self.selected_action, self.n_actions, 1.0, 0.0)
        log_action = tf.reduce_sum(tf.mul(log_pi, action_one_hot), 1) # Log pi(s,a)
        advantage_term = tf.mul(log_action, self.actor_target) # Log pi(s,a) . (R-V(s))
        entropy_term = -0.01 * pi_entropy
        self.actor_objective = tf.reduce_sum(tf.mul(tf.constant(-1.0), advantage_term + entropy_term ))
            
        # Critic loss
        adv_critic = tf.sub(self.critic_target, self.V) #self.critic_target - self.V
        self.critic_loss = tf.mul(tf.constant(0.5), tf.nn.l2_loss(adv_critic))
            
        self.loss = self.critic_loss + self.actor_objective
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Global step (NB: Updated infrequently)
        self.step = tf.Variable(0, name='global_step', trainable=False)


    def predict(self, state):
        # get probabilities from pi with current network
        a, V = self.session.run([self.pi, self.V], feed_dict={self.state: [state]})

        a = a[0] / np.sum(a[0], axis=0)
        action = np.random.choice(range(a.size), p=a)
        return action, V[0]


    def Update(self, histories):

        y_batch = []; a_batch = []; s_batch = []; adv_batch = []
        for h in histories:
          actions = h['actions']
          states = h['states']
          rewards = h['rewards']
          terminals = h['terminals']

          values = self.session.run(self.V, feed_dict={self.state: states})

          if terminals[-1]:
            R = 0
          else:
            R = values[-1]

          for i in reversed(xrange(len(states))):
            if terminals[-1]:
              R = 0
            else:
              R = rewards[i] + self.discount * R

            y_batch.append(R)
            a_batch.append(actions[i])
            s_batch.append(states[i])
            adv_batch.append(R - values[i])

        feed_dict={self.state: s_batch, 
                       self.critic_target: y_batch,
                       self.selected_action: a_batch,
                       self.actor_target: adv_batch}

        grads = self.session.run(self.optim, feed_dict=feed_dict)
        return True


    def network(self, state, d):
        hidden_dim = len(d)-1

        # Create params
        weights = [None]*hidden_dim
        biases = [None]*hidden_dim
        for i in range(hidden_dim):
            weights[i] = tf.Variable(tf.random_normal((d[i],d[i+1])), name='weights'+str(i+1))
            biases[i] = tf.Variable(tf.zeros(d[i+1]), name='biases'+str(i+1))
        V_w = tf.Variable(tf.random_normal((d[-2],1)), name='V_weights')
        V_b = tf.Variable(tf.zeros(1), name='V_biases')
        
        # Build graph
        fc = state
        for i in range(hidden_dim - 1):
            fc = tf.nn.relu(tf.matmul(fc, weights[i]) + biases[i]) 
        pi_ = tf.nn.softmax(tf.matmul(fc, weights[-1]) + biases[-1])
        pi = pi_ / tf.reduce_sum(pi_, 1)
        V = tf.reshape(tf.matmul(fc, V_w) + V_b, [-1])

        # Returns the output policy and value function
        return pi_, V, weights + biases + [V_w] + [V_b]


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, args):
    self.memory_size = args.memory_size
    self.batch_size = args.batch_size
    self.history_len = 4#args.history_len
    self.n_inputs = args.input_size

    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.float16)
    self.states = np.empty((self.memory_size, self.n_inputs), dtype=np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.count = 0
    self.current = 0

  def add(self, action, reward, state, terminal):
    # NB! state is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.states[self.current] = state
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def sample(self, history_len):
    # sample random indexes
    indexes = []
    histories = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = np.random.randint(1, self.count - 1)
        # if wraps over current, then get new one
        if self.current in range(index,index+history_len):
          continue
        # otherwise use this index
        break
      history={'actions': [], 'states': [], 'rewards': [], 'terminals': [] }
      for i in range(history_len):
        ind = (index + i) % self.memory_size
        history['actions'].append(self.actions[ind])
        history['states'].append(self.states[ind])
        history['rewards'].append(self.rewards[ind])
        history['terminals'].append(self.terminals[ind])

      histories.append(history)
      indexes.append(index)

    return histories



def main(_):

    # Launch the graph
    with tf.Session() as sess:

        training_iters = args.training_iters
        display_step = args.display_step
        save_step = display_step*5
        training_start = args.memory_size
        batch_size = args.batch_size

        env = gym.make(args.env)
        state = state_ = env.reset()

        args.input_size = env.observation_space.shape[0]
        args.num_actions = env.action_space.n

        agent = Agent(sess, args)

        # Load saver after agent tf variables initialised
        saver = tf.train.Saver()

        # Variables for keeping track of agent performance
        rewards = []
        ep_r = 0
        r = 0
        v = 0

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

            # Act, and add 
            act, v_ = agent.predict(state)
            state, reward, terminal, _ = env.step(act)
            agent.memory.add(act, reward, state_, terminal)
            state_ = state

            # keep track of total reward
            r += reward
            ep_r += reward
            v += v_

            if terminal:
                #Reset environment and add episode reward to list
                state = state_ = env.reset()
                rewards.append(ep_r); ep_r = 0

            # Train 
            if (agent.memory.count >= training_start):
                # Get transition sample from memory
                his = agent.memory.sample(4)
                # Run optimization op (backprop)
                agent.Update(his)


            # Display Statistics
            if (step) % display_step == 0:
                 r = r/display_step; v = v/display_step # get average reward
                 if rewards != []:
                     max_ep_r = np.amax(rewards); avr_ep_r = np.mean(rewards)
                 else:
                     max_ep_r = avr_ep_r = 0
                 tqdm.write("{}, {:>7}/{}it | avg_r: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, num_eps: {}, avg_V: {:4.2f}".format(time.strftime("%H:%M:%S"), step, \
                            training_iters, r, avr_ep_r, max_ep_r, len(rewards), v))
                 r=0; max_ep_r = 0; v=0
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
    parser.add_argument('--display_step', type=int, default=10000,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--memory_size', type=int, default=1000,
                       help='Time to start training from')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--use_target', type=bool, default=True,
                       help='Use separate target network')
    parser.add_argument('--target_step', type=int, default=1000,
                       help='Steps between updates of the taget network')

    parser.add_argument('--discount', type=float, default=0.9,
                       help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for TD updates')

    parser.add_argument('--layer_sizes', type=str, default='20',
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
    args.layer_sizes = [int(i) for i in (args.layer_sizes.split(',') if args.layer_sizes else [])]

    print args

    tf.app.run()

