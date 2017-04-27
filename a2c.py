from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import cv2
import gym
#import gym_vgdl
import numpy as np
import tensorflow as tf

from ops import linear, conv2d, flatten


class Agent():
    def __init__(self, session, args):
        self.n_input = args.input_size     # Number of features in each observation
        self.num_obs = 2                   # Number of observations in each state
        self.n_actions = args.num_actions  # Number of output q_values
        self.discount = args.discount      # Discount factor
        self.use_target = args.use_target
        self.learning_rate = args.learning_rate
        self.EWC_weight = args.EWC_weight

	#self.layer_sizes = [self.n_input * self.num_obs] + args.layer_sizes + [self.n_actions]

        self.session = session

        self.memory = ReplayMemory(args)


        # Tensorflow variables:

        # Model
        self.state = tf.placeholder("float", [None, self.num_obs, 84, 84])
        with tf.variable_scope('prediction'):
            self.pi, self.V, self.net_weights = self.cnn(self.state, [], self.n_actions)
        with tf.variable_scope('target'):
            _, self.targ_V, self.targ_net_weights = self.cnn(self.state, [], self.n_actions)

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

        # Calculations for EWC weighting
        log_critic_loss = tf.log(self.critic_loss + self.actor_objective)
        loss_grads = flatten(tf.gradients(log_critic_loss, self.net_weights))
        fisher = tf.square(loss_grads)
        EWC_weight = tf.reduce_mean( tf.exp(-tf.abs(adv_critic)) )
        self.batch_EWC_strength = EWC_weight * fisher

        # Member variable to store current EWC_strength
        self.EWC_strength = np.zeros(self.batch_EWC_strength.get_shape())

        self.EWC_strength_ph = tf.placeholder("float", self.batch_EWC_strength.get_shape())
        EWC_term = tf.reduce_sum( self.EWC_strength_ph * tf.square(flatten(self.net_weights) - flatten(self.targ_net_weights)) )
            
        self.loss = self.critic_loss + self.actor_objective + EWC_term
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Global step (NB: Updated infrequently)
        self.step = tf.Variable(0, name='global_step', trainable=False)


    def predict(self, state):
        # get probabilities from pi with current network
        a, V = self.session.run([self.pi, self.V], feed_dict={self.state: [state]})

        #a = a[0] / np.sum(a[0], axis=0)
        #action = np.random.choice(range(a.size), p=a)

        probs = a[0] - np.finfo(np.float32).epsneg
        histogram = np.random.multinomial(1, probs)
        action = int(np.nonzero(histogram)[0])

        return action, V[0], a[0]


    def Update(self, histories):

        y_batch = [] # Targets for V
        a_batch = [] # Actions selected
        s_batch = [] # States
        adv_batch = [] # Advantages (ie R - V(s))
        for h in histories:
          actions = h['actions']
          states = h['states']
          rewards = h['rewards']
          terminals = h['terminals']

          values = self.session.run((self.targ_V if self.use_target else self.V), feed_dict={self.state: states})

          R = values[-1]

          for i in reversed(range(len(states)-1)):
            if terminals[i+1]:
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
                   self.actor_target: adv_batch,
                   self.EWC_strength_ph: self.EWC_strength }

        _, batch_EWC_strength = self.session.run([self.optim, self.batch_EWC_strength], feed_dict=feed_dict)

        # Update EWC with weights of new batch
        self.EWC_strength = self.EWC_weight * batch_EWC_strength + 0.999 * self.EWC_strength
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
        shape = tf.shape(state)
        fc = tf.reshape(state, (shape[0], shape[1] * shape[2]))
        for i in range(hidden_dim - 1):
            fc = tf.nn.relu(tf.matmul(fc, weights[i]) + biases[i]) 
        pi_ = tf.nn.softmax(tf.matmul(fc, weights[-1]) + biases[-1])
        pi = pi_ / tf.reduce_sum(pi_, 1)
        V = tf.reshape(tf.matmul(fc, V_w) + V_b, [-1])

        # Returns the output policy and value function
        return pi_, V, weights + biases + [V_w] + [V_b]


    # Adapted from github.com/devsisters/DQN-tensorflow/
    def cnn(self, state, input_dims, num_actions):
        w = {}
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        state = tf.transpose(state, perm=[0, 2, 3, 1])

        l1, w['l1_w'], w['l1_b'] = conv2d(state,
          32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
        l2, w['l2_w'], w['l2_b'] = conv2d(l1,
          64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
        
        shape = l2.get_shape().as_list()
        l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

        l3, w['l3_w'], w['l3_b'] = linear(l2_flat, 256, activation_fn=activation_fn, name='value_hid')


        value, w['val_w_out'], w['val_w_b'] = linear(l3, 1, name='value_out')
        V = tf.reshape(value, [-1])

        pi_, w['pi_w_out'], w['pi_w_b'] = \
            linear(l3, num_actions, activation_fn=tf.nn.softmax, name='pi_out')

        sums = tf.tile(tf.expand_dims(tf.reduce_sum(pi_, 1), 1), [1, num_actions])
        pi = pi_ / sums

        #A3C is l1 = (16, [8,8], [4,4], ReLu), l2 = (32, [4,4], [2,2], ReLu), l3 = (256, Conn, ReLu), V = (1, Conn, Lin), pi = (#act, Conn, Softmax)
        return pi, V, [ v for v in w.values() ]
      



# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, args):
    self.memory_size = args.memory_size
    self.batch_size = args.batch_size
    self.history_len = 5
    self.num_obs = 2
    self.n_inputs = args.input_size

    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.float16)
    self.obs = np.empty((self.memory_size, 84, 84), dtype=np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.count = 0
    self.current = 0

  def add(self, action, reward, state, terminal):
    # NB! state is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.obs[self.current] = state
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.num_obs - 1:
      # use faster slicing
      return self.obs[(index - (self.num_obs - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.num_obs))]
      return self.obs[indexes, ...]

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
        if self.current in range(index-self.num_obs,index+history_len):
          continue
        # otherwise use this index
        break
      history={'actions': [], 'states': [], 'rewards': [], 'terminals': [] }
      for i in range(history_len):
        ind = (index + i) % self.memory_size
        history['actions'].append(self.actions[ind])
        history['states'].append(self.getState(ind))
        history['rewards'].append(self.rewards[ind])
        history['terminals'].append(self.terminals[ind])

      histories.append(history)
      indexes.append(index)

    return histories



def main(_):
    np.set_printoptions(threshold='nan', precision=3, suppress=True)

    # Launch the graph
    with tf.Session() as sess:

        training_iters = args.training_iters
        display_step = args.display_step
        save_step = display_step*5
        training_start = args.memory_size
        batch_size = args.batch_size

        env = gym.make(args.env)
        scrn = env.reset()
        state = state_ = cv2.resize(cv2.cvtColor(scrn, cv2.COLOR_RGB2GRAY)/255., (84,84))

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
        a = np.zeros(args.num_actions)

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
            sess.run(tf.global_variables_initializer())
            start_step = 0

          # Keep training until reach max iterations
          for step in tqdm(range(start_step,training_iters), ncols=70):

            # Act, and add 
            s = [state_, state]
            act, v_, a_ = agent.predict(s)
            scrn, reward, terminal, _ = env.step(act)
            state = cv2.resize(cv2.cvtColor(scrn, cv2.COLOR_RGB2GRAY)/255., (84,84))
            agent.memory.add(act, reward, state_, terminal)
            state_ = state

            # keep track of total reward
            r += reward
            ep_r += reward
            v += v_
            a += a_

            if terminal:
                #Reset environment and add episode reward to list
                scrn = env.reset()
                state = state_ = cv2.resize(cv2.cvtColor(scrn, cv2.COLOR_RGB2GRAY)/255., (84,84))
                rewards.append(ep_r); ep_r = 0

            # Train 
            if (agent.memory.count >= training_start):
                # Get transition sample from memory
                his = agent.memory.sample(4)
                # Run optimization op (backprop)
                agent.Update(his)

            # Update target network
            if step % args.target_step == 0 & args.use_target:
                ops = [ agent.targ_net_weights[i].assign(agent.net_weights[i]) for i, _ in enumerate(agent.targ_net_weights) ]
                sess.run(ops)


            # Display Statistics
            if (step) % display_step == 0:
                 r = r/display_step; v = v/display_step; a = a/display_step # get average reward
                 ewc = np.mean(agent.EWC_strength)
                 if rewards != []:
                     max_ep_r = np.amax(rewards); avr_ep_r = np.mean(rewards)
                 else:
                     max_ep_r = avr_ep_r = 0
                 tqdm.write("{}, {:>7}/{}it | avg_r: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, num_eps: {}, avg_V: {:4.2f}, ewc: {:6.1f}, probs: {}"\
                            .format(time.strftime("%H:%M:%S"), step, \
                            training_iters, r, avr_ep_r, max_ep_r, len(rewards), v, ewc, a))
                 r=0; max_ep_r = 0; v=0; a = np.zeros(args.num_actions)
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

          while True:
            # Act
            s = [state_, state]
            act, _, _ = agent.predict(s)

            scrn, reward, terminal, _ = env.step(act)
            state = cv2.resize(cv2.cvtColor(scrn, cv2.COLOR_RGB2GRAY)/255., (84,84))

            state_ = state

            ep_r += reward

            env.render()

            if terminal:
                scrn = env.reset()
                state = state_ = cv2.resize(cv2.cvtColor(scrn, cv2.COLOR_RGB2GRAY)/255., (84,84))
                print "{}: Episode finished with reward {}".format(time.strftime("%H:%M:%S"), ep_r)
                ep_r = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0',
                       help='Name of Gym environment')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=10000,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--memory_size', type=int, default=10000,
                       help='Time to start training from')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--use_target', type=bool, default=True,
                       help='Use separate target network')
    parser.add_argument('--target_step', type=int, default=1000,
                       help='Steps between updates of the target network')

    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for TD updates')

    parser.add_argument('--EWC_weight', type=float, default=0.01,
                       help='EWC')

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

