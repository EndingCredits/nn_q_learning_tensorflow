from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

import serial

from ops import flatten

class Agent():
    def __init__(self, session, args):
        self.n_input = args.input_size     # Number of features in each observation
        self.num_obs = 2                   # Number of observations in each state
        self.n_actions = args.num_actions  # Number of output q_values
        self.discount = args.discount      # Discount factor
        self.epsilon = 0.25                # Epsilon
        self.learning_rate = args.learning_rate
        self.regularization = args.reg
        self.use_target = args.use_target
        self.double_q = args.double_q

        self.EWC = args.EWC
        self.EWC_decay = args.EWC_decay

        self.beta = args.beta

	self.layer_sizes = [self.n_input] + args.layer_sizes + [self.n_actions]

        self.session = session

        self.memory = ReplayMemory(args)

        # Tensorflow variables:

        # Model for Q-values

        self.state = tf.placeholder("float", [None, self.n_input])
        with tf.variable_scope('prediction'):
            self.pred_q, self.reg, self.pred_weights = self.network(self.state, self.layer_sizes)
        with tf.variable_scope('target'):
            self.target_pred_q, _, self.target_weights = self.network(self.state, self.layer_sizes)

        self.flattened_weights = flatten(self.pred_weights)
        
        #self.state = tf.placeholder("float", [None, self.num_obs, 84, 84])
        #with tf.variable_scope('prediction'):
        #    self.pred_q, self.reg, self.pred_weights = self.cnn(self.state, [], self.n_actions)
        #with tf.variable_scope('target'):
        #    self.target_pred_q, _, self.target_weights = self.cnn(self.state, [], self.n_actions)

        # Graph for loss function
        self.action = tf.placeholder('int64', [None])
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_q * action_one_hot, reduction_indices=1)

        self.target_q = tf.placeholder("float", [None])
        if self.beta==0:
            self.td_err = self.target_q - q_acted
        else:
            self.td_err = tf.exp(self.beta*self.target_q) - tf.exp(self.beta*q_acted)
        td_loss = tf.reduce_mean(tf.square(self.td_err))# + self.reg


        # Calculations for Elastic Weights
        log_td_loss = tf.log(td_loss)
        grads = flatten(tf.gradients(log_td_loss, self.pred_weights))
        fisher = tf.square(grads)
        fisher = 100 * fisher / tf.reduce_max(fisher) # Max normalise
        self.EWC_strength = fisher

        # Variables for holding dicounted sums
        self.EWC_strength_ = np.zeros(self.EWC_strength.get_shape())
        self.EWC_strength_s = np.zeros(self.EWC_strength.get_shape())
        self.EWC_strength_1 = np.zeros(self.EWC_strength.get_shape())
        self.EWC_strength_1s = np.zeros(self.EWC_strength.get_shape())

        # Placeholders to feed sums into
        self.EWC_strength_ph = tf.placeholder("float", self.EWC_strength.get_shape())
        self.EWC_strength_1_ph = tf.placeholder("float", self.EWC_strength.get_shape())

        #EWC_term = tf.reduce_sum( self.EWC_strength_ph * tf.square(flatten(self.pred_weights) - flatten(self.target_weights)) )
        EWC_term = tf.reduce_sum( self.EWC_strength_ph * tf.square(flatten(self.pred_weights)) - 2 * self.EWC_strength_1_ph * flatten(self.pred_weights) )


        total_loss = td_loss + EWC_term 
        
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        # Global step (NB: Updated infrequently)
        self.step = tf.Variable(0, name='global_step', trainable=False)


    def predict(self, state):
        # get q-vals with current network
        q = self.session.run(self.pred_q, feed_dict={self.state: [state]})

        # get best action and q val via max-q
        action = np.argmax(q, axis=1)[0]; q = np.max(q, axis=1)[0]

        # get action via epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)

        return action, q


    def tdUpdate(self, s_t0, a_t0, r_t0, s_t1, t_t1):

        # Get estimate of value, V, of s_(t+1)
        if self.double_q:
            #Predict action with current network
            action = np.argmax(self.pred_q.eval({self.state: s_t1}), axis=1)
            action_one_hot = np.eye(self.n_actions)[action] #neat little trick for getting one-hot

            # Get value of action from target network
            V_t1 = np.sum(np.multiply(self.target_pred_q.eval({self.state: s_t1}), action_one_hot), axis=1)
        else:
            # Get max value from current/target network
            q_t1 = self.target_pred_q.eval({self.state: s_t1}) if self.use_target else self.pred_q.eval({self.state: s_t1})
            V_t1 = np.max(q_t1, axis=1)
        
        # Set V to zero if episode has ended
        V_t1 = np.multiply(np.ones(shape=np.shape(t_t1)) - t_t1, V_t1)

        # Bellman Equation
        target_q_t = self.discount * V_t1 + r_t0

        # Update current network params
        #loss = self.session.run(self.log_td_loss, feed_dict={self.state: s_t0, self.target_q: target_q_t, self.action: a_t0})
        #grads = tf.gradients(loss, self.pred_weights)
        #print grads

        _, err, stren, weights = self.session.run([self.optim, self.td_err, self.EWC_strength, self.flattened_weights],
            feed_dict={self.state: s_t0, self.target_q: target_q_t, self.action: a_t0,
            self.EWC_strength_ph: self.EWC_strength_-self.EWC_strength_s, self.EWC_strength_1_ph: self.EWC_strength_1-self.EWC_strength_1s })
 
        #e=100 ; i = 0
        #while np.mean(np.abs(e)) > 0.8*np.mean(np.abs(err)) and i < 20:
        #    _ , e, stren, weights = self.session.run([self.optim, self.td_err, self.EWC_strength, self.flattened_weights],
        #      feed_dict={self.state: s_t0, self.target_q: target_q_t, self.action: a_t0,
        #      self.EWC_strength_ph: self.EWC_strength_-self.EWC_strength_s, self.EWC_strength_1_ph: self.EWC_strength_1-self.EWC_strength_1s })
        #    i += 1

        self.EWC_strength_ = self.EWC * stren + self.EWC_decay * self.EWC_strength_
        self.EWC_strength_1 = self.EWC * (stren*weights) + self.EWC_decay * self.EWC_strength_1
        self.EWC_strength_s = self.EWC * stren + np.square(self.EWC_decay) * self.EWC_strength_s
        self.EWC_strength_1s = self.EWC * (stren*weights) + np.square(self.EWC_decay) * self.EWC_strength_1s

        return True


    def network(self, state, d):
        hidden_dim = len(d)-1
        weights = [None]*hidden_dim
        biases = [None]*hidden_dim
    
        # Create params
        with tf.variable_scope("params") as vs:
          for i in range(hidden_dim):
            weights[i] = tf.Variable(tf.random_normal((d[i],d[i+1])), name='weights'+str(i+1))
            biases[i] = tf.Variable(tf.zeros(d[i+1]), name='biases'+str(i+1))
        
        # Build graph
        fc = state
        for i in range(hidden_dim - 1):
            fc = tf.nn.relu(tf.matmul(fc, weights[i]) + biases[i]) 
    
        Qs = tf.matmul(fc, weights[-1]) + biases[-1]
        reg = self.regularization * tf.reduce_mean([tf.reduce_mean(tf.square(w)) for w in weights])# + [tf.reduce_mean(tf.square(b)) for b in biases])

        # Returns the output Q-values
        return Qs, reg, weights + biases

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
        l3, w['l3_w'], w['l3_b'] = conv2d(l2,
          64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')

        shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='l4')
        q, w['q_w'], w['q_b'] = linear(l4, num_actions, name='q')

        reg = self.regularization * tf.reduce_mean([tf.reduce_mean(tf.square(w_)) for w_ in w.values()])

        return q, reg, w


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
    np.set_printoptions(threshold='nan', precision=3, suppress=True)

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
        q = 0

        #ser = serial.Serial('/dev/ttyACM1')
        #print ser.is_open
        

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
            act, q_ = agent.predict(state)
            state, reward, terminal, _ = env.step(act)
            agent.memory.add(act, reward, state, terminal)

            # keep track of total reward
            r += reward
            ep_r += reward
            q += q_ #0.5 * np.log(np.max([q_,1E-6]))

            if terminal:
                #Reset environment and add episode reward to list
                state = env.reset()
                rewards.append(ep_r); ep_r = 0


            # Train 
            if (agent.memory.count >= training_start): #& ((step) % batch_size == 0): ,_ for some reason this causes divergence
                # Get transition sample from memory
                s_t0, a_t0, r_t1, s_t1, t_t1 = agent.memory.sample()
                # Run optimization op (backprop)
                agent.tdUpdate(s_t0, a_t0, r_t1, s_t1, t_t1)

            if step % args.target_step == 0 & args.use_target:
                ops = [ agent.target_weights[i].assign(agent.pred_weights[i]) for i in range(len(agent.target_weights))]
                sess.run(ops)


            # Display Statistics
            if (step) % display_step == 0:
                 r = r/display_step; q = q/display_step # get average reward
                 ewc = np.mean(agent.EWC_strength_)
                 if rewards != []:
                     max_ep_r = np.amax(rewards); avr_ep_r = np.mean(rewards)
                 else:
                     max_ep_r = avr_ep_r = 0
                 tqdm.write("{}, {:>7}/{}it | avg_r: {:4.3f}, avg_q: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, num_eps: {}, epsilon: {:4.3f}, ewc: {:4.1f}"\
                            .format(time.strftime("%H:%M:%S"), step, training_iters, r, q, avr_ep_r, max_ep_r, len(rewards), epsilon, ewc))
                 #ser.write(b'\n\r')
                 #ser.write(b"{} r: {:4.1f}".format(step, avr_ep_r))
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
            act, _ = agent.predict(state)
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
    parser.add_argument('--display_step', type=int, default=2500,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--memory_size', type=int, default=1000,
                       help='Time to start training from')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--use_target', type=bool, default=True,
                       help='Use separate target network')
    parser.add_argument('--target_step', type=int, default=1000,
                       help='Steps between updates of the taget network')
    parser.add_argument('--double_q', type=int, default=1,
                       help='Use Double Q learning')

    parser.add_argument('--EWC', type=float, default=0.0,
                       help='Strength of elastic weights (try 0.00025)')
    parser.add_argument('--EWC_decay', type=float, default=0.999,
                       help='Discount factor for EWC contributions.')

    parser.add_argument('--beta', type=float, default=0, # see particle value functions
                       help='Beta for adjusted returns')

    parser.add_argument('--discount', type=float, default=0.9,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for TD updates')
    parser.add_argument('--reg', type=float, default=0, #0.1 seems to work here
                       help='Regularization parameter for network')

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

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters

    args.layer_sizes = [int(i) for i in (args.layer_sizes.split(',') if args.layer_sizes else [])]

    print args

    tf.app.run()

