from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

import EC_functions


class Agent():
    def __init__(self, session, args):
        self.n_input = args.input_size     # Number of features in each observation
        self.num_obs = 2                   # Number of observations in each state
        self.n_actions = args.num_actions  # Number of output q_values
        self.discount = args.discount      # Discount factor
        self.epsilon = args.epsilon        # Epsilon
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.delta = 0.01
        self.number_nn = 50
	self.layer_sizes = [self.n_input] + args.layer_sizes
        self.session = session


        self.memory = ReplayMemory(args)
        

        # Tensorflow variables:

        # Model for Embeddings
        self.state = tf.placeholder("float", [None, self.n_input])
        with tf.variable_scope('embedding'):
            self.state_embeddings, self.weights = self.network(self.state, self.layer_sizes)

        # DNDs
        self.DNDs = []
        for a in xrange(self.n_actions):
            new_DND = EC_functions.LRU_KNN(1000, self.state_embeddings.get_shape()[-1])
            self.DNDs.append(new_DND)

        # DND Calculations (everything from here on needs these placeholders filled)
        self.dnd_embeddings = tf.placeholder("float", [None, self.number_nn, self.state_embeddings.get_shape()[-1]], name="dnd_embeddings")
        self.dnd_values = tf.placeholder("float", [None, self.number_nn], name="dnd_values")

        weightings = 1.0 / (tf.reduce_sum(tf.square(self.dnd_embeddings - tf.expand_dims(self.state_embeddings,1)), axis=2) + [self.delta]) 
        normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True) #keep dims for broadcasting
        if self.beta==0:
            self.pred_q = tf.reduce_sum(self.dnd_values * normalised_weightings, axis=1)
            #self.pred_q = tf.reduce_mean(self.dnd_values, axis=1)
        else:
            self.pred_q = tf.log(tf.reduce_sum(tf.exp(self.beta * self.dnd_values) * normalised_weightings, axis=1))

        # Loss Function
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = tf.reduce_sum(tf.square(self.td_err))
        
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)


    def get_state_embedding(self, states):
        # Returns the DND hashes for the given states
        embeddings = self.session.run(self.state_embeddings, feed_dict={self.state: states})
        return embeddings


    def get_nearest_neighbours(self, embeddings, actions):
        # Return the embeddings and values of nearest neighbours from the DNDs for the given embeddings and actions
        dnd_embeddings = [] ; dnd_values = []
        for i, a in enumerate(actions):
            e, v = self.DNDs[a].nn(embeddings[i], self.number_nn)
            dnd_embeddings.append(e) ; dnd_values.append(v)

        return dnd_embeddings, dnd_values

    def add_to_dnd(self, state, action, value):
        # Adds the given embedding to the corresponding dnd
        embedding = self.get_state_embedding([state])
        self.DNDs[action].add(embedding[0], value)

        return False


    def predict(self, state):
        # Return action and estimated state value for given state

        # Get state embedding
        embedding = self.get_state_embedding([state])

        # calculate Q-values
        qs = []
        for a in xrange(self.n_actions):
          if self.DNDs[a].curr_capacity < self.number_nn:
            q_ = [0.0]
          else:
            dnd_embeddings, dnd_values = self.get_nearest_neighbours(embedding, [a])
            q_ = self.session.run(self.pred_q, feed_dict={self.state_embeddings: embedding, self.dnd_embeddings: dnd_embeddings, self.dnd_values: dnd_values})
          qs.append(q_[0])
        action = np.argmax(qs) ; V = qs[action]

        # get action via epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            V = qs[action]

        # Return action and estimated state value
        return action, V


    def Train(self, states, actions, Q_targets):

        for a in xrange(self.n_actions):
          if self.DNDs[a].curr_capacity < self.number_nn:
            return True

        # Get nearest neighbours and their embeddings
        state_embeddings = self.get_state_embedding(states)
        dnd_embeddings, dnd_values = self.get_nearest_neighbours(state_embeddings, actions)

        self.session.run(self.optim, feed_dict={self.state: states, self.target_q: Q_targets, self.dnd_embeddings: dnd_embeddings, self.dnd_values: dnd_values})
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

        # Returns the output Q-values
        return Qs, weights + biases


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, args):
    self.memory_size = args.memory_size
    self.batch_size = args.batch_size
    self.n_inputs = args.input_size
    self.n_actions = args.num_actions

    self.states = np.empty((self.memory_size, self.n_inputs), dtype=np.float16)
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.returns = np.empty(self.memory_size, dtype = np.float16)

    self.count = 0
    self.current = 0

  def add(self, state, action, returns):
    self.states[self.current] = state
    self.actions[self.current] = action
    self.returns[self.current] = returns

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def sample(self):
    # sample random indexes
    indexes = []
    watchdog = 0
    while len(indexes) < self.batch_size:
      # find random index 
      index = np.random.randint(1, self.count - 1)
      indexes.append(index)

    return self.states[indexes], self.actions[indexes], self.returns[indexes]



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



    # Training, act and learn

    # Load or initialise variables
    if args.resume_from is not None:
        # Load from file
        ckpt = tf.train.get_checkpoint_state(args.resume_from)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Initialize the variables
        sess.run(tf.initialize_all_variables())
        start_step = 0

    # Trajectory
    state = env.reset()
    states = [state] ; actions = []
    rewards = [] ; episode_t = 0

    n_step = 100

    # Stats for display
    ep_rewards = [] ; ep_reward_last = 0
    qs = [] ; q_last = 0

    # Keep training until reach max iterations
    for step in tqdm(range(start_step,training_iters), ncols=70):

        #TODO: Move this stuff into an agent update

        # Act, and add 
        act, value = agent.predict(state)
        state, reward, terminal, _ = env.step(act)

        actions.append(act) ; rewards.append(reward)
        episode_t += 1 ; states.append(state)

        # Bookeeping
        qs.append(value)

        if terminal:
            # Bookeeping
            ep_rewards.append(np.sum(rewards))

            # Calculate n-step Return for all remaining states
            start_t = episode_t - n_step
            if start_t < 0: start_t = 0
            R_t = 0

            for t in xrange(episode_t-1, start_t, -1):
                R_t = R_t * agent.discount + rewards[t]
    
                # Append to replay memory
                agent.memory.add(states[t], actions[t], R_t)
                agent.add_to_dnd(states[t], actions[t], R_t)

            # Reset environment
            state = env.reset()
            states = [state] ; actions = []
            rewards = [] ; episode_t = 0


        elif episode_t > n_step:

            # Calculate n-step Return
            start_t = episode_t - n_step
            R_t = value
            for t in xrange(episode_t-1, start_t, -1):
                R_t = R_t * agent.discount + rewards[t]

            # Append to replay memory
            agent.memory.add(states[start_t], actions[start_t], R_t)
            agent.add_to_dnd(states[start_t], actions[start_t], R_t)


        # Train 
        if (agent.memory.count >= training_start):
            # Get transition sample from memory
            s, a, R = agent.memory.sample()
            # Run optimization op (backprop)
            agent.Train(s, a, R)


        # Display Statistics
        if (step) % display_step == 0:
            avr_ep_reward = np.mean(ep_rewards[ep_reward_last:]) ; ep_reward_last = len(ep_rewards)
            avr_q = np.mean(qs[q_last:]) ; q_last = len(qs)
            tqdm.write("{}, {:>7}/{}it | q: {:4.3f}, ep_reward: {:4.1f}"\
                .format(time.strftime("%H:%M:%S"), step, training_iters, avr_q, avr_ep_reward))
                 

        # Save model
        if ((step+1) % save_step == 0) & (args.chk_dir is not None):
            sess.run(agent.step.assign(step))
            checkpoint_path = os.path.join(args.chk_dir, args.chk_name + '.ckpt')
            tqdm.write("Saving model to {}".format(checkpoint_path))
            saver.save(sess, checkpoint_path, global_step = step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                       help='Name of Gym environment')

    parser.add_argument('--training_iters', type=int, default=500000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=2500,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--memory_size', type=int, default=1000,
                       help='Size of DND dictionary')
    parser.add_argument('--replay_memory_size', type=int, default=1000,
                       help='Size of replay memory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')

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

    parser.add_argument('--layer_sizes', type=str, default='64',
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

