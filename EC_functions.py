__author__ = 'sudeep raja'
import numpy as np
import cPickle
import heapq
from sklearn.neighbors import BallTree, KDTree


class LRU_KNN:
    def __init__(self, capacity, dimension_result):
        self.capacity = capacity
        self.states = np.zeros((capacity, dimension_result))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.weights = np.ones(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None


    def nn(self, key, knn):
        dist, ind = self.tree.query([key], k=knn)

        for index in ind[0]:
            self.lru[index] = self.tm
            self.tm+=0.01

        embs = self.states[ind[0]]
        values = self.q_values[ind[0]]
        weights = self.weights[ind[0]]

        return embs, values, weights

    def add(self, keys, values):

        skip_indices = []
        if self.curr_capacity >= 1:
            dist, ind = self.tree.query(keys, k=1)
            for i, d in enumerate(dist):
                if d[0] < 0.001:
                    new_value = values[i]
                    index = ind[i][0]
                    self.q_values[index] = self.q_values[index]*0.9 + new_value*0.1
                    skip_indices.append(i)

        for i, _ in enumerate(keys):
            if i in skip_indices: continue
            if self.curr_capacity >= self.capacity:
                # find the LRU entry
                old_index = np.argmin(self.lru)
                self.states[old_index] = keys[i]
                self.q_values[old_index] = values[i]
                self.lru[old_index] = self.tm
            else:
                self.states[self.curr_capacity] = keys[i]
                self.q_values[self.curr_capacity] = values[i]
                self.lru[self.curr_capacity] = self.tm
                self.curr_capacity+=1
            self.tm += 0.01
        self.tree = KDTree(self.states[:self.curr_capacity])


class alltheNN:
    def __init__(self, capacity, dimension_result, alpha=0.05):
        self.capacity = capacity
        self.states = np.zeros((capacity, dimension_result))
        self.q_values = np.zeros(capacity)
        self.weights = np.zeros(capacity)
        self.curr_capacity = 0
        self.last_added = 0

    def nn(self, key, knn=0):
        ind = range(self.curr_capacity)
        embs = self.states[ind]
        values = self.q_values[ind]
        weights = self.weights[ind]

        return embs, values, weights


    def add(self, keys, values):
        for i, _ in enumerate(keys):
            if self.curr_capacity >= self.capacity:
                # find the LRU entry
                index = self.last_added
            else:
                index = self.curr_capacity
                self.curr_capacity+=1

            self.last_added = (self.last_added + 1) % self.capacity
            self.states[index] = keys[i]
            self.q_values[index] = values[i]
            self.weights[index] = 1.0


class Weighted_KNN:
    def __init__(self, capacity, dimension_result, alpha=0.05):
        self.capacity = capacity
        self.states = np.zeros((capacity, dimension_result))
        self.q_values = np.zeros(capacity)
        self.weights = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.alpha = alpha
        self.tree = None

    def nn(self, key, knn):
        dist, ind = self.tree.query(np.pad([key], ((0,0),(0,1)), 'constant', constant_values=1.0), k=knn)
        #dist, ind = self.tree.query([key], k=knn)

        embs = self.states[ind[0]]
        values = self.q_values[ind[0]]
        weights = self.weights[ind[0]]

        return embs, values, weights


    def add(self, keys, values):

        if self.curr_capacity >= 5:
          dist, ind = self.tree.query(np.pad(keys, ((0,0),(0,1)), 'constant', constant_values=1.0), k=5)
          #dist, ind = self.tree.query(keys, k=50)
          for i, ind_ in enumerate(ind):
            stren = 1 - self.alpha
            self.weights[ind_] = self.weights[ind_] * stren

        for i, _ in enumerate(keys):
            low_w = 1.0
            if self.curr_capacity >= self.capacity:
                # find the LRU entry
                old_index = np.argmin(self.weights)
                low_w = min(low_w, self.weights[old_index])
                index = old_index
            else:
                index = self.curr_capacity
                self.curr_capacity+=1

            self.states[index] = keys[i]
            self.q_values[index] = values[i]
            self.weights[index] = 1.0

        self.tree = KDTree(np.concatenate((self.states[:self.curr_capacity], np.expand_dims(self.weights[:self.curr_capacity], axis=1)),axis=1))
        #self.tree = KDTree(self.states[:self.curr_capacity])

