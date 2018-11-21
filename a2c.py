from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pygame
import random
import sys
import tensorflow as tf

from interaction_network import predict_rollout_rl, predict_rollout
from physics_sim import MagneticGame



class Actor(object):
	def __init__(self, a_h_size, a_size, s_size, lr):
		with tf.variable_scope('Actor'):
			self.state = state = tf.placeholder(tf.float32, [None, s_size], "state")
			self.action = action = tf.placeholder(tf.int32, [None], "action")
			self.target = target = tf.placeholder(tf.float32, [None], "target")
			W = tf.get_variable("W", shape = [s_size, a_h_size], initializer = tf.random_normal_initializer(0, 0.1))
			b = tf.get_variable("b", shape = [a_h_size], initializer = tf.constant_initializer(0.1))
			hidden = tf.nn.relu(tf.add(tf.matmul(state, W), b))
			O = tf.get_variable("O", shape = [a_h_size, a_size], initializer = tf.random_normal_initializer(0, 0.1))
			self.output = output = tf.nn.softmax(tf.matmul(hidden, O))
			self.indices = indices = tf.range(0, tf.shape(output)[0]) * 2 + action
			self.act_probs = act_probs = tf.gather(tf.reshape(output, [-1]), indices)
			self.loss = loss = -tf.reduce_sum(tf.log(act_probs) * target)
			self.optimizer = optimizer = tf.train.AdamOptimizer(lr)
			self.train_op = train_op = optimizer.minimize(loss)

	def predict(self, state, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.output, feed_dict={self.state:[state]})

	def update(self, state, target, action, sess = None):
		sess = sess or tf.get_default_session()
		feed_dict = { self.state: state, self.target: target, self.action: action }
		sess.run([self.train_op, self.loss], feed_dict)


class Critic(object):
	def __init__(self, c_h_size, a_size, s_size, lr):
		with tf.variable_scope('Critic'):
			self.state = state = tf.placeholder(tf.float32, [None, s_size], "state")
			self.target = target = tf.placeholder(tf.float32, [None], "target")
			V1 = tf.get_variable("V1", shape = [s_size, c_h_size], initializer = tf.random_normal_initializer(0, 0.1))
			b1 = tf.get_variable("b1", shape = [c_h_size], initializer = tf.constant_initializer(0.1))
			v_out1 = tf.nn.relu(tf.add(tf.matmul(state, V1), b1))
			V2 = tf.get_variable("V2", shape = [c_h_size, 1], initializer = tf.random_normal_initializer(0, 0.1))
			self.v_out = v_out = tf.squeeze(tf.matmul(v_out1, V2))
			self.loss = loss = tf.reduce_sum(tf.square(target - v_out))
			self.optimizer = optimizer = tf.train.AdamOptimizer(lr)
			self.train_op = optimizer.minimize(loss)

	def predict(self, state, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.v_out, { self.state: [state] })

	def update(self, state, target, sess = None):
		sess = sess or tf.get_default_session()
		feed_dict = { self.state: state, self.target: target }
		sess.run([self.train_op, self.loss], feed_dict)


# THESE ARE FOR THE SIMULATION RUNOUT
DF = 0.99
VAR = 0.0003
T_DIFF = 0.001
AGENT_V = 5

def run_dynamic():
	EPSILON = 0.001
	GAMMA = 0.99
	A_HSIZE = 128 
	C_HSIZE = 128
	NUM_EPS = 1000
	VIS = False
	rewards = list()
	scores = list()

	scores = []
		
	for ep in range(10):

		game = MagneticGame(6, vis=True, record=True)
		Ds = game.Ds
		Da = len(game.action_space)
		game.reset()
		st = game.init_state

		hist = list()
		t = -1
		DONE = False

		full_state = np.zeros((6+4, 6), dtype = np.float64)
		for m, obj in enumerate(game.data[-1]):
			full_state[m] = np.array([obj.m, obj.x, obj.y, obj.v_x, obj.v_y, obj.t])
		simulation_data = predict_rollout_rl('./saved_parameters/w18', full_state, 5)

		j = -1

		while not DONE:
			t += 1
			# print(t)
			if t % 100 == 0:
				print(t)

			if t % 5 == 0:
				full_state = np.zeros((6+4, 6), dtype = np.float64)
				for k, obj in enumerate(game.data[-1]):
					full_state[k] = np.array([obj.m, obj.x, obj.y, obj.v_x, obj.v_y, obj.t])
				simulation_data = predict_rollout_rl('./saved_parameters/w18', full_state, 5)
				j = -1

			j += 1
			best_act = -1
			max_dist = 0
			for act in range(4):
				ap = st[-2:] + T_DIFF * game.action_space[act] * AGENT_V
				all_ = []
				third_closest = 10000
				three_closest = [third_closest]
				print(st)
				print(simulation_data[0])
				for obj in simulation_data[j]:
					dist_o = np.sqrt(((obj[0]*100-ap[0]*100))**2 + ((obj[1]*100-ap[1]*100))**2)
					all_.append(dist_o)
					if dist_o < third_closest:
						if len(three_closest) == 3:
							three_closest.remove(third_closest)
						three_closest.append(dist_o)
						third_closest = np.max(three_closest)

				print(three_closest)
				print(all_)
				return

				dist = np.sum(three_closest)
				if dist > max_dist:
					max_dist = dist
					best_act = act

			# j += 1
			# best_act = -1
			# max_dist = 0
			# for act in range(4):
			# 	ap = st[-2:] + T_DIFF * game.action_space[act] * AGENT_V
			# 	dist = 0
			# 	three_closest = []
			# 	third_closest = 0
			# 	for obj in simulation_data[j]:
			# 		dist_o = np.sqrt(((obj[0]-ap[0])*100)**2 + ((obj[1]-ap[1])*100)**2)
			# 		# print(dist_o)
			# 		if dist_o < 0.2:
			# 			continue
			# 		else:
			# 			dist += dist_o
			# 	if dist > max_dist:
			# 		max_dist = dist
			# 		best_act = act

			if best_act == -1:
				print("I'm gonna LOSE")
				best_act = 0
				
			game.step(best_act)
			st1, reward, done = game.curr_state
			hist.append((st, act, st1, reward))


			if done:
				print("Score: " + str(t))
				scores.append(t)
				if t > 300:
					return
				DONE = True
			else:
				st = st1

		pygame.quit()

	print(scores)
	print(np.mean(scores))


def run_actor_critic():
	EPSILON = 0.001
	GAMMA = 0.99
	A_HSIZE = 128
	C_HSIZE = 128
	NUM_EPS = 1000
	MAX_T = 50
	VIS = False

	game = MagneticGame(6, vis=VIS)
	Ds = game.Ds
	Da = len(game.action_space)

	rewards = list()
	scores = list()
	
	graph = tf.get_default_graph()
	with tf.Session() as sess:
		actor = Actor(A_HSIZE, Da, Ds, EPSILON)
		critic = Critic(C_HSIZE, Da, Ds, EPSILON)
		sess.run(tf.global_variables_initializer())

		for ep in range(NUM_EPS):
			print(ep)
			game.reset()
			st = game.init_state
			hist = list()
			t = -1
			DONE = False

			while not DONE:
				t += 1

				act_dist = actor.predict(st, sess)[0]
				act = np.argmax(act_dist)
				game.step(act)
				st1, reward, done = game.curr_state
				hist.append((st, act, st1, reward, critic.predict(st, sess)))

				if done:
					d_reward = list()
					for i in range(t, -1, -1):
						dt = 0
						for j in range(t - i):
							dt += np.power(GAMMA, j) * hist[i + j][3]
						d_reward.append(dt)
					d_reward.reverse()

					states = [x[0] for x in hist]
					actions = [x[1] for x in hist]
					advantage = [dt - x[4] for dt, x in zip(d_reward, hist)]

					critic.update(states, d_reward, sess)
					actor.update(states, advantage, actions, sess)

					rewards.append(t)
					DONE = True

				st = st1

			if ep % 100 == 0:
				x = np.mean(rewards[max(len(rewards) - 100, 0):])
				scores.append(x)
				if x > 400:
					return x

	return np.max(scores)


if __name__ == "__main__":
	run_dynamic()
	# run_actor_critic()
