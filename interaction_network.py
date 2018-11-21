from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import sys
import numpy as np
import random
import tensorflow as tf
import time

from math import ceil
from physics_sim import MagneticEnvironment, MagneticGame


FLAGS = None
MAX_EPOCH = 5 # 2000 * 20 in OG
MINI_BATCH_NUM = 100
EPSILON = 5e-4
BETA = 0.001 # totally random right now
SET_NUM = 250 # was 2000 in OG
WEIGHT_SAVE_FILE = './saved_parameters/w21'


def relation_data(mini_batch_num):
	"""
	To format the simple relations between entities necessary in the model.
	"""
	Rr_data = np.zeros((mini_batch_num, FLAGS.No, FLAGS.Nr), dtype=float)
	Rs_data = np.zeros((mini_batch_num, FLAGS.No, FLAGS.Nr), dtype=float)
	Ra_data = np.zeros((mini_batch_num, FLAGS.Dr, FLAGS.Nr), dtype=float)
	X_data = np.zeros((mini_batch_num, FLAGS.Dx, FLAGS.No), dtype=float)

	cnt = 0
	for i in range(FLAGS.No):
		for j in range(FLAGS.No):
			if i != j:
				Rr_data[:, i, cnt] = 1.0
				Rs_data[:, j, cnt] = 1.0
				cnt += 1

	return Rr_data, Rs_data, Ra_data, X_data


def train():
	"""
	End-to-end training of the Interaction Network architecture.
	"""
	################################
	# INPUTS
	################################

	O = tf.placeholder(tf.float32, [None, FLAGS.Ds, FLAGS.No], name="O")
	Rr = tf.placeholder(tf.float32, [None, FLAGS.No, FLAGS.Nr], name="Rr")
	Rs = tf.placeholder(tf.float32, [None, FLAGS.No, FLAGS.Nr], name="Rs")
	Ra = tf.placeholder(tf.float32, [None, FLAGS.Dr, FLAGS.Nr], name="Ra")
	X = tf.placeholder(tf.float32, [None, FLAGS.Dx, FLAGS.No], name="X")
	P_label = tf.placeholder(tf.float32, [None, FLAGS.Dp, FLAGS.No], name="P_label")

	################################
	# ARCHITECTURE
	################################

	# marshalling; m(G) = B, G = <O, R>
	B = tf.concat([(tf.matmul(O, Rr) - tf.matmul(O, Rs)), Ra], 1)

	# relational model; phi_R(B) = E
	r_h_size = 150
	B_trans = tf.transpose(B, [0, 2, 1])
	B_trans = tf.reshape(B_trans, [-1, (FLAGS.Ds + FLAGS.Dr)])
	w1 = tf.Variable(tf.truncated_normal([(FLAGS.Ds+FLAGS.Dr), r_h_size], stddev=0.1), name="r_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([r_h_size]), name="r_b1", dtype=tf.float32)
	h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1)
	w2 = tf.Variable(tf.truncated_normal([r_h_size, r_h_size], stddev=0.1), name="r_w2", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([r_h_size]), name="r_b2", dtype=tf.float32)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.Variable(tf.truncated_normal([r_h_size, r_h_size], stddev=0.1), name="r_w3", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([r_h_size]), name="r_b3", dtype=tf.float32)
	h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
	w4 = tf.Variable(tf.truncated_normal([r_h_size, r_h_size], stddev=0.1), name="r_w4", dtype=tf.float32)
	b4 = tf.Variable(tf.zeros([r_h_size]), name="r_b4", dtype=tf.float32);
	h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)
	w5 = tf.Variable(tf.truncated_normal([r_h_size, FLAGS.De], stddev=0.1), name="r_w5", dtype=tf.float32)
	b5 = tf.Variable(tf.zeros([FLAGS.De]), name="r_b5", dtype=tf.float32)
	h5 = tf.matmul(h4, w5) + b5
	h5_trans = tf.reshape(h5, [-1, FLAGS.Nr, FLAGS.De])
	E = tf.transpose(h5_trans, [0, 2, 1])

	# aggregation; a(G, X, E) = C, G = <O, Rr>
	E_bar = tf.matmul(E, tf.transpose(Rr, [0, 2, 1]))
	O_2 = tf.stack(tf.unstack(O, FLAGS.Ds, 1)[3:5], 1)
	C = tf.concat([O_2, X, E_bar], 1)

	# object model; phi_O(C) = P
	o_h_size = 100
	C_trans = tf.transpose(C, [0, 2, 1])
	C_trans = tf.reshape(C_trans, [-1, (2+FLAGS.Dx+FLAGS.De)])
	w1 = tf.Variable(tf.truncated_normal([(2+FLAGS.Dx+FLAGS.De), o_h_size], stddev=0.1), name="o_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([o_h_size]), name="o_b1", dtype=tf.float32)
	h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1)
	w2 = tf.Variable(tf.truncated_normal([o_h_size, FLAGS.Dp], stddev=0.1), name="o_w2", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([FLAGS.Dp]), name="o_b2", dtype=tf.float32)
	h2 = tf.matmul(h1, w2) + b2
	# tf.add_to_collection('unshaped_output', h2)
	h2_trans = tf.reshape(h2,[-1, FLAGS.No, FLAGS.Dp])
	P = tf.transpose(h2_trans, [0, 2, 1])
	tf.add_to_collection('prediction', P)

	#######################################
	# LOSS/OPTIMIZATION
	#######################################

	params_list = tf.global_variables()

	mse = tf.reduce_mean(tf.reduce_mean(tf.square(P - P_label), [1, 2]))
	tf.add_to_collection('error', mse)
	loss = BETA * tf.nn.l2_loss(E)
	for i in params_list:
		loss += BETA * tf.nn.l2_loss(i)
	global EPSILON
	optimizer = tf.train.AdamOptimizer(EPSILON)
	trainer = optimizer.minimize(mse + loss)
	# trainer = optimizer.minimize(mse)

	# session
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	saver = tf.train.Saver()

	#######################################
	# DATA GENERATION
	#######################################

	total_data = np.zeros((998 * SET_NUM, FLAGS.Ds, FLAGS.No), dtype = object)
	total_label = np.zeros((998 * SET_NUM, FLAGS.Dp, FLAGS.No), dtype = object)
	for i in range(SET_NUM):
		env = MagneticEnvironment(6, 1000)
		env.initial_state()
		raw_data = env.run()

		# [m,x1,y1,x2,y2,t] - i.e. no dynamic data
		data = np.zeros((998, FLAGS.Ds, FLAGS.No), dtype=float)
		label = np.zeros((998, FLAGS.Dp, FLAGS.No), dtype=object)
		for ind in range(1, 999):
			state_data = np.zeros((FLAGS.No, FLAGS.Ds), dtype=object)
			state_label = np.zeros((FLAGS.No, FLAGS.Dp), dtype=object)
			for j in range(FLAGS.No):
				po1 = raw_data[ind-1][j] # previous state of current object
				po2 = raw_data[ind][j] # current state of current object
				po3 = raw_data[ind+1][j] # next state of current object
				state_data[j] = np.array([po2[0],po1[1],po1[2],po2[1],po2[2],po2[5]])
				state_label[j] = np.array([po3[1],po3[2]])
			data[ind-1] = state_data.T
			label[ind-1] = state_label.T

		total_data[i*998:(i+1)*998,:] = data
		total_label[i*998:(i+1)*998,:] = label

	# shuffle
	tr_data_num = 998 * ceil(0.6 * SET_NUM)
	val_data_num = 998 * ceil(0.2 * SET_NUM)
	total_idx = list(range(len(total_data)))
	np.random.shuffle(total_idx)
	mixed_data = total_data[total_idx]
	mixed_label = total_label[total_idx]

	# # # training/validation/test data
	train_data = mixed_data[:tr_data_num]
	train_label = mixed_label[:tr_data_num]
	val_data = mixed_data[tr_data_num:tr_data_num + val_data_num]
	val_label = mixed_label[tr_data_num:tr_data_num + val_data_num]
	test_data = mixed_data[tr_data_num + val_data_num:]
	test_label = mixed_label[tr_data_num + val_data_num:]

	# relation data
	Rr_data, Rs_data, Ra_data, X_data = relation_data(MINI_BATCH_NUM)


	#######################################
	# TRAINING
	#######################################

	t = 0
	for i in range(MAX_EPOCH):

		total_idx = list(range(len(total_data)))
		np.random.shuffle(total_idx)
		mixed_data = total_data[total_idx]
		mixed_label = total_label[total_idx]
		train_data = mixed_data[:tr_data_num]
		train_label = mixed_label[:tr_data_num]

		tr_loss = 0
		for j in range(int(len(train_data) / MINI_BATCH_NUM)):
			batch_data = train_data[j * MINI_BATCH_NUM: (j+1) * MINI_BATCH_NUM]
			batch_label = train_label[j * MINI_BATCH_NUM: (j+1) * MINI_BATCH_NUM]
			tr_loss_part, _ = sess.run([mse, trainer], feed_dict = {O: batch_data, Rr: Rr_data, Rs: Rs_data, Ra: Ra_data, P_label: batch_label, X: X_data})
			tr_loss += tr_loss_part
		train_idx = list(range(len(train_data)))
		np.random.shuffle(train_idx)
		train_data = train_data[train_idx]
		train_label = train_label[train_idx]

		t += int(len(train_data) / MINI_BATCH_NUM)
		EPSILON = 5e-4 * np.exp(t/1.5e4)

		# val_loss = 0
		# for j in range(int(len(val_data) / MINI_BATCH_NUM)):
		# 	batch_data = val_data[j * MINI_BATCH_NUM: (j+1) * MINI_BATCH_NUM]
		# 	batch_label = val_label[j * MINI_BATCH_NUM: (j+1) * MINI_BATCH_NUM]
		# 	val_loss_part, estimated = sess.run([mse, P], feed_dict = {O: batch_data, Rr: Rr_data, Rs: Rs_data, Ra: Ra_data, P_label: batch_label, X: X_data})
		# 	val_loss += val_loss_part
		# val_idx = list(range(len(val_data)))
		# np.random.shuffle(val_idx)
		# val_data = val_data[val_idx]
		# val_label = val_label[val_idx]

		# print("Epoch " + str(i+1) + " Training MSE: " + str(tr_loss / (int(len(train_data) / MINI_BATCH_NUM))) + " Validation MSE: " + str(val_loss/(j+1)))
		print("Epoch " + str(i+1) + " Training MSE: " + str(tr_loss / (int(len(train_data) / MINI_BATCH_NUM))))

	saver.save(sess, WEIGHT_SAVE_FILE)

	val_loss = 0
	for k in range(int(len(val_data) / MINI_BATCH_NUM)):
		batch_data = val_data[j*MINI_BATCH_NUM:(j+1)*MINI_BATCH_NUM]
		batch_label = val_label[j*MINI_BATCH_NUM:(j+1)*MINI_BATCH_NUM]
		val_loss_part, estimated = sess.run([mse,P],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data})
		val_loss += val_loss_part

	print("Val MSE: " + str(val_loss / int(len(val_data) / MINI_BATCH_NUM)))

	# calculate test loss
	# test_loss = 0
	# for k in range(int(len(test_data) / MINI_BATCH_NUM)):
	# 	batch_data = test_data[j*MINI_BATCH_NUM:(j+1)*MINI_BATCH_NUM]
	# 	batch_label = test_label[j*MINI_BATCH_NUM:(j+1)*MINI_BATCH_NUM]
	# 	test_loss_part, estimated = sess.run([mse,P],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data})
	# 	test_loss += test_loss_part

	# print("Test MSE: " + str(test_loss / int(len(test_data) / MINI_BATCH_NUM)))

	# env = MagneticEnvironment(6, 1000)
	# env.initial_state()
	# raw_data = env.run()
	# formatted_data = np.zeros((998, FLAGS.No, FLAGS.Ds), dtype=float)
	# formatted_labels = np.zeros((998, FLAGS.No, FLAGS.Dp), dtype=float)
	# for ind in range(1, 999):
	# 	state_data = np.zeros((FLAGS.No, FLAGS.Ds), dtype=float)
	# 	state_label = np.zeros((FLAGS.No, FLAGS.Dp), dtype=float)
	# 	for j in range(FLAGS.No):
	# 		po1 = raw_data[ind-1][j] # previous state of current object
	# 		po2 = raw_data[ind][j] # current state of current object
	# 		po3 = raw_data[ind+1][j] # future state of current object
	# 		state_data[j] = np.array([po2[0],po1[1],po1[2],po2[1],po2[2],po2[5]])
	# 		state_label[j] = np.array([po3[1], po3[2]])
	# 	formatted_data[ind-1] = state_data
	# 	formatted_labels[ind-1] = state_label
	# total_idx = list(range(len(formatted_data)))
	# np.random.shuffle(total_idx)
	# data = formatted_data[total_idx]
	# labels = formatted_labels[total_idx]

	# predictions = np.zeros((998,FLAGS.No,FLAGS.Dp), dtype=float)
	# total_loss = 0
	# for i in range(1, 999):
	# 	part_loss, p = sess.run([mse, P], feed_dict={O:[data[i-1].T],Rs:[Rs_data[0]],X:[X_data[0]],Ra:[Ra_data[0]],Rr:[Rr_data[0]],P_label:[labels[i-1].T]})
	# 	total_loss += part_loss
	# 	p = p[0].T
	# 	predictions[i-1] = p

	# print(total_loss)


def predict_rollout_rl(params_file, initial_state, num_steps):
	"""
	Predicts the rollout of an entire simulation based on learnt architecture parameters.
	"""
	sess = tf.Session()
	saver = tf.train.import_meta_graph(params_file + ".meta")
	saver.restore(sess, params_file)
	graph = tf.get_default_graph()

	O = graph.get_tensor_by_name("O:0")
	Rr = graph.get_tensor_by_name("Rr:0")
	Rs = graph.get_tensor_by_name("Rs:0")
	Ra = graph.get_tensor_by_name("Ra:0")
	X = graph.get_tensor_by_name("X:0")
	P = tf.get_collection("prediction")[0]
	Rr_data, Rs_data, Ra_data, X_data = relation_data(1)

	pred_states = np.zeros((num_steps, FLAGS.No, FLAGS.Ds))
	pred_states[0] = initial_state
	predictions = np.zeros((num_steps, FLAGS.No, FLAGS.Dp), dtype=float)
	total_loss = 0
	for i in range(num_steps):
		p = sess.run(P, feed_dict={O: [pred_states[i].T], Rs:Rs_data, X:X_data, Ra:Ra_data, Rr:Rr_data})
		p = p[0].T
		next_state = np.zeros((FLAGS.No,FLAGS.Ds), dtype=float) # (10,6)
		for j in range(FLAGS.No):
			next_state[j,0] = initial_state[j,0]
			next_state[j,1] = pred_states[i-1,j,3]
			next_state[j,2] = pred_states[i-1,j,4]
			next_state[j,3] = p[j][0]
			next_state[j,4] = p[j][1]
			next_state[j,5] = initial_state[j,5]
		np.append(pred_states, next_state)
		predictions[i] = p
	
	return predictions


def predict_rollout(params_file, actual_data, num_steps):
	"""
	Predicts the rollout of an entire simulation based on learnt architecture parameters.
	"""
	sess = tf.Session()
	saver = tf.train.import_meta_graph(params_file + ".meta")
	saver.restore(sess, params_file)
	graph = tf.get_default_graph()

	O = graph.get_tensor_by_name("O:0")
	Rr = graph.get_tensor_by_name("Rr:0")
	Rs = graph.get_tensor_by_name("Rs:0")
	Ra = graph.get_tensor_by_name("Ra:0")
	X = graph.get_tensor_by_name("X:0")
	P = tf.get_collection("prediction")[0]

	Rr_data, Rs_data, Ra_data, X_data = relation_data(1)
	
	pred_states = [actual_data[0]]
	predictions = np.zeros((num_steps, FLAGS.No, FLAGS.Dp), dtype=float)
	total_loss = 0
	for i in range(num_steps):
		# p = sess.run(P, feed_dict={O: [actual_data[i].T], Rs:Rs_data, X:X_data, Ra:Ra_data, Rr:Rr_data})
		p = sess.run(P, feed_dict={O: [pred_states[-1].T], Rs:Rs_data, X:X_data, Ra:Ra_data, Rr:Rr_data})
		p = p[0].T
		next_state = np.zeros((FLAGS.No,FLAGS.Ds), dtype=float) # (10,6)
		for j in range(FLAGS.No):
			next_state[j,0] = actual_data[i,j,0]
			next_state[j,1] = actual_data[i,j,3]
			next_state[j,2] = actual_data[i,j,4]
			next_state[j,3] = p[j][0]
			next_state[j,4] = p[j][1]
			next_state[j,5] = actual_data[i,j,5]
		pred_states.append(next_state)
		predictions[i] = p
	
	return predictions


def convert_state(st):
	full_state = np.zeros((6+4, 6), dtype = np.float64)
	for m, obj in enumerate(st):
		full_state[m] = np.array([obj.m, obj.x, obj.y, obj.v_x, obj.v_y, obj.t])
	return full_state


def run_dynamic():
	EPSILON = 0.001
	GAMMA = 0.99
	NUM_EPS = 1000
	VIS = True
	T_DIFF = 0.0001
	AGENT_V = 15
	scores = []
		
	for ep in range(15):

		DONE = False
		t = -1
		j = -1

		game = MagneticGame(6, vis=True, record=False)
		Ds = game.Ds
		Da = len(game.action_space)
		game.reset()

		st = game.init_state
		full_state = convert_state(game.data[-1])
		simulation_data = predict_rollout_rl('./saved_parameters/w18', full_state, 5)

		while not DONE:
			# if t % 5 == 0:
			# 	full_state = convert_state(game.data[-1])
			# 	simulation_data = predict_rollout_rl('./saved_parameters/w18', full_state, 5)
			# 	j = -1

			# t += 1
			# # if t % 100 == 0:
			# # 	print(t)
			# j += 1

			act = random.randint(0,3)
			best_act = -1
			ap = st[-2:] + T_DIFF * game.action_space[act] * AGENT_V
			for o in range(6):
				obj = simulation_data[j][o]
				dist_o = np.sqrt(((obj[0]*100-ap[0]*100))**2 + ((obj[1]*100-ap[1]*100))**2)
				if dist_o / 100 < 0.2:
					continue
			for w in game.walls:
				if w.x1 == w.x2:
					dist_o = np.absolute(ap[0] - w.x1) * 100
				elif w.y1 == w.y2:
					dist_o = np.absolute(ap[1] - w.y1) * 100
				if dist_o / 100 < 0.2:
					continue
			if best_act != -1:
				best_act == act

			# max_dist = 0.2 * 100
			# for act in range(4):
			# 	ap = st[-2:] + T_DIFF * game.action_space[act] * AGENT_V
			# 	third_closest = float('inf')
			# 	three_closest = [third_closest]
			# 	# for obj in simulation_data[j]:
			# 	for o in range(6):
			# 		obj = simulation_data[j][o]
			# 		dist_o = np.sqrt(((obj[0]*100-ap[0]*100))**2 + ((obj[1]*100-ap[1]*100))**2)
			# 		if dist_o / 100 < 0.2:
			# 			break
			# 		if dist_o < third_closest:
			# 			if len(three_closest) == 3:
			# 				three_closest.remove(third_closest)
			# 			three_closest.append(dist_o)
			# 			third_closest = np.max(three_closest)
			# 	for w in game.walls:
			# 		if w.x1 == w.x2:
			# 			dist_o = np.absolute(ap[0] - w.x1) * 100
			# 		elif w.y1 == w.y2:
			# 			dist_o = np.absolute(ap[1] - w.y1) * 100
			# 		if dist_o / 100 < 0.2:
			# 			break
			# 		if dist_o < third_closest:
			# 			if len(three_closest) == 3:
			# 				three_closest.remove(third_closest)
			# 			three_closest.append(dist_o)
			# 			third_closest = np.max(three_closest)

			# 	# print(three_closest)
			# 	# print(np.sort(three_closest)[0] / 100)

			# 	factor = np.sort(three_closest)[0] / np.sum(three_closest)
			# 	dist = np.sum(three_closest) * factor
			# 	if dist > max_dist:
			# 		max_dist = dist
			# 		best_act = act

			# # GREEDY POLICY
			# best_act = -1
			# max_dist = 0
			# for act in range(4):
			# 	ap = st[-2:] + T_DIFF * game.action_space[act] * AGENT_V
			# 	dist = 0
			# 	for obj in simulation_data[j]:
			# 		dist_o = np.sqrt(((obj[0]-ap[0])*100)**2 + ((obj[1]-ap[1])*100)**2)
			# 		if dist_o < 0.2:
			# 			break
			# 		else:
			# 			dist += dist_o
			# 	if dist > max_dist:
			# 		max_dist = dist
			# 		best_act = act

			if best_act == -1:
				print("I'M GONNA LOSE")
				done = True
			else:
				game.step(best_act)
				st1, reward, done = game.curr_state

			if done:
				print("Score: " + str(t))
				scores.append(t)
				# if t > 300:
				# 	return
				DONE = True
			else:
				st = st1

	print(scores)
	print(np.mean(scores))




def main(_):
	FLAGS.log_dir += str(int(time.time()))
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	# train()
	# return

	run_dynamic()

	# env = MagneticEnvironment(6, 1000)
	# env.initial_state()
	# raw_data = env.run()

	# with open('./simulations/demo0.npy', 'wb') as f:
	# 	np.save(f, raw_data)

	# total_mses = 0
	# for i in range(10):

	# 	# raw_data = np.load("./simulations/demo"+str(i)+".npy")
	# 	# FLAGS.No = 7
	# 	# FLAGS.Nr = 42
	# 	# env = MagneticEnvironment(FLAGS.No-4, 1000)
	# 	# env.initial_state()
	# 	# raw_data = env.run()

	# 	raw_data = np.load("./demo13.npy")
	# 	data = np.zeros((998, FLAGS.No, FLAGS.Ds), dtype=float)
	# 	labels = np.zeros((998, FLAGS.No, FLAGS.Dp), dtype=float)
	# 	for ind in range(1, 999):
	# 		state_data = np.zeros((FLAGS.No, FLAGS.Ds), dtype=float)
	# 		state_label = np.zeros((FLAGS.No, FLAGS.Dp), dtype=float)
	# 		for j in range(FLAGS.No):
	# 			po1 = raw_data[ind-1][j] # previous state of current object
	# 			po2 = raw_data[ind][j] # current state of current object
	# 			po3 = raw_data[ind+1][j] # future state of current object
	# 			state_data[j] = np.array([po2[0],po1[1],po1[2],po2[1],po2[2],po2[5]])
	# 			state_label[j] = np.array([po3[1], po3[2]])
	# 		data[ind-1] = state_data
	# 		labels[ind-1] = state_label

	# 	num_steps = 100
	# 	indent = 0
	# 	p = predict_rollout('./saved_parameters/w18', data[indent:], num_steps)
	# 	# p = predict_rollout_rl('./saved_parameters/w18', data[0], num_steps)
		
	# 	from physics_sim import data_simulation
	# 	record_dir = "./simulation_images/rollout_demo_ground/"
	# 	record_dir2 = "./simulation_images/rollout_demo_sim/"
	# 	data_simulation(labels[indent:indent+num_steps], record=True, record_dir=record_dir)
	# 	data_simulation(p, record=True, record_dir=record_dir2)

	# 	return


		# print(p[50:60,0])
		# print(labels[50:60,0])

		# mse = np.mean(np.square(p - labels[indent:indent+num_steps]))
		# total_mses += mse

	# print(total_mses / 10)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default='/tmp/interaction-network/', help = 'summaries log directory')
	parser.add_argument('--No', type=int, default=10, help='number of objects') # 6 poles & 4 walls
	parser.add_argument('--Nr', type=int, default=90, help='number of relations') # 9 relations each
	parser.add_argument('--Ds', type=int, default=6, help='state dimension') # (x, y, v_x, v_y, obj_type)
	parser.add_argument('--De', type=int, default=50, help='effect dimension') # experiment w this
	parser.add_argument('--Dr', type=int, default=1, help='relationship dimension') # experiment
	parser.add_argument('--Dx', type=int, default=1, help='external effect dimension') # experiment
	parser.add_argument('--Dp', type=int, default=2, help='object model output dimension')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
