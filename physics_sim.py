from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pymunk
from pymunk import pygame_util
import pygame
from pygame.locals import *
import random
import sys
import time

from copy import copy
from math import ceil, pi


MU = 4 * pi * 1e-7
T_DIFF = 0.001
SAVE_IMAGE_DIR = './saved_simulation_images/'


class CircleMagnet(object):
	"""
	Class for a spherical (point) magnet.
	"""
	def __init__(self, x=None, y=None, m=None, q=None, r=None, v_x=None, v_y=None):
		self.m = m # NB: This is stored as inverse mass
		self.r = r
		self.x = x
		self.y = y
		self.q = q
		self.v_x = v_x
		self.v_y = v_y
		self.shape = None
		self.t = 0


class WallMagnet(object):
	"""
	Class for a wall magnet used in the magnetic simulation.
	"""
	def __init__(self, x1=None, y1=None, x2=None, y2=None, q=None):
		self.m = 0
		self.x1 = x1
		self.x2 = x2
		self.x = (x1 + x2) / 2.0
		self.y1 = y1
		self.y2 = y2
		self.y = (y1 + y2) / 2.0
		self.v_x = 0
		self.v_y = 0
		self.q = q
		self.t = 1


class Agent(object):
	def __init__(self, x=None, y=None, shape=None):
		self.r = 0.05
		self.m = 120
		self.x = x
		self.y = y
		self.v_mag = 2.5
		self.shape = shape


def data_simulation(data, record=False, record_dir=None): # format data here so it's (num_steps, No, 2)
	"""
	Visualizes a sequence of game states input as data as a simulation.
	"""
	pygame.init()
	screen = pygame.display.set_mode((ceil(data[0,7,0]) * 100, ceil(data[0,8,1]) * 100))
	clock = pygame.time.Clock()
	space = pymunk.Space()
	draw_options = pygame_util.DrawOptions(screen)

	curr_shapes = []
	for j, ts in enumerate(data):
		for s in curr_shapes:
			space.remove(s, s.body)
		curr_shapes.clear()
		for i in range(len(ts) - 4): # only need to visualise circles
			obj = ts[i]
			body = pymunk.Body()
			body.position = obj[0] * 100, obj[1] * 100
			shape = pymunk.Circle(body, 0.15 * 100)
			space.add(shape, body)
			curr_shapes.append(shape)
		for event in pygame.event.get():
			if event.type == QUIT:
				sys.exit(0)
			elif event.type == KEYDOWN and event.key == K_ESCAPE:
				sys.exit(0)

		if record:
			pygame.image.save(screen, record_dir+str(j)+".jpeg")

		screen.fill((255,255,255))
		space.debug_draw(draw_options)
		pygame.display.flip()
		pygame.display.set_caption("fps: " + str(clock.get_fps()))
		space.step(T_DIFF)
		clock.tick(1 / T_DIFF)

	pygame.quit()


class MagneticEnvironment(object):
	"""
	Class for a projectile motion physics simulator.
	Detailed outline in project README.
	"""
	def __init__(self, num_objects, num_timesteps, vis=False, record=False):
		self.vis = vis
		self.record = record
		self.No = num_objects
		self.T = num_timesteps
		self.max_x = max_x = ceil(3 + np.random.rand() * 3) # 3-6m
		self.max_y = max_y = ceil(3 + np.random.rand() * 3) # 3-6m
		self.left_wall = WallMagnet(0.0, 0.0, 0.0, max_y, 1e5)
		self.right_wall = WallMagnet(max_x, 0.0, max_x, max_y, 1e5)
		self.top_wall = WallMagnet(0.0, max_y, max_x, max_y, 1e5)
		self.bottom_wall = WallMagnet(0.0, 0.0, max_x, 0.0, 1e5)
		self.walls = [self.left_wall, self.right_wall, self.top_wall, self.bottom_wall]
		self.data = [[]]

	def initial_state(self):
		"""
		Sets up the initial state of the magnetic environment
		"""
		if self.vis:
			pygame.init()
			self.screen = pygame.display.set_mode((self.max_x*100, self.max_y*100)) # 100pixels/meter
			self.clock = pygame.time.Clock()
			self.space = pymunk.Space()
			self.draw_options = pygame_util.DrawOptions(self.screen)

		obj_positions = []
		for i in range(self.No): # create No magnetic poles
			yes = True
			while yes: # loops until we get an initialization of objects with no collisions
				r = 0.15 # + np.random.rand() * 0.1 # 0.1-0.2m
				x = r + np.random.rand() * (self.max_x - 2*r)
				y = r + np.random.rand() * (self.max_y - 2*r)
				collision = False
				for o in obj_positions:
					if 2*r >= np.sqrt((x-o[0])**2 + (y-o[1])**2):
						collision = True
						continue
				if not collision:
					yes = False
					obj_positions.append((x,y,r))

			m = 1 / (0.75 + np.random.rand() * 0.5) # 0.75-1.25kg - stored as m^-1
			v_x = -5 + np.random.rand() * 10 # -5-5m/s
			v_y = -5 + np.random.rand() * 10 # -5-5m/s
			q = 7.5 * 1e3
			obj = CircleMagnet(x, y, m, q, r, v_x, v_y)

			if self.vis:
				obj.shape = self.add_obj(obj)
				self.space.add(obj.shape, obj.shape.body)
			
			self.data[0].append(obj)
		self.data[0].extend(self.walls)

	def run(self):
		"""
		Calculates the next world state given the previous, T times.
		"""
		for t in range(1, self.T):

			if self.vis:
				for event in pygame.event.get():
					if event.type == QUIT:
						sys.exit(0)
					elif event.type == KEYDOWN and event.key == K_ESCAPE:
						sys.exit(0)

			if self.record:
				pygame.image.save(self.screen, SAVE_IMAGE_DIR+"trial/"+str(t)+".jpeg")

			f_matrix = np.zeros((self.No, self.No+4, 2), dtype=np.float64) # matrix of forces between objects
			f_sum = np.zeros((self.No+4, 2), dtype=np.float64) # agg force on each object
			acc = np.zeros((self.No, 2), dtype=np.float64) # acc on each circle
			old_state = self.data[-1]
			new_state = []

			for i in range(self.No): # receiver poles
				old_obj = old_state[i]
				for j in range(self.No): # sender poles
					fij = self.get_circle_circle_f(old_obj, old_state[j])
					f_matrix[i,j] += fij
					f_matrix[j,i] -= fij
				for k, w in enumerate(self.walls): # sender walls
					fw = self.get_circle_wall_f(old_obj, w)
					f_matrix[i,self.No+k] += fw
				f_sum[i] = np.sum(f_matrix[i], axis=0)
				acc[i] = f_sum[i] * old_obj.m
				new_obj = CircleMagnet(m=old_obj.m, q=old_obj.q, r=old_obj.r)
				new_obj.v_x = old_obj.v_x + acc[i][0] * T_DIFF
				new_obj.v_y = old_obj.v_y + acc[i][1] * T_DIFF
				delta_x = old_obj.v_x * T_DIFF + 0.5 * acc[i][0] * T_DIFF**2
				delta_y = old_obj.v_y * T_DIFF + 0.5 * acc[i][1] * T_DIFF**2
				new_obj.x = old_obj.x + delta_x
				new_obj.y = old_obj.y + delta_y

				if self.oob(new_obj.x, new_obj.y, new_obj.r):
					print("DOOT DOOT")
					self.data = [[]]
					pygame.quit()
					self.initial_state()
					return self.run()

				if self.vis:
					self.space.remove(old_obj.shape, old_obj.shape.body)
					new_obj.shape = self.add_obj(new_obj)
					self.space.add(new_obj.shape, new_obj.shape.body)
				
				new_state.append(new_obj)

			new_state.extend(self.walls)
			self.data.append(new_state)

			if self.vis:
				self.screen.fill((255,255,255))
				self.space.debug_draw(self.draw_options)
				pygame.display.flip()
				pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
				self.space.step(T_DIFF)
				self.clock.tick(1 / T_DIFF)

		new_data = np.zeros((1000, self.No+4, 6), dtype = np.float64)
		for i, state in enumerate(self.data):
			for j, obj in enumerate(state):
				new_data[i, j] = np.array([obj.m, obj.x, obj.y, obj.v_x, obj.v_y, obj.t])

		if self.vis:
			pygame.quit()

		return new_data

	def oob(self, x, y, r):
		"""
		Indicates whether object has moved outside the arena.
		"""
		if x  < 0 or y < 0 or x > self.max_x or y > self.max_y:
			return True
		else:
			return False

	def get_circle_circle_f(self, r, s):
		"""
		Calculate the pole-pole magnetic repulsion force between two circles.
		"""
		diff = np.array([r.x - s.x, r.y - s.y])
		dist = np.linalg.norm(diff) - r.r - s.r
		if dist < 0.1:
			dist = 0.1

		return (MU * r.q * s.q) / (4 * pi * dist**2) * diff

	def get_circle_wall_f(self, c, w):
		"""
		Calculate the pole-pole magnetic repulsion force between a circle and a wall.
		"""
		if w.x1 - w.x2 == 0:
			diff = np.array([c.x - w.x1, 0])
		else:
			diff = np.array([0, c.y - w.y1])
		dist = np.linalg.norm(diff)

		return (MU * w.q * c.q) / (4 * pi * dist**2) * diff

	def add_obj(self, obj):
		"""
		Adds an object to the pymunk/pygame simulator
		"""
		moment = pymunk.moment_for_circle(obj.m, 0, obj.r * 100)
		body = pymunk.Body(obj.m, moment)
		body.position = obj.x * 100, obj.y * 100
		
		return pymunk.Circle(body, obj.r * 100)


class MagneticGame(object):
	"""
	Class for a game based on magnetic environment defined above.
	"""
	def __init__(self, num_objects, max_x=3, max_y=3, wall_q=1e5, vis=False, record=False):
		self.Ds = 22
		self.No = num_objects
		self.vis = vis
		self.record = record
		self.max_x, self.max_y = max_x, max_y
		self.action_space = {0: np.array([0,1]), 1: np.array([1,0]), 2: np.array([0,-1]), 3: np.array([1,0])} # n,e,s,w
		self.left_wall = WallMagnet(0.0, 0.0, 0.0, max_y, wall_q)
		self.right_wall = WallMagnet(max_x, 0.0, max_x, max_y, wall_q)
		self.top_wall = WallMagnet(0.0, max_y, max_x, max_y, wall_q)
		self.bottom_wall = WallMagnet(0.0, 0.0, max_x, 0.0, wall_q)
		self.walls = [self.left_wall, self.right_wall, self.top_wall, self.bottom_wall]
		self.data = [[]]
		self.init_state = np.zeros(((self.No+4)*2+2), dtype=float) # x,y for all objects and the agent
		self.curr_state = None, 0.0, False # state, reward, done
		self.agent = Agent()
		self.t = 0

	def reset(self):
		"""
		Simply returns the initial state of the game
		"""
		if self.vis:
			pygame.init()
			self.screen = pygame.display.set_mode((ceil(self.max_x) * 100, ceil(self.max_y) * 100)) # 100pixels/meter
			self.clock = pygame.time.Clock()
			self.space = pymunk.Space()
			self.draw_options = pygame_util.DrawOptions(self.screen)

		# Initialise objects in game
		obj_positions = []
		for i in range(self.No):
			yes = True
			while yes: # loops until we get an initialization of objects with no collisions
				r = 0.15
				x = r + np.random.rand() * (self.max_x - 2*r)
				y = r + np.random.rand() * (self.max_y - 2*r)
				collision = False
				for o in obj_positions:
					if 2*r >= np.sqrt((x-o[0])**2 + (y-o[1])**2):
						collision = True
						continue
				if not collision:
					yes = False
					obj_positions.append((x,y,r))

			m = 1.0 / (0.75 + np.random.rand() * 0.5) # 0.75-1.25kg - stored as m^-1
			v_x = -5 + np.random.rand() * 10 # -5-5m/s
			v_y = -5 + np.random.rand() * 10 # -5-5m/s
			q = 7.5 * 1e3
			obj = CircleMagnet(x, y, m, q, r, v_x, v_y)
			
			if self.vis:
				obj.shape = self.add_obj(obj)
				self.space.add(obj.shape, obj.shape.body)
			
			self.data[0].append(obj)
		self.data[0].extend(self.walls)

		# Initialize agent
		yes = True
		while yes:
			x = np.random.rand() * self.max_x
			y = np.random.rand() * self.max_y
			collision = False
			for o in obj_positions:
				if 0.15+self.agent.r >= np.sqrt((x-o[0])**2 + (y-o[1])**2): # 0.2 = sum of agent and object radii
					collision = True
					continue
			if not collision:
				yes = False
				self.agent.x, self.agent.y = x, y

		# add agent to visualization
		if self.vis:
			self.agent.shape = self.add_agent(self.agent)
			self.space.add(self.agent.shape, self.agent.shape.body)

		# Convert state to RL format for output
		prev_state = self.data[0]
		for i in range(0, (self.No)*2, 2):
			self.init_state[i] = prev_state[(i+1)//2].x
			self.init_state[i+1] = prev_state[(i+1)//2].y
		self.init_state[(self.No+4)*2] = self.agent.x
		self.init_state[(self.No+4)*2+1] = self.agent.y


	def step(self, action):
		"""
		Takes in a chosen action, moves forward 1 time step, and returns new state, reward, and indicator for whether game is over
		"""
		if self.vis:
			for event in pygame.event.get():
				if event.type == QUIT:
					sys.exit(0)
				elif event.type == KEYDOWN and event.key == K_ESCAPE:
					sys.exit(0)

		if self.record:
			pygame.image.save(self.screen, "./simulation_images/game_demo/"+str(self.t)+".jpeg")
		self.t += 1

		# update object positions
		f_matrix = np.zeros((self.No, self.No+4, 2), dtype=np.float64) # matrix of forces between objects
		f_sum = np.zeros((self.No+4, 2), dtype=np.float64) # agg force on each object
		acc = np.zeros((self.No, 2), dtype=np.float64) # acc on each circle
		old_state = self.data[-1] # previous list of objects
		new_state = []
		for i in range(self.No): # receiver poles
			old_obj = old_state[i]
			for j in range(self.No): # sender poles
				fij = self.get_circle_circle_f(old_obj, old_state[j])
				f_matrix[i,j] += fij
				f_matrix[j,i] -= fij
			for k, w in enumerate(self.walls): # sender walls
				fw = self.get_circle_wall_f(old_obj, w)
				f_matrix[i,self.No+k] += fw
			f_sum[i] = np.sum(f_matrix[i], axis=0)
			acc[i] = f_sum[i] * old_obj.m
			new_obj = CircleMagnet(m=old_obj.m, q=old_obj.q, r=old_obj.r)
			new_obj.v_x = old_obj.v_x + acc[i][0] * T_DIFF
			new_obj.v_y = old_obj.v_y + acc[i][1] * T_DIFF
			delta_x = old_obj.v_x * T_DIFF + 0.5 * acc[i][0] * T_DIFF**2
			delta_y = old_obj.v_y * T_DIFF + 0.5 * acc[i][1] * T_DIFF**2
			new_obj.x = old_obj.x + delta_x
			new_obj.y = old_obj.y + delta_y

			if self.oob(new_obj.x, new_obj.y, new_obj.r):
				if self.vis:
					self.space.remove(old_obj.shape, old_obj.shape.body)
				new_obj.x = 0
				new_obj.y = 0
				new_obj.m = 0

			if self.vis:
				self.space.remove(old_obj.shape, old_obj.shape.body)
				new_obj.shape = self.add_obj(new_obj)
				self.space.add(new_obj.shape, new_obj.shape.body)

			new_state.append(new_obj)

		new_state.extend(self.walls)
		self.data.append(new_state)

		# update agent position
		self.agent.x += self.agent.v_mag * self.action_space[action][0] * T_DIFF
		self.agent.y += self.agent.v_mag * self.action_space[action][1] * T_DIFF

		if self.vis:
			self.space.remove(self.agent.shape, self.agent.shape.body)
			self.agent.shape = self.add_agent(self.agent)
			self.space.add(self.agent.shape, self.agent.shape.body)

		curr_state = np.zeros(((self.No+4)*2+2), dtype=float)
		for i in range(0,(self.No+4)*2,2):
			curr_state[i] = new_state[(i+1)//2].x
			curr_state[i+1] = new_state[(i+1)//2].y
		curr_state[(self.No+4)*2] = self.agent.x
		curr_state[(self.No+4)*2+1] = self.agent.y

		# calculate new state, reward and game-done indicator (collision into objects or boundaries)
		reward = 1.0
		done = False

		ax, ay = self.agent.x, self.agent.y
		for o in self.data[-1][:self.No]:
			x, y = o.x, o.y
			if 0.15 + self.agent.r > np.sqrt((x-ax)**2 + (y-ax)**2):
				reward = 0
				done = True
		if (self.agent.x + self.agent.r > self.max_x) or (self.agent.x - self.agent.r) < 0:
			reward = 0
			done = True
		elif (self.agent.y + self.agent.r) > self.max_y or (self.agent.y - self.agent.r) < 0:
			reward = 0
			done = True

		# visualize changes
		if self.vis:
			self.screen.fill((255,255,255))
			self.space.debug_draw(self.draw_options)
			pygame.display.flip()
			pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
			self.space.step(T_DIFF)
			self.clock.tick(1 / T_DIFF)

		# pygame.image.save(self.screen, "./random/x" + str(pygame.time.get_ticks()) + ".jpeg")
		self.curr_state = curr_state, reward, done
		if done and self.vis: pygame.quit()

	def get_actions(self):
		"""
		Returns possible actions for an agent.
		"""
		return list(self.action_space.keys())

	def get_circle_circle_f(self, r, s):
		"""
		Calculate the pole-pole magnetic repulsion force between two circles.
		"""
		diff = np.array([r.x - s.x, r.y - s.y])
		dist = np.linalg.norm(diff) - r.r - s.r

		return (MU * r.q * s.q) / (4 * pi * dist**2) * diff

	def get_circle_wall_f(self, c, w):
		"""
		Calculate the pole-pole magnetic repulsion force between a circle and a wall.
		"""
		if w.x1 - w.x2 == 0:
			diff = np.array([c.x - w.x1, 0])
		else:
			diff = np.array([0, c.y - w.y1])
		dist = np.linalg.norm(diff)

		return (MU * w.q * c.q) / (4 * pi * dist**2) * diff

	def add_obj(self, obj):
		"""
		Adds an object to the pymunk/pygame simulator
		"""
		moment = pymunk.moment_for_circle(obj.m, 0, obj.r * 100)
		body = pymunk.Body(obj.m, moment)
		body.position = obj.x * 100, obj.y * 100
		
		return pymunk.Circle(body, obj.r * 100)

	def add_agent(self, agent):
		"""
		Adds an agent to the visualizer
			r = 0.05
			m = 120
		"""
		moment = pymunk.moment_for_circle(120, 0, agent.r*100) # initialize as 120kg i.e. wouldn't be too affected by forces??
		body = pymunk.Body(120, moment)
		body.position = agent.x*100, agent.y*100

		return pymunk.Circle(body, agent.r*100)

	def oob(self, x, y, r):
		"""
		Indicates whether object has moved outside the arena.
		"""
		if x  < 0 or y < 0 or x > self.max_x or y > self.max_y:
			return True
		else:
			return False


def main():
	# for i in range(10):
	env = MagneticEnvironment(6, 1000, vis=False, record=False)
	env.initial_state()
	data = env.run()
	print(data)
	with open("./demo"+str(13)+".npy", 'wb') as f:
		np.save(f, data)

	# game = MagneticGame(6, vis=True)
	# game.reset()

	# data_simulation(np.load("./sim_data2.npy"))

	# game = MagneticGame(6, vis=True) # this should return the initial state
	# game.initial_state()
	# for i in range(1000):
	# 	game.step(random.randint(0,3))
	# 	if game.curr_state[2]:
	# 		print("DOOT DOOT")
	# 		break



if __name__ == '__main__':
	main()
