from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os

from math import sin, cos, radians, pi


total_time = 1000 # number of time steps
num_features = 5 # number of state features (e.g. mass, x, y, x_vel, y_vel)
G = 10**5 # He also had G = 6.67428e-11 written in original code??
diff_t = 0.001 # time step


def init(total_time, n_body, num_features, orbit):
  """
  Sets feature values for all n objects at t = 0.
  """
  data = np.zeros((total_time, n_body, num_features), dtype=float);
  if (orbit): # I DON'T NEED THIS BECAUSE I'M CONSIDERING MADE UP DYNAMICS NOT GRAVITY THINGS CIRCLING...
    data[0][0][0] = 100
    data[0][0][1:5] = 0.0
    for i in range(1, n_body):
      data[0][i][0] = np.random.rand() * 8.98 + 0.02
      distance = np.random.rand() * 90.0 + 10.0
      theta = np.random.rand() * 360
      theta_rad = pi/2 - radians(theta)
      data[0][i][1] = distance * cos(theta_rad)
      data[0][i][2] = distance * sin(theta_rad)
      data[0][i][3] = -1 * data[0][i][2] / norm(data[0][i][1:3]) * (G * data[0][0][0] / norm(data[0][i][1:3])**2) * distance / 1000
      data[0][i][4] = data[0][i][1] / norm(data[0][i][1:3]) * (G * data[0][0][0] / norm(data[0][i][1:3])**2) * distance / 1000
  else:
    for i in range(n_body):
      data[0][i][0] = np.random.rand() * 8.98 + 0.02 # why is this * 8.98
      distance = np.random.rand() * 90.0 + 10.0
      theta = np.random.rand() * 360
      theta_rad = pi / 2 - radians(theta)
      data[0][i][1] = distance * cos(theta_rad)
      data[0][i][2] = distance * sin(theta_rad)
      data[0][i][3] = np.random.rand()*6.0-3.0
      data[0][i][4] = np.random.rand()*6.0-3.0

  return data


def norm(x):
  """
  Calculate the 2-norm.
  """
  return np.sqrt(np.sum(x**2))


def get_f(reciever, sender):
  """
  Calculates the force acting between two objects in the model.
  """
  diff = sender[1:3] - reciever[1:3]
  dist = norm(diff)
  if (dist < 1):
    dist = 1

  return G * reciever[0] * sender[0] / (dist**3) * diff
 

def calc(cur_state, n_body):
  """
  Calculates next world state given previous
  """
  next_state = np.zeros((n_body, num_features), dtype=float)
  f_mat = np.zeros((n_body, n_body, 2), dtype = float) # matrix of forces between objects
  f_sum = np.zeros((n_body, 2), dtype = float) # agg force on each object
  acc = np.zeros((n_body, 2), dtype=float) # acceleration on each body
  for i in range(n_body):
    for j in range(i+1, n_body):
      if (j != i):
        f = get_f(cur_state[i][:3], cur_state[j][:3])
        f_mat[i,j] += f
        f_mat[j,i] -= f  
    f_sum[i] = np.sum(f_mat[i], axis=0) # combines all forces on one object into a 2-tuple of x and y direction forces
    acc[i] = f_sum[i] / cur_state[i][0]
    next_state[i][0] = cur_state[i][0]
    next_state[i][3:5] = cur_state[i][3:5] + acc[i] * diff_t
    next_state[i][1:3] = cur_state[i][1:3] + next_state[i][3:5] * diff_t

  return next_state


def gen(n_body, orbit):
  data = init(total_time, n_body, num_features, orbit)
  for i in range(1, total_time):
    data[i] = calc(data[i-1], n_body)

  return data


def make_video(xy, filename):
  os.system("rm -rf pics/*");
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
  writer = FFMpegWriter(fps=15, metadata=metadata)
  fig = plt.figure()
  plt.xlim(-200, 200)
  plt.ylim(-200, 200)
  fig_num = len(xy)
  color=['ro','bo','go','ko','yo','mo','co']
  with writer.saving(fig, filename, len(xy)):
    for i in range(len(xy)):
      for j in range(len(xy[0])):
        plt.plot(xy[i,j,1], xy[i,j,0], color[j%len(color)])
      writer.grab_frame()

if __name__=='__main__':
  data = gen(3, True)
  xy = data[:, :, 1:3]
  make_video(xy, "test.mp4")
