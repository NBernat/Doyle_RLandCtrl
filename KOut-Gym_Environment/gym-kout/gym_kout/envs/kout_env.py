import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class KoutEnv(gym.Env):
	metadata = {
		'render.modes' : ['human', 'rgb_array'],
		'video.frames_per_second' : 30
	}

	def __init__(self, a=0, mu=0, sigma=0, reg=0.001, horiz=5, seed=None):

		self.A = np.array([[1+a, 0],[0,1]])
		self.B = np.array([1,1])

		self.mu = mu #noise parameters
		self.sigma = sigma

		self.reg = reg #regularization parameter for least squares 

		self.max_x1=100
		self.max_x2=100
		self.max_k =100

		self.start_zone = 2 # only used for no noise, see reset()
		self.fail_thresh = 1e5 #using horizon_len instead of fail_thresh
		self.horizon_len = horiz
		self.num_steps = None

		self.viewer = None

		high = np.array([self.max_x1, self.max_x2])
		self.action_space = spaces.Box(low=-self.max_k, high=self.max_k, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

		self.state = None
		self.seed(seed)


	def seed(self, seed=None):

		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def noise(self):

		return np.random.normal(self.mu, self.sigma, 2)

	def step(self,k):

		self.last_k = k
		u = np.matmul(k,self.state)

		# Calculate new state
		self.state = np.matmul(self.A,self.state) \
			+ u*self.B \
			+ self.noise()

		# Squared error, normed by horiz_len (for comparisons between horiz's)
		costs = (np.dot(self.state, self.state) \
			+ self.reg*np.dot(u,u)) \
			* (1.0/self.horizon_len)

		self.num_steps += 1

		# Has it gone too far?
		done = self.check()

		return self.state, -costs, done, {}

	def check(self):

		x1 = self.state[0]
		x2 = self.state[1]

		done = x1 < -self.fail_thresh \
			or x1 > self.fail_thresh \
			or x2 < -self.fail_thresh \
			or x2 > self.fail_thresh \
			or self.num_steps >= self.horizon_len
		done = bool(done)

		return done

	def reset(self):

		if self.sigma==0:
			self.state = self.np_random.uniform(low=-self.start_zone, high=self.start_zone, size=(2,))
		else:
			self.state = np.array([0,0])

		self.num_steps = 0
		self.last_u = None

		return self.state

	def render(self, mode='human'):

		screen_w = 500
		screen_h = 500
		origin = [screen_w/2.0, screen_h/2.0]

		dot_w = 10.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_w,screen_h)
			# thresh = rendering.draw_circle(self.fail_thresh, 30, False)

			dot = rendering.make_circle(dot_w/2.0)
			dot.set_color(0,0,0)
			self.dottrans = rendering.Transform()
			dot.add_attr(self.dottrans)
			self.viewer.add_geom(dot)
			self._dot_geom = dot


		self.dottrans.set_translation(self.state[0], self.state[1])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None


