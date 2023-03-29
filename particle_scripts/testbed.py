import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).joinpath("../../build").resolve()))
import numpy as np
from tqdm import tqdm
import json

import pyngp as ngp

from manifest import Manifest, linear_to_srgb


class Testbed:
	def __init__(self, gui: bool = False, config: str = "base"):
		ngp.free_temporary_memory()
		self.testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
		if type(config) == str:
			self.testbed.reload_network_from_file(f"configs/nerf/{config}.json")
		else:
			self.testbed.reload_network_from_json(config)

		self.testbed.display_gui = False
		self.testbed.nerf.training.random_bg_color = True
		self.testbed.render_mode = ngp.RenderMode.Shade
		if gui:
			self.testbed.init_window(640, 480)

	def set_n_training_images(self, n:int):
		self.testbed.nerf.training.n_images_for_training = n

	def load_scene(self, scene: Path):
		self.testbed.load_training_data(str(scene))

	def debug(self, val=True):
		self.testbed.nerf.visualize_cameras = val	
		self.testbed.visualize_unit_cube = val	

	def set_depth_supervision(self, val:float):
		self.testbed.nerf.training.depth_supervision_lambda = val
	
	def training_step(self):
		return self.testbed.training_step

	def create_empty_dataset(self, n_images: int):
		self.testbed.create_empty_nerf_dataset(n_images=n_images)

	def _render(self, width: int, height: int, xform: np.ndarray, fov_x: float, spp:int=1):
		testbed = self.testbed
		testbed.shall_train = False
		testbed.fov_axis = 0
		testbed.fov = fov_x
		testbed.set_nerf_camera_matrix(xform)
		img = testbed.render(int(width), int(height), spp, True)  # linear space
		return img

	def render_rgb(self, width: int, height: int, xform: np.ndarray, fov_x: float, spp:int=1):
		tmp_render_mode = self.testbed.render_mode
		self.testbed.render_mode = ngp.RenderMode.Shade
		tmp = self.testbed.background_color
		self.testbed.background_color = np.array([0, 0,0,1])
		img = self._render(width, height, xform, fov_x, spp)
		self.testbed.background_color = tmp
		self.testbed.render_mode = tmp_render_mode
		img[..., :3] = np.clip(linear_to_srgb(img[..., :3]), 0.0, 1.0)
		return img

	def render_depth(self, width: int, height: int, xform: np.ndarray, fov_x: float, spp:int=1):
		tmp_render_mode = self.testbed.render_mode
		self.testbed.render_mode = ngp.RenderMode.Depth
		tmp = self.testbed.background_color
		self.testbed.background_color = np.array([0, 0,0,1.0])
		img = self._render(width, height, xform, fov_x, spp)
		self.testbed.background_color = tmp
		self.testbed.render_mode = tmp_render_mode
		return img[..., 0]

	def render_positions(self, width: int, height: int, xform: np.ndarray, fov_x: float, spp:int=1):
		self.testbed.render_mode = ngp.RenderMode.Positions
		tmp = self.testbed.background_color
		self.testbed.background_color = np.array([0, 0,0,0])
		img = self._render(width, height, xform, fov_x, spp)
		self.testbed.background_color = tmp
		return img  

	def train(self, n_steps: int):
		losses = []
		self.testbed.shall_train = True
		# self.testbed.render_mode = ngp.RenderMode.Positions
		old_training_step = 0
		while self.testbed.frame():
			# What will happen when training is done?
			if self.testbed.training_step >= n_steps:
				# self.testbed.shall_train = False
				# continue
				break
			if self.testbed.shall_train and self.testbed.training_step > 0:
				losses.append(self.testbed.loss)
		return losses