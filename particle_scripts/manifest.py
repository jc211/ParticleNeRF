from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import json
from PIL import Image
from camera import *
import shutil
from common import *

def calculate_fov(f, dim):
	# fovx = calculate_fov(f_x, w) 
	# fovy = calculate_fov(f_y, h) 
	return np.arctan(dim/(f*2))*2 * 180/np.pi

@dataclass
class Frame:
	manifest: Any
	name: str
	X_WV: np.ndarray 
	file_path: Path = None
	rgb_path: Optional[Path] = None
	depth_path: Optional[Path] = None
	alpha_path: Optional[Path] = None
	has_depth: bool = False

	def depth_scale(self):
		return self.manifest.depth_scale

	def width(self):
		return self.manifest.w

	def height(self):
		return self.manifest.h

	def fov_x(self):
		return self.manifest.fov_x
	
	def fx(self):
		return self.manifest.fx

	def fy(self):
		return self.manifest.fy

	def cx(self):
		return self.manifest.cx

	def cy(self):
		return self.manifest.cy
	
	def preload(self):
		self.get_rgb()
		self.get_depth()

	def get_full_alpha(self):
		alpha = Image.open(self.alpha_path)
		alpha = np.asarray(alpha, dtype=np.float32)
		return alpha

	def get_camera(self):
		X_WV = np.eye(4)
		X_WV[:3, :] = self.X_WV
		return Camera(X_WV, self.manifest.K, self.manifest.w, self.manifest.h)

	def get_alpha(self, obj_id = 1):
		alpha = Image.open(self.alpha_path)
		alpha = np.asarray(alpha)
		if obj_id == None:
			return alpha[..., 0] != 0
		return alpha[..., 0] == obj_id

	def get_depth(self):
		depth = Image.open(self.depth_path)
		depth = np.array(depth, dtype=np.float32) 
		return depth
	
	def get_rgb(self, background_color = np.array([0, 0, 0, 1])):
		rgb = Image.open(self.rgb_path).convert(mode="RGBA")
		rgb = np.array(rgb, dtype=np.float32)/255.0
		return rgb

	def get_alpha_depth(self, instance_id=None):
		mask = self.get_alpha(instance_id)
		depth = self.get_depth()
		depth[~mask] = 0
		return depth
	
	def get_alpha_rgb(self, instance_id=None, background_color = np.array([0, 0, 0, 1])):
		mask = self.get_alpha(instance_id)
		rgb = self.get_rgb(background_color)
		rgb[~mask] *=0
		return rgb 	

@dataclass
class SceneObject:
	name: str
	bound_box: np.ndarray
	X_WO: np.ndarray
	instance_id: int

class Manifest:
	def __init__(self, manifest_path: Path):
		assert manifest_path.exists()
		self.path = manifest_path
		self.manifest = self.load_manifest(manifest_path)

		if "w" in self.manifest and "h" in self.manifest:
			self.w = self.manifest["w"]
			self.h = self.manifest["h"]
		else:
			# read first image and find out
			rgb_path = Path(self.manifest["frames"][0]["file_path"])
			if not rgb_path.name.lower().endswith(('.png', '.jpg', '.jpeg')):
				rgb_path = rgb_path.with_suffix(".png")
			rgb_path = self.path.parent / rgb_path
			rgb = Image.open(rgb_path)
			self.w, self.h = rgb.size
		
		self.fx = 0
		self.fy = 0
		
		if "fl_x" in self.manifest and "fl_y" in self.manifest:
			self.fx = self.manifest["fl_x"]
			self.fy = self.manifest["fl_y"]
			self.cx = self.manifest["cx"]
			self.cy = self.manifest["cy"]
		elif "camera_angle_x" in self.manifest:
			self.fov_x = self.manifest["camera_angle_x"] * 180/np.pi
			self.fx = self.w / (2 * np.tan(self.fov_x * np.pi / 360))
			self.cx = self.w / 2
			self.fy = self.fx
			self.cy = self.h / 2
				
		self.K = create_projection_matrix(self.fx, self.fy, self.cx, self.cy)
		self.has_depth = False
		if "aabb" in self.manifest:  
			self.aabb = np.asarray(self.manifest["aabb"], dtype=np.float32)
		if "integer_depth_scale" in self.manifest:
			self.depth_scale = self.manifest["integer_depth_scale"]
			self.has_depth = True
		else: 
			self.depth_scale = 1.0
		self.fov_x = calculate_fov(self.fx, self.w)
		self.fov_y = calculate_fov(self.fy, self.h)
		self.n_frames = len(self.manifest["frames"])
		self.frames = [self.get_frame(self.manifest, i) for i in range(self.n_frames)]
		self.frames = list(filter(lambda x: x != None, self.frames))
		self.cameras = [f.get_camera() for f in self.frames] 
		self.n_frames = len(self.frames)

		self.objects: List[SceneObject] = []
		if "objects" in self.manifest:
			for obj in self.manifest["objects"]:
				scene_object = SceneObject(name=obj["name"], bound_box=np.array(obj["bound_box"], dtype=np.float32), X_WO=np.array(obj["X_WO"], dtype=np.float32), instance_id = obj["instance_id"])
				self.objects.append(scene_object)

	def load_manifest(self, manifest_path):
		with open(manifest_path) as f:
			manifest = json.load(f)
		return manifest
	
	def get_frame(self, manifest_json, i):
		frame = manifest_json["frames"][i]
		xform = np.array(frame["transform_matrix"])[:-1,:]
		image_path = Path(frame["file_path"])
		folder = image_path.parent.name
		base_path = self.path.parent.joinpath(folder)
		image_name = image_path.stem
		if image_path.name.lower().endswith((".png", ".jpg", ".jpeg")):
			rgb_path = base_path.joinpath(f"{image_path.name}")
		else:
			rgb_path = base_path.joinpath(f"{image_name}.png")
		alpha_path = base_path.joinpath(f"{image_name}.alpha.png")
		depth_path = base_path.joinpath(f"{image_name}.depth.png")
		frame = Frame(
			name=image_name, 
			X_WV=xform, 
			manifest=self,
			file_path=image_path,
			)
		if rgb_path.exists():
			frame.rgb_path = rgb_path
		else:
			print(f"Warning frame {rgb_path} does not exist")
			return None
		if alpha_path.exists():
			frame.alpha_path = alpha_path
		if depth_path.exists():
			frame.depth_path = depth_path
			frame.has_depth = True
		return frame 
	
	def get_camera(self, manifest_json, i):
		frame = manifest_json["frames"][i]
		X_WV = np.array(frame["transform_matrix"], dtype=np.float32)
		return Camera(X_WV= X_WV, K=self.K, image_width=self.w, image_height=self.h)
	
	def save(self, target_path: Path):
		shutil.copytree(self.path, target_path)
