import os
import subprocess

from path_config import MODELS_PATH

def run(cmd):
	subprocess.run(cmd, shell=True, check=True)

def download_models():
	# SuperPoint
	run("curl -s -L -o sp_v6.tgz https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz")
	run("tar -zxf sp_v6.tgz && rm sp_v6.tgz")

	# DINOv2 ViT-B/14
	run("curl -s -L -o dinov2_vitb14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")

	# OmniGlue export
	run("curl -s -L -o og_export.zip https://storage.googleapis.com/omniglue/og_export.zip")
	run("unzip -qq og_export.zip && rm og_export.zip")

if __name__ == "__main__":
	os.makedirs(MODELS_PATH, exist_ok=True)
	os.chdir(MODELS_PATH)
	
	print(f"Downloading models to {MODELS_PATH} ...")
	download_models()
	print(f"✅ Models ready in: {MODELS_PATH}")
	