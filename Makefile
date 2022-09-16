environment_cu111:
	pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
	pip install torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
	pip install -r requirements.txt
	pip install -e .
.PHONY: install

environment_cpu:
	conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
	pip install torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
	pip install -r requirements.txt
	pip install -e .
.PHONY: install
