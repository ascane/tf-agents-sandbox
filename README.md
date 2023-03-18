# tf-agents-sandbox

## Setup

### Create virtual env
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install tf-agents[reverb]
```

### Problem shooting

#### Missing libraries
https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7
```
$ pip install tensorrt

$ cd venv/lib/python3.9/site-packages/tensorrt
```
Create symbolic links
```
$ ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7
$ ln -s libnvinfer.so.8 libnvinfer.so.7
```

Add tensorrt to library path
```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/venv/lib/python3.9/site-packages/tensorrt
```

#### Incompatibility between numpy and TF agents

Downgrade numpy for compatibility with TF agents
https://stackoverflow.com/questions/74893742/how-to-solve-attributeerror-module-numpy-has-no-attribute-bool
```
$ python -m pip uninstall numpy
$ python -m pip install numpy==1.23.1
```

#### Cuda not linked properly

Error message
```
W tensorflow/compiler/xla/service/gpu/nvptx_helper.cc:56 Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may result in compilation or runtime failures, if the program we try to run uses routines from libdevice.
Searched for CUDA in the following directories:
  ./cuda_sdk_lib
  /usr/local/cuda-11.2
  /usr/local/cuda
  .
You can choose the search directory by setting xla_gpu_cuda_data_dir in HloModule's DebugOptions.  For most apps, setting the environment variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.
```

```
$ export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

## Test the SDE environment
```
$ python src/envs/sde_env_test.py
```

## Run the RL training script on the SDE environment
```
$ python src/examples/rl_sde_example.py
```
