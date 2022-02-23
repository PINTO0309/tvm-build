# tvm-build
## 1. build
```bash
docker build -t pinto0309/ubuntu2004-cuda114-cudnn8-tensorrt823-tvm:09dev0 .
```
## 2. run
```bash
docker run --rm -it --gpus all \
-v `pwd`:/home/user/workdir \
pinto0309/ubuntu2004-cuda114-cudnn8-tensorrt823-tvm:09dev0
```
