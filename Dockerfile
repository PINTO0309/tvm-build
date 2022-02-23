# https://github.com/PINTO0309/openvino2tensorflow#1-environment
FROM ghcr.io/pinto0309/openvino2tensorflow:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update \
    && sudo apt-get install -y \
        software-properties-common \
        libmkldnn-dev \
    && sudo apt clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Ninja
RUN sudo pip3 install pip --upgrade \
    && sudo pip install ninja gluoncv dgl

# LLVM
RUN wget https://apt.llvm.org/llvm.sh \
    && sudo chmod +x llvm.sh \
    && sudo ./llvm.sh 14 \
    && sudo apt clean \
    && sudo rm -rf /var/lib/apt/lists/*

# NNPACK
RUN git clone --recursive https://github.com/Maratyszcza/NNPACK.git \
    && cd NNPACK \
    && git checkout c07e3a0400713d546e0dea2d5466dd22ea389c73 \
    && sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt \
    && sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt \
    && mkdir build \
    && cd build \
    && cmake -G Ninja -D BUILD_SHARED_LIBS=ON .. \
    && ninja \
    && sudo ninja install \
    && sudo sh -c "echo '/usr/local/lib' > /etc/ld.so.conf.d/nnpack.conf" \
    && sudo ldconfig

# TVM
RUN git clone --recursive https://github.com/apache/tvm tvm
RUN cd tvm \
    # 0.9.dev0
    && git checkout faa2e6ad578f09b1248b5303b17c3fa24ebd0bd6 \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake \
        -DUSE_CUDA=ON \
        # -DUSE_MICRO=ON \
        # -DUSE_MICRO_STANDALONE_RUNTIME=ON \
        -DUSE_CPP_RPC=ON \
        -DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON \
        -DUSE_PIPELINE_EXECUTOR=ON \
        -DUSE_RPC=ON \
        -DUSE_LLVM=ON \
        -DUSE_MKLDNN=ON \
        -DUSE_OPENMP=ON \
        -DUSE_NNPACK=ON \
        # -DUSE_TFLITE=ON \
        # -DUSE_EDGETPU=ON \
        -DUSE_CUDNN=ON \
        -DUSE_TENSORRT_CODEGEN=ON \
        -DUSE_TENSORRT_RUNTIME=ON \
        -DUSE_TF_TVMDSOOP=ON \
        -DUSE_PT_TVMDSOOP=ON \
        -DUSE_TARGET_ONNX=ON \
        -DBUILD_STATIC_RUNTIME=ON \
        -DSUMMARIZE=ON \
        .. \
        -G Ninja \
    && ninja \
    && sudo ninja install

# TVM Python binding
RUN cd tvm/python \
    && python setup.py install --user