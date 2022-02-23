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
## 3. summary
```
--   ---------------- Summary ----------------
--   CMake version         : 3.16.3
--   CMake executable      : /usr/bin/cmake
--   Generator             : Ninja
--   System                : Linux
--   C++ compiler          : /usr/bin/c++
--   C++ compiler ID       : GNU
--   C++ compiler version  : 9.3.0
--   CXX flags             : -std=c++14 -faligned-new -O2 -Wall -fPIC 
--   Build type            : 
--   Compile definitions   : TVM_INDEX_DEFAULT_I64=1;USE_PROFILER=1;TVM_THREADPOOL_USE_OPENMP=0;DMLC_USE_FOPEN64=0;NDEBUG=1;_GNU_SOURCE;__STDC_CONSTANT_MACROS;__STDC_FORMAT_MACROS;__STDC_LIMIT_MACROS;TVM_LLVM_VERSION=140;USE_DNNL=1;TF_TVMDSOOP_ENABLE_GPU;PT_TVMDSOOP_ENABLE_GPU;TVM_GRAPH_EXECUTOR_TENSORRT
--   Options:
--    BUILD_STATIC_RUNTIME               : ON
--    COMPILER_RT_PATH                   : 3rdparty/compiler-rt
--    DLPACK_PATH                        : 3rdparty/dlpack/include
--    DMLC_PATH                          : 3rdparty/dmlc-core/include
--    HIDE_PRIVATE_SYMBOLS               : OFF
--    INDEX_DEFAULT_I64                  : ON
--    INSTALL_DEV                        : OFF
--    PICOJSON_PATH                      : 3rdparty/picojson
--    RANG_PATH                          : 3rdparty/rang/include
--    ROCM_PATH                          : /opt/rocm
--    SUMMARIZE                          : ON
--    USE_ARM_COMPUTE_LIB                : OFF
--    USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR : OFF
--    USE_BLAS                           : none
--    USE_BNNS                           : OFF
--    USE_BYODT_POSIT                    : OFF
--    USE_CMSISNN                        : OFF
--    USE_COREML                         : OFF
--    USE_CPP_RPC                        : ON
--    USE_CUBLAS                         : OFF
--    USE_CUDA                           : ON
--    USE_CUDNN                          : ON
--    USE_CUTLASS                        : OFF
--    USE_DNNL_CODEGEN                   : OFF
--    USE_ETHOSN                         : OFF
--    USE_FALLBACK_STL_MAP               : OFF
--    USE_GRAPH_EXECUTOR                 : ON
--    USE_GRAPH_EXECUTOR_CUDA_GRAPH      : ON
--    USE_GTEST                          : AUTO
--    USE_HEXAGON_DEVICE                 : OFF
--    USE_HEXAGON_RPC                    : OFF
--    USE_HEXAGON_SDK                    : /path/to/sdk
--    USE_IOS_RPC                        : OFF
--    USE_LIBBACKTRACE                   : ON
--    USE_LLVM                           : ON
--    USE_METAL                          : OFF
--    USE_MICRO                          : OFF
--    USE_MICRO_STANDALONE_RUNTIME       : OFF
--    USE_MIOPEN                         : OFF
--    USE_MKL                            : OFF
--    USE_MKLDNN                         : ON
--    USE_MSVC_MT                        : OFF
--    USE_NNPACK                         : ON
--    USE_OPENCL                         : OFF
--    USE_OPENMP                         : ON
--    USE_PAPI                           : OFF
--    USE_PROFILER                       : ON
--    USE_PT_TVMDSOOP                    : ON
--    USE_RANDOM                         : ON
--    USE_RELAY_DEBUG                    : OFF
--    USE_ROCBLAS                        : OFF
--    USE_ROCM                           : OFF
--    USE_RPC                            : ON
--    USE_RTTI                           : ON
--    USE_RUST_EXT                       : OFF
--    USE_SORT                           : ON
--    USE_STACKVM_RUNTIME                : OFF
--    USE_TARGET_ONNX                    : ON
--    USE_TENSORFLOW_PATH                : none
--    USE_TENSORRT_CODEGEN               : ON
--    USE_TENSORRT_RUNTIME               : ON
--    USE_TFLITE                         : OFF
--    USE_TF_TVMDSOOP                    : ON
--    USE_THREADS                        : ON
--    USE_THRUST                         : OFF
--    USE_VITIS_AI                       : OFF
--    USE_VULKAN                         : OFF
```
