# tvm-build
1. https://github.com/apache/tvm
2. https://tvm.apache.org/docs/tutorial/install.html
## 1. Pull
```bash
docker pull pinto0309/ubuntu2004-cuda114-cudnn8-tensorrt823-tvm:09dev0
```
## 2. Build
```bash
docker build -t pinto0309/ubuntu2004-cuda114-cudnn8-tensorrt823-tvm:09dev0 .
```
## 3. Run
```bash
docker run --rm -it --gpus all \
-v `pwd`:/home/user/workdir \
pinto0309/ubuntu2004-cuda114-cudnn8-tensorrt823-tvm:09dev0
```
## 4. TVM Summary

<details><summary>Build summary</summary><div>

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

</div></details>

## 5. Environment

<details><summary>List of installed tools</summary><div>

```
TVM v0.9.dev0
Python 3.8+
TensorFlow v2.8.0+
PyTorch v1.10.0+
TorchVision
TorchAudio
OpenVINO 2021.4.582+
TensorRT 8.2+
trtexec
pycuda 2021.1
tensorflowjs
coremltools
paddle2onnx
onnx
onnxruntime
onnxruntime-extensions
onnx_graphsurgeon
onnx-simplifier
onnxconverter-common
onnxmltools
onnx-tensorrt
onnx2json
json2onnx
tf2onnx
torch2trt
onnx-tf
tensorflow-datasets
tf_slim
edgetpu_compiler
tflite2tensorflow
openvino2tensorflow
gdown
pandas
matplotlib
paddlepaddle
paddle2onnx
pycocotools
scipy
Intel-Media-SDK
Intel iHD GPU (iGPU) support
OpenCL
```

</div></details>

## 6. Tutorial
### 6-1. tvmc
https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html#compiling-and-optimizing-a-model-with-tvmc
```bash
$ python -m tvm.driver.tvmc
usage: tvmc [-v] [--version] [-h] {run,tune,compile} ...

TVM compiler driver

optional arguments:
  -v, --verbose       increase verbosity
  --version           print the version and exit
  -h, --help          show this help message and exit.

commands:
  {run,tune,compile}
    run               run a compiled module
    tune              auto-tune a model
    compile           compile a model.

TVMC - TVM driver command-line interface
```
```bash
$ wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

$ python -m tvm.driver.tvmc compile \
--target "llvm" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx

One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
```
![image](https://user-images.githubusercontent.com/33194443/155353724-b440fc7c-60e7-41fd-b684-5ef33021c556.png)
```bash
$ mkdir model
$ tar -xvf resnet50-v2-7-tvm.tar -C model
$ ls -l model

total 100496
-rw-r--r-- 1 user user     89142 Feb 23 15:41 mod.json
-rw-r--r-- 1 user user 102125470 Feb 23 15:41 mod.params
-rwxr-xr-x 1 user user    685072 Feb 23 15:41 mod.so
```
- **`mod.so`** is the model, represented as a C++ library, that can be loaded by the TVM runtime.
- **`mod.json`** is a text representation of the TVM Relay computation graph.
- **`mod.params`** is a file containing the parameters for the pre-trained model.

![image](https://user-images.githubusercontent.com/33194443/155355104-34b6f069-c136-4492-ac2d-6d1cba4d2b47.png)
- preprocess.py
  ```python
  from tvm.contrib.download import download_testdata
  from PIL import Image
  import numpy as np

  img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
  img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

  # Resize it to 224x224
  resized_image = Image.open(img_path).resize((224, 224))
  img_data = np.asarray(resized_image).astype("float32")

  # ONNX expects NCHW input, so convert the array
  img_data = np.transpose(img_data, (2, 0, 1))

  # Normalize according to ImageNet
  imagenet_mean = np.array([0.485, 0.456, 0.406])
  imagenet_stddev = np.array([0.229, 0.224, 0.225])
  norm_img_data = np.zeros(img_data.shape).astype("float32")
  for i in range(img_data.shape[0]):
      norm_img_data[i,:,:] = (img_data[i,:,:] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

  # Add batch dimension
  img_data = np.expand_dims(norm_img_data, axis=0)

  # Save to .npz (outputs imagenet_cat.npz)
  np.savez("imagenet_cat", data=img_data)
  ```
- kitten.jpg  
  ![kitten](https://user-images.githubusercontent.com/33194443/155369334-7f836eeb-433d-4dbd-b433-2df1a4755edb.jpg)
```bash
$ python preprocess.py
```
![image](https://user-images.githubusercontent.com/33194443/155358586-854d785e-e2a6-41b2-812c-fe6e626187a6.png)
```bash
$ python -m tvm.driver.tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar
```
![image](https://user-images.githubusercontent.com/33194443/155358960-3dddde72-4fcb-4466-960b-78720bacdb3f.png)
- postprocess.py
  ```python
  import os.path
  import numpy as np

  from scipy.special import softmax
  from tvm.contrib.download import download_testdata

  # Download a list of labels
  labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
  labels_path = download_testdata(labels_url, "synset.txt", module="data")

  with open(labels_path, "r") as f:
      labels = [l.rstrip() for l in f]

  output_file = "predictions.npz"

  # Open the output and read the output tensor
  if os.path.exists(output_file):
      with np.load(output_file) as data:
          scores = softmax(data["output_0"])
          scores = np.squeeze(scores)
          ranks = np.argsort(scores)[::-1]

          for rank in ranks[0:5]:
              print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
  ```
```
$ python postprocess.py

class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```
```bash
$ python -m tvm.driver.tvmc tune --help
```

<details><summary>parameters</summary><div>

```
usage: tvmc tune [-h]
--target TARGET
-o OUTPUT
[--early-stopping EARLY_STOPPING]
[--min-repeat-ms MIN_REPEAT_MS]
[--model-format {keras,onnx,pb,tflite,pytorch,paddle}]
[--number NUMBER]
[--parallel PARALLEL]
[--repeat REPEAT]
[--rpc-key RPC_KEY]
[--rpc-tracker RPC_TRACKER]
[--target-example_target_hook-from_device TARGET_EXAMPLE_TARGET_HOOK_FROM_DEVICE]
[--target-example_target_hook-libs TARGET_EXAMPLE_TARGET_HOOK_LIBS]
[--target-example_target_hook-model TARGET_EXAMPLE_TARGET_HOOK_MODEL]
[--target-example_target_hook-tag TARGET_EXAMPLE_TARGET_HOOK_TAG]
[--target-example_target_hook-device TARGET_EXAMPLE_TARGET_HOOK_DEVICE]
[--target-example_target_hook-keys TARGET_EXAMPLE_TARGET_HOOK_KEYS]
[--target-ext_dev-from_device TARGET_EXT_DEV_FROM_DEVICE]
[--target-ext_dev-libs TARGET_EXT_DEV_LIBS]
[--target-ext_dev-model TARGET_EXT_DEV_MODEL]
[--target-ext_dev-system-lib TARGET_EXT_DEV_SYSTEM_LIB]
[--target-ext_dev-tag TARGET_EXT_DEV_TAG]
[--target-ext_dev-device TARGET_EXT_DEV_DEVICE]
[--target-ext_dev-keys TARGET_EXT_DEV_KEYS]
[--target-llvm-fast-math TARGET_LLVM_FAST_MATH]
[--target-llvm-opt-level TARGET_LLVM_OPT_LEVEL]
[--target-llvm-unpacked-api TARGET_LLVM_UNPACKED_API]
[--target-llvm-from_device TARGET_LLVM_FROM_DEVICE]
[--target-llvm-fast-math-ninf TARGET_LLVM_FAST_MATH_NINF]
[--target-llvm-mattr TARGET_LLVM_MATTR]
[--target-llvm-num-cores TARGET_LLVM_NUM_CORES]
[--target-llvm-libs TARGET_LLVM_LIBS]
[--target-llvm-fast-math-nsz TARGET_LLVM_FAST_MATH_NSZ]
[--target-llvm-link-params TARGET_LLVM_LINK_PARAMS]
[--target-llvm-interface-api TARGET_LLVM_INTERFACE_API]
[--target-llvm-fast-math-contract TARGET_LLVM_FAST_MATH_CONTRACT]
[--target-llvm-system-lib TARGET_LLVM_SYSTEM_LIB]
[--target-llvm-tag TARGET_LLVM_TAG]
[--target-llvm-mtriple TARGET_LLVM_MTRIPLE]
[--target-llvm-model TARGET_LLVM_MODEL]
[--target-llvm-mfloat-abi TARGET_LLVM_MFLOAT_ABI]
[--target-llvm-mcpu TARGET_LLVM_MCPU]
[--target-llvm-device TARGET_LLVM_DEVICE]
[--target-llvm-runtime TARGET_LLVM_RUNTIME]
[--target-llvm-fast-math-arcp TARGET_LLVM_FAST_MATH_ARCP]
[--target-llvm-fast-math-reassoc TARGET_LLVM_FAST_MATH_REASSOC]
[--target-llvm-mabi TARGET_LLVM_MABI]
[--target-llvm-keys TARGET_LLVM_KEYS]
[--target-llvm-fast-math-nnan TARGET_LLVM_FAST_MATH_NNAN]
[--target-hybrid-from_device TARGET_HYBRID_FROM_DEVICE]
[--target-hybrid-libs TARGET_HYBRID_LIBS]
[--target-hybrid-model TARGET_HYBRID_MODEL]
[--target-hybrid-system-lib TARGET_HYBRID_SYSTEM_LIB]
[--target-hybrid-tag TARGET_HYBRID_TAG]
[--target-hybrid-device TARGET_HYBRID_DEVICE]
[--target-hybrid-keys TARGET_HYBRID_KEYS]
[--target-aocl-from_device TARGET_AOCL_FROM_DEVICE]
[--target-aocl-libs TARGET_AOCL_LIBS]
[--target-aocl-model TARGET_AOCL_MODEL]
[--target-aocl-system-lib TARGET_AOCL_SYSTEM_LIB]
[--target-aocl-tag TARGET_AOCL_TAG]
[--target-aocl-device TARGET_AOCL_DEVICE]
[--target-aocl-keys TARGET_AOCL_KEYS]
[--target-nvptx-max_num_threads TARGET_NVPTX_MAX_NUM_THREADS]
[--target-nvptx-thread_warp_size TARGET_NVPTX_THREAD_WARP_SIZE]
[--target-nvptx-from_device TARGET_NVPTX_FROM_DEVICE]
[--target-nvptx-libs TARGET_NVPTX_LIBS]
[--target-nvptx-model TARGET_NVPTX_MODEL]
[--target-nvptx-system-lib TARGET_NVPTX_SYSTEM_LIB]
[--target-nvptx-mtriple TARGET_NVPTX_MTRIPLE]
[--target-nvptx-tag TARGET_NVPTX_TAG]
[--target-nvptx-mcpu TARGET_NVPTX_MCPU]
[--target-nvptx-device TARGET_NVPTX_DEVICE]
[--target-nvptx-keys TARGET_NVPTX_KEYS]
[--target-opencl-max_num_threads TARGET_OPENCL_MAX_NUM_THREADS]
[--target-opencl-thread_warp_size TARGET_OPENCL_THREAD_WARP_SIZE]
[--target-opencl-from_device TARGET_OPENCL_FROM_DEVICE]
[--target-opencl-libs TARGET_OPENCL_LIBS]
[--target-opencl-model TARGET_OPENCL_MODEL]
[--target-opencl-system-lib TARGET_OPENCL_SYSTEM_LIB]
[--target-opencl-tag TARGET_OPENCL_TAG]
[--target-opencl-device TARGET_OPENCL_DEVICE]
[--target-opencl-keys TARGET_OPENCL_KEYS]
[--target-metal-max_num_threads TARGET_METAL_MAX_NUM_THREADS]
[--target-metal-thread_warp_size TARGET_METAL_THREAD_WARP_SIZE]
[--target-metal-from_device TARGET_METAL_FROM_DEVICE]
[--target-metal-libs TARGET_METAL_LIBS]
[--target-metal-keys TARGET_METAL_KEYS]
[--target-metal-model TARGET_METAL_MODEL]
[--target-metal-system-lib TARGET_METAL_SYSTEM_LIB]
[--target-metal-tag TARGET_METAL_TAG]
[--target-metal-device TARGET_METAL_DEVICE]
[--target-metal-max_function_args TARGET_METAL_MAX_FUNCTION_ARGS]
[--target-webgpu-max_num_threads TARGET_WEBGPU_MAX_NUM_THREADS]
[--target-webgpu-from_device TARGET_WEBGPU_FROM_DEVICE]
[--target-webgpu-libs TARGET_WEBGPU_LIBS]
[--target-webgpu-model TARGET_WEBGPU_MODEL]
[--target-webgpu-system-lib TARGET_WEBGPU_SYSTEM_LIB]
[--target-webgpu-tag TARGET_WEBGPU_TAG]
[--target-webgpu-device TARGET_WEBGPU_DEVICE]
[--target-webgpu-keys TARGET_WEBGPU_KEYS]
[--target-rocm-max_num_threads TARGET_ROCM_MAX_NUM_THREADS]
[--target-rocm-thread_warp_size TARGET_ROCM_THREAD_WARP_SIZE]
[--target-rocm-from_device TARGET_ROCM_FROM_DEVICE]
[--target-rocm-libs TARGET_ROCM_LIBS]
[--target-rocm-model TARGET_ROCM_MODEL]
[--target-rocm-system-lib TARGET_ROCM_SYSTEM_LIB]
[--target-rocm-mtriple TARGET_ROCM_MTRIPLE]
[--target-rocm-tag TARGET_ROCM_TAG]
[--target-rocm-mcpu TARGET_ROCM_MCPU]
[--target-rocm-device TARGET_ROCM_DEVICE]
[--target-rocm-keys TARGET_ROCM_KEYS]
[--target-vulkan-max_num_threads TARGET_VULKAN_MAX_NUM_THREADS]
[--target-vulkan-thread_warp_size TARGET_VULKAN_THREAD_WARP_SIZE]
[--target-vulkan-max_block_size_z TARGET_VULKAN_MAX_BLOCK_SIZE_Z]
[--target-vulkan-max_per_stage_descriptor_storage_buffer TARGET_VULKAN_MAX_PER_STAGE_DESCRIPTOR_STORAGE_BUFFER]
[--target-vulkan-driver_version TARGET_VULKAN_DRIVER_VERSION]
[--target-vulkan-from_device TARGET_VULKAN_FROM_DEVICE]
[--target-vulkan-supports_16bit_buffer TARGET_VULKAN_SUPPORTS_16BIT_BUFFER]
[--target-vulkan-libs TARGET_VULKAN_LIBS]
[--target-vulkan-supported_subgroup_operations TARGET_VULKAN_SUPPORTED_SUBGROUP_OPERATIONS]
[--target-vulkan-supports_dedicated_allocation TARGET_VULKAN_SUPPORTS_DEDICATED_ALLOCATION]
[--target-vulkan-max_storage_buffer_range TARGET_VULKAN_MAX_STORAGE_BUFFER_RANGE]
[--target-vulkan-max_push_constants_size TARGET_VULKAN_MAX_PUSH_CONSTANTS_SIZE]
[--target-vulkan-supports_push_descriptor TARGET_VULKAN_SUPPORTS_PUSH_DESCRIPTOR]
[--target-vulkan-supports_int64 TARGET_VULKAN_SUPPORTS_INT64]
[--target-vulkan-supports_float32 TARGET_VULKAN_SUPPORTS_FLOAT32]
[--target-vulkan-model TARGET_VULKAN_MODEL]
[--target-vulkan-max_block_size_x TARGET_VULKAN_MAX_BLOCK_SIZE_X]
[--target-vulkan-system-lib TARGET_VULKAN_SYSTEM_LIB]
[--target-vulkan-max_block_size_y TARGET_VULKAN_MAX_BLOCK_SIZE_Y]
[--target-vulkan-tag TARGET_VULKAN_TAG]
[--target-vulkan-supports_int8 TARGET_VULKAN_SUPPORTS_INT8]
[--target-vulkan-max_spirv_version TARGET_VULKAN_MAX_SPIRV_VERSION]
[--target-vulkan-vulkan_api_version TARGET_VULKAN_VULKAN_API_VERSION]
[--target-vulkan-supports_8bit_buffer TARGET_VULKAN_SUPPORTS_8BIT_BUFFER]
[--target-vulkan-device_type TARGET_VULKAN_DEVICE_TYPE]
[--target-vulkan-supports_int32 TARGET_VULKAN_SUPPORTS_INT32]
[--target-vulkan-device TARGET_VULKAN_DEVICE]
[--target-vulkan-driver_name TARGET_VULKAN_DRIVER_NAME]
[--target-vulkan-supports_float16 TARGET_VULKAN_SUPPORTS_FLOAT16]
[--target-vulkan-supports_storage_buffer_storage_class TARGET_VULKAN_SUPPORTS_STORAGE_BUFFER_STORAGE_CLASS]
[--target-vulkan-supports_float64 TARGET_VULKAN_SUPPORTS_FLOAT64]
[--target-vulkan-max_uniform_buffer_range TARGET_VULKAN_MAX_UNIFORM_BUFFER_RANGE]
[--target-vulkan-device_name TARGET_VULKAN_DEVICE_NAME]
[--target-vulkan-keys TARGET_VULKAN_KEYS]
[--target-vulkan-max_shared_memory_per_block TARGET_VULKAN_MAX_SHARED_MEMORY_PER_BLOCK]
[--target-vulkan-supports_int16 TARGET_VULKAN_SUPPORTS_INT16]
[--target-cuda-max_num_threads TARGET_CUDA_MAX_NUM_THREADS]
[--target-cuda-thread_warp_size TARGET_CUDA_THREAD_WARP_SIZE]
[--target-cuda-from_device TARGET_CUDA_FROM_DEVICE]
[--target-cuda-arch TARGET_CUDA_ARCH]
[--target-cuda-libs TARGET_CUDA_LIBS]
[--target-cuda-shared_memory_per_block TARGET_CUDA_SHARED_MEMORY_PER_BLOCK]
[--target-cuda-model TARGET_CUDA_MODEL]
[--target-cuda-system-lib TARGET_CUDA_SYSTEM_LIB]
[--target-cuda-tag TARGET_CUDA_TAG]
[--target-cuda-device TARGET_CUDA_DEVICE]
[--target-cuda-mcpu TARGET_CUDA_MCPU]
[--target-cuda-max_threads_per_block TARGET_CUDA_MAX_THREADS_PER_BLOCK]
[--target-cuda-registers_per_block TARGET_CUDA_REGISTERS_PER_BLOCK]
[--target-cuda-keys TARGET_CUDA_KEYS]
[--target-sdaccel-from_device TARGET_SDACCEL_FROM_DEVICE]
[--target-sdaccel-libs TARGET_SDACCEL_LIBS]
[--target-sdaccel-model TARGET_SDACCEL_MODEL]
[--target-sdaccel-system-lib TARGET_SDACCEL_SYSTEM_LIB]
[--target-sdaccel-tag TARGET_SDACCEL_TAG]
[--target-sdaccel-device TARGET_SDACCEL_DEVICE]
[--target-sdaccel-keys TARGET_SDACCEL_KEYS]
[--target-composite-from_device TARGET_COMPOSITE_FROM_DEVICE]
[--target-composite-libs TARGET_COMPOSITE_LIBS]
[--target-composite-devices TARGET_COMPOSITE_DEVICES]
[--target-composite-model TARGET_COMPOSITE_MODEL]
[--target-composite-tag TARGET_COMPOSITE_TAG]
[--target-composite-device TARGET_COMPOSITE_DEVICE]
[--target-composite-keys TARGET_COMPOSITE_KEYS]
[--target-stackvm-from_device TARGET_STACKVM_FROM_DEVICE]
[--target-stackvm-libs TARGET_STACKVM_LIBS]
[--target-stackvm-model TARGET_STACKVM_MODEL]
[--target-stackvm-system-lib TARGET_STACKVM_SYSTEM_LIB]
[--target-stackvm-tag TARGET_STACKVM_TAG]
[--target-stackvm-device TARGET_STACKVM_DEVICE]
[--target-stackvm-keys TARGET_STACKVM_KEYS]
[--target-aocl_sw_emu-from_device TARGET_AOCL_SW_EMU_FROM_DEVICE]
[--target-aocl_sw_emu-libs TARGET_AOCL_SW_EMU_LIBS]
[--target-aocl_sw_emu-model TARGET_AOCL_SW_EMU_MODEL]
[--target-aocl_sw_emu-system-lib TARGET_AOCL_SW_EMU_SYSTEM_LIB]
[--target-aocl_sw_emu-tag TARGET_AOCL_SW_EMU_TAG]
[--target-aocl_sw_emu-device TARGET_AOCL_SW_EMU_DEVICE]
[--target-aocl_sw_emu-keys TARGET_AOCL_SW_EMU_KEYS]
[--target-c-unpacked-api TARGET_C_UNPACKED_API]
[--target-c-from_device TARGET_C_FROM_DEVICE]
[--target-c-libs TARGET_C_LIBS]
[--target-c-constants-byte-alignment TARGET_C_CONSTANTS_BYTE_ALIGNMENT]
[--target-c-executor TARGET_C_EXECUTOR]
[--target-c-link-params TARGET_C_LINK_PARAMS]
[--target-c-model TARGET_C_MODEL]
[--target-c-workspace-byte-alignment TARGET_C_WORKSPACE_BYTE_ALIGNMENT]
[--target-c-system-lib TARGET_C_SYSTEM_LIB]
[--target-c-tag TARGET_C_TAG]
[--target-c-interface-api TARGET_C_INTERFACE_API]
[--target-c-mcpu TARGET_C_MCPU]
[--target-c-device TARGET_C_DEVICE]
[--target-c-runtime TARGET_C_RUNTIME]
[--target-c-keys TARGET_C_KEYS]
[--target-c-march TARGET_C_MARCH]
[--target-hexagon-from_device TARGET_HEXAGON_FROM_DEVICE]
[--target-hexagon-libs TARGET_HEXAGON_LIBS]
[--target-hexagon-mattr TARGET_HEXAGON_MATTR]
[--target-hexagon-model TARGET_HEXAGON_MODEL]
[--target-hexagon-llvm-options TARGET_HEXAGON_LLVM_OPTIONS]
[--target-hexagon-mtriple TARGET_HEXAGON_MTRIPLE]
[--target-hexagon-system-lib TARGET_HEXAGON_SYSTEM_LIB]
[--target-hexagon-mcpu TARGET_HEXAGON_MCPU]
[--target-hexagon-device TARGET_HEXAGON_DEVICE]
[--target-hexagon-tag TARGET_HEXAGON_TAG]
[--target-hexagon-link-params TARGET_HEXAGON_LINK_PARAMS]
[--target-hexagon-keys TARGET_HEXAGON_KEYS]
[--target-host TARGET_HOST]
[--timeout TIMEOUT]
[--trials TRIALS]
[--tuning-records PATH]
[--desired-layout {NCHW,NHWC}]
[--enable-autoscheduler]
[--cache-line-bytes CACHE_LINE_BYTES]
[--num-cores NUM_CORES]
[--vector-unit-bytes VECTOR_UNIT_BYTES]
[--max-shared-memory-per-block MAX_SHARED_MEMORY_PER_BLOCK]
[--max-local-memory-per-block MAX_LOCAL_MEMORY_PER_BLOCK]
[--max-threads-per-block MAX_THREADS_PER_BLOCK]
[--max-vthread-extent MAX_VTHREAD_EXTENT]
[--warp-size WARP_SIZE]
[--include-simple-tasks]
[--log-estimated-latency]
[--tuner {ga,gridsearch,random,xgb,xgb_knob,xgb-rank}]
[--input-shapes INPUT_SHAPES]
FILE

positional arguments:
    FILE
        path to the input model file

optional arguments:
    -h, --help
        show this help message and exit
    --early-stopping EARLY_STOPPING
        minimum number of trials before early stopping
    --min-repeat-ms MIN_REPEAT_MS
        minimum time to run each trial, in milliseconds.
        Defaults to 0 on x86 and 1000 on all other targets
    --model-format {keras,onnx,pb,tflite,pytorch,paddle}
        specify input model format
    --number NUMBER
        number of runs a single repeat is made of.
        The final number of tuning executions is: (1 + number * repeat)
    -o OUTPUT, --output OUTPUT
        output file to store the tuning records for the tuning process
    --parallel PARALLEL
        the maximum number of parallel devices to use when tuning
    --repeat REPEAT
        how many times to repeat each measurement
    --rpc-key RPC_KEY
        the RPC tracker key of the target device.
        Required when --rpc-tracker is provided.
    --rpc-tracker RPC_TRACKER
        hostname (required) and port (optional, defaults to 9090) of the RPC tracker,
        e.g. '192.168.0.100:9999'
    --target TARGET
        compilation target as plain string, inline JSON or path to a JSON file
    --target-host TARGET_HOST
        the host compilation target, defaults to 'llvm'
    --timeout TIMEOUT
        compilation timeout, in seconds
    --trials TRIALS
        the maximum number of tuning trials to perform
    --tuning-records PATH
        path to an auto-tuning log file by AutoTVM.
    --desired-layout {NCHW,NHWC}
        change the data layout of the whole graph
    --enable-autoscheduler
        enable tuning the graph through the autoscheduler
    --input-shapes INPUT_SHAPES
        specify non-generic shapes for model to run,
        format is "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"

    target example_target_hook:
    --target-example_target_hook-from_device TARGET_EXAMPLE_TARGET_HOOK_FROM_DEVICE
        target example_target_hook from_device
    --target-example_target_hook-libs TARGET_EXAMPLE_TARGET_HOOK_LIBS
        target example_target_hook libs options
    --target-example_target_hook-model TARGET_EXAMPLE_TARGET_HOOK_MODEL
        target example_target_hook model string
    --target-example_target_hook-tag TARGET_EXAMPLE_TARGET_HOOK_TAG
        target example_target_hook tag string
    --target-example_target_hook-device TARGET_EXAMPLE_TARGET_HOOK_DEVICE
        target example_target_hook device string
    --target-example_target_hook-keys TARGET_EXAMPLE_TARGET_HOOK_KEYS
        target example_target_hook keys options

    target ext_dev:
    --target-ext_dev-from_device TARGET_EXT_DEV_FROM_DEVICE
        target ext_dev from_device
    --target-ext_dev-libs TARGET_EXT_DEV_LIBS
        target ext_dev libs options
    --target-ext_dev-model TARGET_EXT_DEV_MODEL
        target ext_dev model string
    --target-ext_dev-system-lib TARGET_EXT_DEV_SYSTEM_LIB
        target ext_dev system-lib
    --target-ext_dev-tag TARGET_EXT_DEV_TAG
        target ext_dev tag string
    --target-ext_dev-device TARGET_EXT_DEV_DEVICE
        target ext_dev device string
    --target-ext_dev-keys TARGET_EXT_DEV_KEYS
        target ext_dev keys options

    target llvm:
    --target-llvm-fast-math TARGET_LLVM_FAST_MATH
        target llvm fast-math
    --target-llvm-opt-level TARGET_LLVM_OPT_LEVEL
        target llvm opt-level
    --target-llvm-unpacked-api TARGET_LLVM_UNPACKED_API
        target llvm unpacked-api
    --target-llvm-from_device TARGET_LLVM_FROM_DEVICE
        target llvm from_device
    --target-llvm-fast-math-ninf TARGET_LLVM_FAST_MATH_NINF
        target llvm fast-math-ninf
    --target-llvm-mattr TARGET_LLVM_MATTR
        target llvm mattr options
    --target-llvm-num-cores TARGET_LLVM_NUM_CORES
        target llvm num-cores
    --target-llvm-libs TARGET_LLVM_LIBS
        target llvm libs options
    --target-llvm-fast-math-nsz TARGET_LLVM_FAST_MATH_NSZ
        target llvm fast-math-nsz
    --target-llvm-link-params TARGET_LLVM_LINK_PARAMS
        target llvm link-params
    --target-llvm-interface-api TARGET_LLVM_INTERFACE_API
        target llvm interface-api string
    --target-llvm-fast-math-contract TARGET_LLVM_FAST_MATH_CONTRACT
        target llvm fast-math-contract
    --target-llvm-system-lib TARGET_LLVM_SYSTEM_LIB
        target llvm system-lib
    --target-llvm-tag TARGET_LLVM_TAG
        target llvm tag string
    --target-llvm-mtriple TARGET_LLVM_MTRIPLE
        target llvm mtriple string
    --target-llvm-model TARGET_LLVM_MODEL
        target llvm model string
    --target-llvm-mfloat-abi TARGET_LLVM_MFLOAT_ABI
        target llvm mfloat-abi string
    --target-llvm-mcpu TARGET_LLVM_MCPU
        target llvm mcpu string
    --target-llvm-device TARGET_LLVM_DEVICE
        target llvm device string
    --target-llvm-runtime TARGET_LLVM_RUNTIME
        target llvm runtime string
    --target-llvm-fast-math-arcp TARGET_LLVM_FAST_MATH_ARCP
        target llvm fast-math-arcp
    --target-llvm-fast-math-reassoc TARGET_LLVM_FAST_MATH_REASSOC
        target llvm fast-math-reassoc
    --target-llvm-mabi TARGET_LLVM_MABI
        target llvm mabi string
    --target-llvm-keys TARGET_LLVM_KEYS
        target llvm keys options
    --target-llvm-fast-math-nnan TARGET_LLVM_FAST_MATH_NNAN
        target llvm fast-math-nnan

    target hybrid:
    --target-hybrid-from_device TARGET_HYBRID_FROM_DEVICE
        target hybrid from_device
    --target-hybrid-libs TARGET_HYBRID_LIBS
        target hybrid libs options
    --target-hybrid-model TARGET_HYBRID_MODEL
        target hybrid model string
    --target-hybrid-system-lib TARGET_HYBRID_SYSTEM_LIB
        target hybrid system-lib
    --target-hybrid-tag TARGET_HYBRID_TAG
        target hybrid tag string
    --target-hybrid-device TARGET_HYBRID_DEVICE
        target hybrid device string
    --target-hybrid-keys TARGET_HYBRID_KEYS
        target hybrid keys options

    target aocl:
    --target-aocl-from_device TARGET_AOCL_FROM_DEVICE
        target aocl from_device
    --target-aocl-libs TARGET_AOCL_LIBS
        target aocl libs options
    --target-aocl-model TARGET_AOCL_MODEL
        target aocl model string
    --target-aocl-system-lib TARGET_AOCL_SYSTEM_LIB
        target aocl system-lib
    --target-aocl-tag TARGET_AOCL_TAG
        target aocl tag string
    --target-aocl-device TARGET_AOCL_DEVICE
        target aocl device string
    --target-aocl-keys TARGET_AOCL_KEYS
        target aocl keys options

    target nvptx:
    --target-nvptx-max_num_threads TARGET_NVPTX_MAX_NUM_THREADS
        target nvptx max_num_threads
    --target-nvptx-thread_warp_size TARGET_NVPTX_THREAD_WARP_SIZE
        target nvptx thread_warp_size
    --target-nvptx-from_device TARGET_NVPTX_FROM_DEVICE
        target nvptx from_device
    --target-nvptx-libs TARGET_NVPTX_LIBS
        target nvptx libs options
    --target-nvptx-model TARGET_NVPTX_MODEL
        target nvptx model string
    --target-nvptx-system-lib TARGET_NVPTX_SYSTEM_LIB
        target nvptx system-lib
    --target-nvptx-mtriple TARGET_NVPTX_MTRIPLE
        target nvptx mtriple string
    --target-nvptx-tag TARGET_NVPTX_TAG
        target nvptx tag string
    --target-nvptx-mcpu TARGET_NVPTX_MCPU
        target nvptx mcpu string
    --target-nvptx-device TARGET_NVPTX_DEVICE
        target nvptx device string
    --target-nvptx-keys TARGET_NVPTX_KEYS
        target nvptx keys options

    target opencl:
    --target-opencl-max_num_threads TARGET_OPENCL_MAX_NUM_THREADS
        target opencl max_num_threads
    --target-opencl-thread_warp_size TARGET_OPENCL_THREAD_WARP_SIZE
        target opencl thread_warp_size
    --target-opencl-from_device TARGET_OPENCL_FROM_DEVICE
        target opencl from_device
    --target-opencl-libs TARGET_OPENCL_LIBS
        target opencl libs options
    --target-opencl-model TARGET_OPENCL_MODEL
        target opencl model string
    --target-opencl-system-lib TARGET_OPENCL_SYSTEM_LIB
        target opencl system-lib
    --target-opencl-tag TARGET_OPENCL_TAG
        target opencl tag string
    --target-opencl-device TARGET_OPENCL_DEVICE
        target opencl device string
    --target-opencl-keys TARGET_OPENCL_KEYS
        target opencl keys options

    target metal:
    --target-metal-max_num_threads TARGET_METAL_MAX_NUM_THREADS
        target metal max_num_threads
    --target-metal-thread_warp_size TARGET_METAL_THREAD_WARP_SIZE
        target metal thread_warp_size
    --target-metal-from_device TARGET_METAL_FROM_DEVICE
        target metal from_device
    --target-metal-libs TARGET_METAL_LIBS
        target metal libs options
    --target-metal-keys TARGET_METAL_KEYS
        target metal keys options
    --target-metal-model TARGET_METAL_MODEL
        target metal model string
    --target-metal-system-lib TARGET_METAL_SYSTEM_LIB
        target metal system-lib
    --target-metal-tag TARGET_METAL_TAG
        target metal tag string
    --target-metal-device TARGET_METAL_DEVICE
        target metal device string
    --target-metal-max_function_args TARGET_METAL_MAX_FUNCTION_ARGS
        target metal max_function_args

    target webgpu:
    --target-webgpu-max_num_threads TARGET_WEBGPU_MAX_NUM_THREADS
        target webgpu max_num_threads
    --target-webgpu-from_device TARGET_WEBGPU_FROM_DEVICE
        target webgpu from_device
    --target-webgpu-libs TARGET_WEBGPU_LIBS
        target webgpu libs options
    --target-webgpu-model TARGET_WEBGPU_MODEL
        target webgpu model string
    --target-webgpu-system-lib TARGET_WEBGPU_SYSTEM_LIB
        target webgpu system-lib
    --target-webgpu-tag TARGET_WEBGPU_TAG
        target webgpu tag string
    --target-webgpu-device TARGET_WEBGPU_DEVICE
        target webgpu device string
    --target-webgpu-keys TARGET_WEBGPU_KEYS
        target webgpu keys options

    target rocm:
    --target-rocm-max_num_threads TARGET_ROCM_MAX_NUM_THREADS
        target rocm max_num_threads
    --target-rocm-thread_warp_size TARGET_ROCM_THREAD_WARP_SIZE
        target rocm thread_warp_size
    --target-rocm-from_device TARGET_ROCM_FROM_DEVICE
        target rocm from_device
    --target-rocm-libs TARGET_ROCM_LIBS
        target rocm libs options
    --target-rocm-model TARGET_ROCM_MODEL
        target rocm model string
    --target-rocm-system-lib TARGET_ROCM_SYSTEM_LIB
        target rocm system-lib
    --target-rocm-mtriple TARGET_ROCM_MTRIPLE
        target rocm mtriple string
    --target-rocm-tag TARGET_ROCM_TAG
        target rocm tag string
    --target-rocm-mcpu TARGET_ROCM_MCPU
        target rocm mcpu string
    --target-rocm-device TARGET_ROCM_DEVICE
        target rocm device string
    --target-rocm-keys TARGET_ROCM_KEYS
        target rocm keys options

    target vulkan:
    --target-vulkan-max_num_threads TARGET_VULKAN_MAX_NUM_THREADS
        target vulkan max_num_threads
    --target-vulkan-thread_warp_size TARGET_VULKAN_THREAD_WARP_SIZE
        target vulkan thread_warp_size
    --target-vulkan-max_block_size_z TARGET_VULKAN_MAX_BLOCK_SIZE_Z
        target vulkan max_block_size_z
    --target-vulkan-max_per_stage_descriptor_storage_buffer TARGET_VULKAN_MAX_PER_STAGE_DESCRIPTOR_STORAGE_BUFFER
        target vulkan max_per_stage_descriptor_storage_buffer
    --target-vulkan-driver_version TARGET_VULKAN_DRIVER_VERSION
        target vulkan driver_version
    --target-vulkan-from_device TARGET_VULKAN_FROM_DEVICE
        target vulkan from_device
    --target-vulkan-supports_16bit_buffer TARGET_VULKAN_SUPPORTS_16BIT_BUFFER
        target vulkan supports_16bit_buffer
    --target-vulkan-libs TARGET_VULKAN_LIBS
        target vulkan libs options
    --target-vulkan-supported_subgroup_operations TARGET_VULKAN_SUPPORTED_SUBGROUP_OPERATIONS
        target vulkan supported_subgroup_operations
    --target-vulkan-supports_dedicated_allocation TARGET_VULKAN_SUPPORTS_DEDICATED_ALLOCATION
        target vulkan supports_dedicated_allocation
    --target-vulkan-max_storage_buffer_range TARGET_VULKAN_MAX_STORAGE_BUFFER_RANGE
        target vulkan max_storage_buffer_range
    --target-vulkan-max_push_constants_size TARGET_VULKAN_MAX_PUSH_CONSTANTS_SIZE
        target vulkan max_push_constants_size
    --target-vulkan-supports_push_descriptor TARGET_VULKAN_SUPPORTS_PUSH_DESCRIPTOR
        target vulkan supports_push_descriptor
    --target-vulkan-supports_int64 TARGET_VULKAN_SUPPORTS_INT64
        target vulkan supports_int64
    --target-vulkan-supports_float32 TARGET_VULKAN_SUPPORTS_FLOAT32
        target vulkan supports_float32
    --target-vulkan-model TARGET_VULKAN_MODEL
        target vulkan model string
    --target-vulkan-max_block_size_x TARGET_VULKAN_MAX_BLOCK_SIZE_X
        target vulkan max_block_size_x
    --target-vulkan-system-lib TARGET_VULKAN_SYSTEM_LIB
        target vulkan system-lib
    --target-vulkan-max_block_size_y TARGET_VULKAN_MAX_BLOCK_SIZE_Y
        target vulkan max_block_size_y
    --target-vulkan-tag TARGET_VULKAN_TAG
        target vulkan tag string
    --target-vulkan-supports_int8 TARGET_VULKAN_SUPPORTS_INT8
        target vulkan supports_int8
    --target-vulkan-max_spirv_version TARGET_VULKAN_MAX_SPIRV_VERSION
        target vulkan max_spirv_version
    --target-vulkan-vulkan_api_version TARGET_VULKAN_VULKAN_API_VERSION
        target vulkan vulkan_api_version
    --target-vulkan-supports_8bit_buffer TARGET_VULKAN_SUPPORTS_8BIT_BUFFER
        target vulkan supports_8bit_buffer
    --target-vulkan-device_type TARGET_VULKAN_DEVICE_TYPE
        target vulkan device_type string
    --target-vulkan-supports_int32 TARGET_VULKAN_SUPPORTS_INT32
        target vulkan supports_int32
    --target-vulkan-device TARGET_VULKAN_DEVICE
        target vulkan device string
    --target-vulkan-driver_name TARGET_VULKAN_DRIVER_NAME
        target vulkan driver_name string
    --target-vulkan-supports_float16 TARGET_VULKAN_SUPPORTS_FLOAT16
        target vulkan supports_float16
    --target-vulkan-supports_storage_buffer_storage_class TARGET_VULKAN_SUPPORTS_STORAGE_BUFFER_STORAGE_CLASS
        target vulkan supports_storage_buffer_storage_class
    --target-vulkan-supports_float64 TARGET_VULKAN_SUPPORTS_FLOAT64
        target vulkan supports_float64
    --target-vulkan-max_uniform_buffer_range TARGET_VULKAN_MAX_UNIFORM_BUFFER_RANGE
        target vulkan max_uniform_buffer_range
    --target-vulkan-device_name TARGET_VULKAN_DEVICE_NAME
        target vulkan device_name string
    --target-vulkan-keys TARGET_VULKAN_KEYS
        target vulkan keys options
    --target-vulkan-max_shared_memory_per_block TARGET_VULKAN_MAX_SHARED_MEMORY_PER_BLOCK
        target vulkan max_shared_memory_per_block
    --target-vulkan-supports_int16 TARGET_VULKAN_SUPPORTS_INT16
        target vulkan supports_int16

    target cuda:
    --target-cuda-max_num_threads TARGET_CUDA_MAX_NUM_THREADS
        target cuda max_num_threads
    --target-cuda-thread_warp_size TARGET_CUDA_THREAD_WARP_SIZE
        target cuda thread_warp_size
    --target-cuda-from_device TARGET_CUDA_FROM_DEVICE
        target cuda from_device
    --target-cuda-arch TARGET_CUDA_ARCH
        target cuda arch string
    --target-cuda-libs TARGET_CUDA_LIBS
        target cuda libs options
    --target-cuda-shared_memory_per_block TARGET_CUDA_SHARED_MEMORY_PER_BLOCK
        target cuda shared_memory_per_block
    --target-cuda-model TARGET_CUDA_MODEL
        target cuda model string
    --target-cuda-system-lib TARGET_CUDA_SYSTEM_LIB
        target cuda system-lib
    --target-cuda-tag TARGET_CUDA_TAG
        target cuda tag string
    --target-cuda-device TARGET_CUDA_DEVICE
        target cuda device string
    --target-cuda-mcpu TARGET_CUDA_MCPU
        target cuda mcpu string
    --target-cuda-max_threads_per_block TARGET_CUDA_MAX_THREADS_PER_BLOCK
        target cuda max_threads_per_block
    --target-cuda-registers_per_block TARGET_CUDA_REGISTERS_PER_BLOCK
        target cuda registers_per_block
    --target-cuda-keys TARGET_CUDA_KEYS
        target cuda keys options

    target sdaccel:
    --target-sdaccel-from_device TARGET_SDACCEL_FROM_DEVICE
        target sdaccel from_device
    --target-sdaccel-libs TARGET_SDACCEL_LIBS
        target sdaccel libs options
    --target-sdaccel-model TARGET_SDACCEL_MODEL
        target sdaccel model string
    --target-sdaccel-system-lib TARGET_SDACCEL_SYSTEM_LIB
        target sdaccel system-lib
    --target-sdaccel-tag TARGET_SDACCEL_TAG
        target sdaccel tag string
    --target-sdaccel-device TARGET_SDACCEL_DEVICE
        target sdaccel device string
    --target-sdaccel-keys TARGET_SDACCEL_KEYS
        target sdaccel keys options

    target composite:
    --target-composite-from_device TARGET_COMPOSITE_FROM_DEVICE
        target composite from_device
    --target-composite-libs TARGET_COMPOSITE_LIBS
        target composite libs options
    --target-composite-devices TARGET_COMPOSITE_DEVICES
        target composite devices options
    --target-composite-model TARGET_COMPOSITE_MODEL
        target composite model string
    --target-composite-tag TARGET_COMPOSITE_TAG
        target composite tag string
    --target-composite-device TARGET_COMPOSITE_DEVICE
        target composite device string
    --target-composite-keys TARGET_COMPOSITE_KEYS
        target composite keys options

    target stackvm:
    --target-stackvm-from_device TARGET_STACKVM_FROM_DEVICE
        target stackvm from_device
    --target-stackvm-libs TARGET_STACKVM_LIBS
        target stackvm libs options
    --target-stackvm-model TARGET_STACKVM_MODEL
        target stackvm model string
    --target-stackvm-system-lib TARGET_STACKVM_SYSTEM_LIB
        target stackvm system-lib
    --target-stackvm-tag TARGET_STACKVM_TAG
        target stackvm tag string
    --target-stackvm-device TARGET_STACKVM_DEVICE
        target stackvm device string
    --target-stackvm-keys TARGET_STACKVM_KEYS
        target stackvm keys options

    target aocl_sw_emu:
    --target-aocl_sw_emu-from_device TARGET_AOCL_SW_EMU_FROM_DEVICE
        target aocl_sw_emu from_device
    --target-aocl_sw_emu-libs TARGET_AOCL_SW_EMU_LIBS
        target aocl_sw_emu libs options
    --target-aocl_sw_emu-model TARGET_AOCL_SW_EMU_MODEL
        target aocl_sw_emu model string
    --target-aocl_sw_emu-system-lib TARGET_AOCL_SW_EMU_SYSTEM_LIB
        target aocl_sw_emu system-lib
    --target-aocl_sw_emu-tag TARGET_AOCL_SW_EMU_TAG
        target aocl_sw_emu tag string
    --target-aocl_sw_emu-device TARGET_AOCL_SW_EMU_DEVICE
        target aocl_sw_emu device string
    --target-aocl_sw_emu-keys TARGET_AOCL_SW_EMU_KEYS
        target aocl_sw_emu keys options

    target c:
    --target-c-unpacked-api TARGET_C_UNPACKED_API
        target c unpacked-api
    --target-c-from_device TARGET_C_FROM_DEVICE
        target c from_device
    --target-c-libs TARGET_C_LIBS
        target c libs options
    --target-c-constants-byte-alignment TARGET_C_CONSTANTS_BYTE_ALIGNMENT
        target c constants-byte-alignment
    --target-c-executor TARGET_C_EXECUTOR
        target c executor string
    --target-c-link-params TARGET_C_LINK_PARAMS
        target c link-params
    --target-c-model TARGET_C_MODEL
        target c model string
    --target-c-workspace-byte-alignment TARGET_C_WORKSPACE_BYTE_ALIGNMENT
        target c workspace-byte-alignment
    --target-c-system-lib TARGET_C_SYSTEM_LIB
        target c system-lib
    --target-c-tag TARGET_C_TAG
        target c tag string
    --target-c-interface-api TARGET_C_INTERFACE_API
        target c interface-api string
    --target-c-mcpu TARGET_C_MCPU
        target c mcpu string
    --target-c-device TARGET_C_DEVICE
        target c device string
    --target-c-runtime TARGET_C_RUNTIME
        target c runtime string
    --target-c-keys TARGET_C_KEYS
        target c keys options
    --target-c-march TARGET_C_MARCH
        target c march string

    target hexagon:
    --target-hexagon-from_device TARGET_HEXAGON_FROM_DEVICE
        target hexagon from_device
    --target-hexagon-libs TARGET_HEXAGON_LIBS
        target hexagon libs options
    --target-hexagon-mattr TARGET_HEXAGON_MATTR
        target hexagon mattr options
    --target-hexagon-model TARGET_HEXAGON_MODEL
        target hexagon model string
    --target-hexagon-llvm-options TARGET_HEXAGON_LLVM_OPTIONS
        target hexagon llvm-options options
    --target-hexagon-mtriple TARGET_HEXAGON_MTRIPLE
        target hexagon mtriple string
    --target-hexagon-system-lib TARGET_HEXAGON_SYSTEM_LIB
        target hexagon system-lib
    --target-hexagon-mcpu TARGET_HEXAGON_MCPU
        target hexagon mcpu string
    --target-hexagon-device TARGET_HEXAGON_DEVICE
        target hexagon device string
    --target-hexagon-tag TARGET_HEXAGON_TAG
        target hexagon tag string
    --target-hexagon-link-params TARGET_HEXAGON_LINK_PARAMS
        target hexagon link-params
    --target-hexagon-keys TARGET_HEXAGON_KEYS
        target hexagon keys options

    Autoscheduler options:
    Autoscheduler options, used when --enable-autoscheduler is provided

    --cache-line-bytes CACHE_LINE_BYTES
        the size of cache line in bytes.
        If not specified, it will be autoset for the current machine.
    --num-cores NUM_CORES
        the number of device cores.
        If not specified, it will be autoset for the current machine.
    --vector-unit-bytes VECTOR_UNIT_BYTES
        the width of vector units in bytes.
        If not specified, it will be autoset for the current machine.
    --max-shared-memory-per-block MAX_SHARED_MEMORY_PER_BLOCK
        the max shared memory per block in bytes.
        If not specified, it will be autoset for the current machine.
    --max-local-memory-per-block MAX_LOCAL_MEMORY_PER_BLOCK
        the max local memory per block in bytes.
        If not specified, it will be autoset for the current machine.
    --max-threads-per-block MAX_THREADS_PER_BLOCK
        the max number of threads per block.
        If not specified, it will be autoset for the current machine.
    --max-vthread-extent MAX_VTHREAD_EXTENT
        the max vthread extent.
        If not specified, it will be autoset for the current machine.
    --warp-size WARP_SIZE
        the thread numbers of a warp.
        If not specified, it will be autoset for the current machine.
    --include-simple-tasks
        whether to extract simple tasks that do not include complicated ops
    --log-estimated-latency
        whether to log the estimated latency to the file after tuning a task

    autotvm options:
    autotvm options, used when the autoscheduler is not enabled

    --tuner {ga,gridsearch,random,xgb,xgb_knob,xgb-rank}
        type of tuner to use when tuning with autotvm.
```

</div></details>

### 6-2. -march=x86 -mcpu=xxx
```
$ llc-14 -march=x86 -mattr=help
```

<details><summary>CPU list</summary><div>

```
Available CPUs for this target:

  alderlake      - Select the alderlake processor.
  amdfam10       - Select the amdfam10 processor.
  athlon         - Select the athlon processor.
  athlon-4       - Select the athlon-4 processor.
  athlon-fx      - Select the athlon-fx processor.
  athlon-mp      - Select the athlon-mp processor.
  athlon-tbird   - Select the athlon-tbird processor.
  athlon-xp      - Select the athlon-xp processor.
  athlon64       - Select the athlon64 processor.
  athlon64-sse3  - Select the athlon64-sse3 processor.
  atom           - Select the atom processor.
  barcelona      - Select the barcelona processor.
  bdver1         - Select the bdver1 processor.
  bdver2         - Select the bdver2 processor.
  bdver3         - Select the bdver3 processor.
  bdver4         - Select the bdver4 processor.
  bonnell        - Select the bonnell processor.
  broadwell      - Select the broadwell processor.
  btver1         - Select the btver1 processor.
  btver2         - Select the btver2 processor.
  c3             - Select the c3 processor.
  c3-2           - Select the c3-2 processor.
  cannonlake     - Select the cannonlake processor.
  cascadelake    - Select the cascadelake processor.
  cooperlake     - Select the cooperlake processor.
  core-avx-i     - Select the core-avx-i processor.
  core-avx2      - Select the core-avx2 processor.
  core2          - Select the core2 processor.
  corei7         - Select the corei7 processor.
  corei7-avx     - Select the corei7-avx processor.
  generic        - Select the generic processor.
  geode          - Select the geode processor.
  goldmont       - Select the goldmont processor.
  goldmont-plus  - Select the goldmont-plus processor.
  haswell        - Select the haswell processor.
  i386           - Select the i386 processor.
  i486           - Select the i486 processor.
  i586           - Select the i586 processor.
  i686           - Select the i686 processor.
  icelake-client - Select the icelake-client processor.
  icelake-server - Select the icelake-server processor.
  ivybridge      - Select the ivybridge processor.
  k6             - Select the k6 processor.
  k6-2           - Select the k6-2 processor.
  k6-3           - Select the k6-3 processor.
  k8             - Select the k8 processor.
  k8-sse3        - Select the k8-sse3 processor.
  knl            - Select the knl processor.
  knm            - Select the knm processor.
  lakemont       - Select the lakemont processor.
  nehalem        - Select the nehalem processor.
  nocona         - Select the nocona processor.
  opteron        - Select the opteron processor.
  opteron-sse3   - Select the opteron-sse3 processor.
  penryn         - Select the penryn processor.
  pentium        - Select the pentium processor.
  pentium-m      - Select the pentium-m processor.
  pentium-mmx    - Select the pentium-mmx processor.
  pentium2       - Select the pentium2 processor.
  pentium3       - Select the pentium3 processor.
  pentium3m      - Select the pentium3m processor.
  pentium4       - Select the pentium4 processor.
  pentium4m      - Select the pentium4m processor.
  pentiumpro     - Select the pentiumpro processor.
  prescott       - Select the prescott processor.
  rocketlake     - Select the rocketlake processor.
  sandybridge    - Select the sandybridge processor.
  sapphirerapids - Select the sapphirerapids processor.
  silvermont     - Select the silvermont processor.
  skx            - Select the skx processor.
  skylake        - Select the skylake processor.
  skylake-avx512 - Select the skylake-avx512 processor.
  slm            - Select the slm processor.
  tigerlake      - Select the tigerlake processor.
  tremont        - Select the tremont processor.
  westmere       - Select the westmere processor.
  winchip-c6     - Select the winchip-c6 processor.
  winchip2       - Select the winchip2 processor.
  x86-64         - Select the x86-64 processor.
  x86-64-v2      - Select the x86-64-v2 processor.
  x86-64-v3      - Select the x86-64-v3 processor.
  x86-64-v4      - Select the x86-64-v4 processor.
  yonah          - Select the yonah processor.
  znver1         - Select the znver1 processor.
  znver2         - Select the znver2 processor.
  znver3         - Select the znver3 processor.
```

</div></details>

### 6-3. -march=aarch64 -mcpu=xxx
```
$ llc-14 -march=aarch64 -mattr=help
```

<details><summary>CPU list</summary><div>

```
Available CPUs for this target:

  a64fx           - Select the a64fx processor.
  apple-a10       - Select the apple-a10 processor.
  apple-a11       - Select the apple-a11 processor.
  apple-a12       - Select the apple-a12 processor.
  apple-a13       - Select the apple-a13 processor.
  apple-a14       - Select the apple-a14 processor.
  apple-a7        - Select the apple-a7 processor.
  apple-a8        - Select the apple-a8 processor.
  apple-a9        - Select the apple-a9 processor.
  apple-latest    - Select the apple-latest processor.
  apple-m1        - Select the apple-m1 processor.
  apple-s4        - Select the apple-s4 processor.
  apple-s5        - Select the apple-s5 processor.
  carmel          - Select the carmel processor.
  cortex-a34      - Select the cortex-a34 processor.
  cortex-a35      - Select the cortex-a35 processor.
  cortex-a510     - Select the cortex-a510 processor.
  cortex-a53      - Select the cortex-a53 processor.
  cortex-a55      - Select the cortex-a55 processor.
  cortex-a57      - Select the cortex-a57 processor.
  cortex-a65      - Select the cortex-a65 processor.
  cortex-a65ae    - Select the cortex-a65ae processor.
  cortex-a710     - Select the cortex-a710 processor.
  cortex-a72      - Select the cortex-a72 processor.
  cortex-a73      - Select the cortex-a73 processor.
  cortex-a75      - Select the cortex-a75 processor.
  cortex-a76      - Select the cortex-a76 processor.
  cortex-a76ae    - Select the cortex-a76ae processor.
  cortex-a77      - Select the cortex-a77 processor.
  cortex-a78      - Select the cortex-a78 processor.
  cortex-a78c     - Select the cortex-a78c processor.
  cortex-r82      - Select the cortex-r82 processor.
  cortex-x1       - Select the cortex-x1 processor.
  cortex-x1c      - Select the cortex-x1c processor.
  cortex-x2       - Select the cortex-x2 processor.
  cyclone         - Select the cyclone processor.
  exynos-m3       - Select the exynos-m3 processor.
  exynos-m4       - Select the exynos-m4 processor.
  exynos-m5       - Select the exynos-m5 processor.
  falkor          - Select the falkor processor.
  generic         - Select the generic processor.
  kryo            - Select the kryo processor.
  neoverse-512tvb - Select the neoverse-512tvb processor.
  neoverse-e1     - Select the neoverse-e1 processor.
  neoverse-n1     - Select the neoverse-n1 processor.
  neoverse-n2     - Select the neoverse-n2 processor.
  neoverse-v1     - Select the neoverse-v1 processor.
  saphira         - Select the saphira processor.
  thunderx        - Select the thunderx processor.
  thunderx2t99    - Select the thunderx2t99 processor.
  thunderx3t110   - Select the thunderx3t110 processor.
  thunderxt81     - Select the thunderxt81 processor.
  thunderxt83     - Select the thunderxt83 processor.
  thunderxt88     - Select the thunderxt88 processor.
  tsv110          - Select the tsv110 processor.
```

</div></details>

```bash
$ sudo pip3 install xgboost
$ python -m tvm.driver.tvmc tune \
--target "llvm -mcpu=x86-64-v3" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx
```
![image](https://user-images.githubusercontent.com/33194443/155495105-f4612704-41e7-4f81-8dd3-ebeaa1981138.png)
