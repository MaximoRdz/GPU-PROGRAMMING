# Tensor RT: Accelerated Inference (torch-tensorrt in our case)

TensorRT is an SDK for high-performance deep learning inference across GPU-accelerated platforms. Compile your pytroch with it and use in different NVIDIA platforms.
It is not constraint to pytorch, just take your model trained in other frameworks like tensorflow or onnix. They'll be compiled into engines for GPU deployment with support for mixed precision (fp32, fp16, bf16, fp8, int8, fp4, int4), dynamic shapes, specialized optimization for different architectures etc.

TensorRT optimizations such as FP16 and INT8 reduced precision are available and also offers a fallback to native pytorch when TensorRT does not support the model subgraphs.

## Complementary GPU Features

Beyond engine compilation and runtim APIs, NVIDIA GPUs expose platform features for sharing or partitioning hardware among concurrent workloads.

### Multi-instance GPU
Feature of NVIDIA GPUs ($\gt$ampere) that enables `user-directed partitioning of a single GPU into multiple smaller GPUs`. These physical partitions provide dedicated compute and memory slices with quality of service, supporting as well independent execution of parallel workloads on fractions of the GPU.

**Applications with low GPU utilization** in tensorRT can utilize MIG to increase throughput with little or no latency impact. `optimal partioning` is application specfic and should be used when strict hardware isolation is required.

### Multi-device Execution

Feature for partitioning a single network across multiple GPUs, enabling multi-GPU inference. Each GPU runs its own instance of the tensorRT engine as a distinct rank and exchanges intermediate tensors with other ranks using distributed collective primitives.

**Applications with large models or latency dominated by compute** are likely to scale and performe nicely on this set up (at the cost of inter-GPU communication)

## TensorRT Fundamentals

Trained deep learning model and turn into fast GPU-engine, compte predictions on new inputs.

1. build-phase: compile the network and select the fastest available kernel for each layer on your target GPU. Output is a `serialized` binary called engine (aka *plan file*)
1. runtime phase: load the engine into your application and execute it on the GPU

You are expected to control:
- tensorrt objects, ownership, usage, how long each needs to live, memory allocation and reutilization
- thread-safe and not-thread-safe operations
- determinism guarantees
- **lean runtime** and **dispatch runtime** trade off against the full runtime when shippnig to production

### Objects Lifetimes

TensorRT uses a factory design pattern, objects create and own other objects. 

![tensorrt-objects-lifetimes](https://docs.nvidia.com/deeplearning/tensorrt/latest/_images/trt-object-lifetimes.svg)

Owned by user: the lifetime of a factory object must span the lifetime of objects it creates. E.g. the builder object creates and therfor owns networkdefinition and builderconfig, hence those two must be destroyed before the builder factory object.

> *Execption: creating and engine from a builder. You can destroy the builder and keep using the engine


## End2end Naive tutorial

* **Reqs**
1. nvidia tensorRT 11.2.1 (trtexec tool on Path)
1. cuda toolkit 13.x

```bash
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx \
    -O models/resnet50.onnx

trtexec --onnx=./models/resnet50.onnx --saveEngine=resnet50.engine

trtexec --loadEngine=resnet50.engine --shapes=data:1x3x224x224

python onnix_python.py
```

Pull trained model from somewhere else, compile and build a GPU-specific engine, and finally, runtime can load and execute that engine `minimum viable inference loop set up`

## Torch-TensorRT compiler's architecture

Consists of three phases:

### 1. Lowering the torchscript module

> TorchScript is an intermediate representation of a PyTorch model that allows for serialization, optimization, and deployment in high-performance environments like C++ without requiring a Python runtime.

Torch-tensorrt flowers the torchscript module, simplifying implmeentations of common operations to representations that map more directly to tensorrt. this lowering pass does not affect the graph itself

### 2. Conversion

torch-tensorrt identifies tensorrt compatible subgraphs and translates them to tensorrt operations:
- nodes with static values are evaluated and mapped to constants
- tensor computations can be converted to one or more tensorrt layers
- no compatible nodes stay in torch scripting and fall back to JIT (just in time) compilation

![conversion example](https://developer-blogs.nvidia.com/wp-content/uploads/2021/12/transforming-conv2d-layer-to-tensorrt-engine.png)

### 3. Execution

the compile module is executed transparently to the user who doesn't know whether the pytorch op is being executed by the tensorrt engine or by usual pytorch

![example execution](https://developer-blogs.nvidia.com/wp-content/uploads/2021/11/Image4-4.png)

## torch-tensorrt features

* support for INT8 via (1) post-training quatization and (2) quatization-aware training
* sparsity: maximum throughput exploiting sparsity (tensor cores) e.g. in convolution and fully connected layers

## Benchmarking

Turning a TensorRT model into trustworthy numbers (latency, throughput, per-layer cost)

> Without a stable measurement baseline, every optimization you try is a guess.

# References
1. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
1. https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
1. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html#best-practices
