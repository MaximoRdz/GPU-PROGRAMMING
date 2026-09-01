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

#### Engine Deserialization and TRUST BOUNDARY

engine files:
- model graph
- compiled cuda tactics
- plugin invocation pointers
- etc. (state that `IRunTime::deserializeCudaEngine` instantiates inside process)

> Deserializing and engine is **equivalent** to executing untrusted native code on the GPU

### Error Handling and Logging

TensorRT top-level interfaces (builder, runtime, ...) required our own implementation of the `Logger` interface (it must be thread-safe). The logger will propagate down and be used downstream:
`runtime logger -> creates execution context -> enqueue` will use the upper logger.

[Python Error Recoder interface](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ErrorRecorder.html): CUDA errors are generally asynchronous, the main problem is, then, that somw CUDA errors are **sticky** and contageous, meaning other execution contexts from different engines could show the same unrelated error. Refer to the [Cross-Context cuda error isolation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html#cross-context-cuda-error-isolation)

### Memory

TensorRT uses quite a lot of GPU memory as opposed to host memory.

#### The build phase

TRT allocates device memory for timing layer implementations (large amounts of temporary memory for large tensors), maximum allowed temporary memory can be controled  through `memory pool limits` of the builder config. the `workspace size` defaults to the full size of the device's global memory but can also be restricted at will. Timing requires creating buffers for intput, output and weights, TRT directly handle Out-of-memory (OOO) operating system errors coming from such allcoations.

During the build phase at least **2 copies of the model weights** are in host memory (1) original network and (2) those in the engine as we build it. Weight combination, e.g. `convolution + batch normalization` requires some extra host memory as well.

> all this affects mainly host memory and can be monitored via `trtexec --monitorMemory`

#### The runtime phase

opposite as before, now host memory is barely used but can use considerable device memory.

* Memory allocation on device, model weights after deserialization of the engine
* statistics about the engine can be retrieved via `ICudaEngine::getEngineStat()`
    - `ktotal-weight-size` total size in bytes of the weights
    - `kstripped-weight-size` size in bytes of stripped weights for stripped-built engines
* `ExecutionContext` memory usage
    - some layers implementations require persistent memory, e.g. convolutions with edge masks, and these cannot be shared accross contexts because its size is input-dependant. Mmemory allcoated at the craetion of the context and lasts for its lifetime
    - enqueue memory holds intermediate results while processing the network: intermediate tensors (activation memory), temporary storage (scratch memory, bounded via `setMemoryPoolLimit()`). TRT does optimize this memory usage:
        - sharing a block of device memory across activatoin tensors with disjoint lifetimes
        - allowing transient (scratch) tensors to occupy unused activation memory where feasible
    * enqueue memory is a very rich area of tensorRT:
        - memory range `[total activation memory, total activation memory + scratch memory]`
        - `ICudaEngine::createExecutionContextWithoutDeviceMemory()` no enqueue memory config, use that memory for some else

> By default, TensorRT allocates device memory directly from CUDA. However, you can attach an implementation of TensorRT’s IGpuAllocator interface (C++, Python) to the builder or runtime and manage device memory yourself.

#### CUDA Lazy loading

reduce peak GPU and host memory usage of TRT and speed up initialization (they say $\lt 1\%$ performance impact). Environment variable `CUDA_MODULE_LOADING=LAZY`

#### L2 Persistent Cache Managment

For NVIDIA architectures later than Ampere L2 cache persistence is supported. It allows for L2 cache lines for retention when a line is chosen for eviction, choose what you wish to retain to reduce DRAM traffic and power consumption. Cache-allocation is per-execution context (`setPersistentCacheLimit`) and the total persistent cache among all contexts should not exceed `cudaDeviceProp::persistingL2CacheMaxSize`.

#### Threading

Shared objects: runtime deserializes the engines, each engine can create an execution context each on a thread with its own network state. From there we can distinguish two types of operations:
- thread-safe: non-modifiying access to runtime/engine, deserializing an engine, creating executing context and registering/deregistering plugins
- not thread-safe: concurrent access to shared objects from different threads and using the same execution context from different threads

![threading](https://docs.nvidia.com/deeplearning/tensorrt/latest/_images/trt-threading-model.svg)

This is a delicate topic as there are operations thread-safe like using multiple builders but, in spite of that, they can still interfer with each other, e.g. the buildir timing interface will not yield the best kernel possible if the GPU is utilized by something else.

#### Determinism

> The TensorRT builder uses timing to find the fastest kernel to implement a given layer. Timing kernels are subject to noise, such as other work running on the GPU and GPU clock speed fluctuations. 

> The Editable Timing Cache mechanism allows you to force the builder to pick a particular implementation for a given layer. Use this to ensure the builder picks the same kernels from run to run. 

#### Runtime Options

This is usefull for C++ applications (link to the appropriate library), default, lean (lighter), and dispatch. For python applications the same can be accessed via different packages: tensorrt, tensorrt_lean, and tensorrt_dispatch.

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

# TODO
- [ ] implement weight combination of layers optimization, e.g. `convolution + batch norm`
