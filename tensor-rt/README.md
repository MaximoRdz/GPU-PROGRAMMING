# TensorRT: Accelerated Inference (torch-tensorrt in our case)

TensorRT is an SDK for high-performance deep learning inference across GPU-accelerated platforms. Compile your PyTorch model with it and use it on different NVIDIA platforms.
It is not constrained to PyTorch; you can also take your model trained in other frameworks like TensorFlow or ONNX. They'll be compiled into engines for GPU deployment with support for mixed precision (fp32, fp16, bf16, fp8, int8, fp4, int4), dynamic shapes, specialized optimization for different architectures, etc.

TensorRT optimizations such as FP16 and INT8 reduced precision are available, and it also offers a fallback to native PyTorch when TensorRT does not support the model subgraphs.

## Too Long, Didn't Read

- Compilation settings list: https://docs.pytorch.org/TensorRT/user_guide/compilation/torch_compile.html
    - Dynamo defaults source: https://github.com/pytorch/TensorRT/blob/main/py/torch_tensorrt/dynamo/_defaults.py
- Torch-TensorRT tuning fundamentals: https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html

## Complementary GPU Features

Beyond engine compilation and runtime APIs, NVIDIA GPUs expose platform features for sharing or partitioning hardware among concurrent workloads.

### Multi-Instance GPU

Feature of NVIDIA GPUs (> Ampere) that enables `user-directed partitioning of a single GPU into multiple smaller GPUs`. These physical partitions provide dedicated compute and memory slices with quality of service, also supporting independent execution of parallel workloads on fractions of the GPU.

**Applications with low GPU utilization** in TensorRT can utilize MIG to increase throughput with little or no latency impact. `Optimal partitioning` is application-specific and should be used when strict hardware isolation is required.

### Multi-Device Execution

Feature for partitioning a single network across multiple GPUs, enabling multi-GPU inference. Each GPU runs its own instance of the TensorRT engine as a distinct rank and exchanges intermediate tensors with other ranks using distributed collective primitives.

**Applications with large models or latency dominated by compute** are likely to scale and perform nicely in this setup (at the cost of inter-GPU communication).

## TensorRT Fundamentals

Take a trained deep learning model and turn it into a fast GPU engine to compute predictions on new inputs.

1. Build phase: compile the network and select the fastest available kernel for each layer on your target GPU. Output is a `serialized` binary called an engine (aka *plan file*).
2. Runtime phase: load the engine into your application and execute it on the GPU.

You are expected to control:
- TensorRT objects, ownership, usage, how long each needs to live, memory allocation, and reutilization
- Thread-safe and not-thread-safe operations
- Determinism guarantees
- **Lean runtime** and **dispatch runtime** trade-offs against the full runtime when shipping to production

### Object Lifetimes

TensorRT uses a factory design pattern; objects create and own other objects.

![tensorrt-objects-lifetimes](https://docs.nvidia.com/deeplearning/tensorrt/latest/_images/trt-object-lifetimes.svg)

Owned by user: the lifetime of a factory object must span the lifetime of the objects it creates. E.g., the builder object creates, and therefore owns, `NetworkDefinition` and `BuilderConfig`; hence those two must be destroyed before the builder factory object.

> *Exception*: creating an engine from a builder. You can destroy the builder and keep using the engine.

#### Engine Deserialization and Trust Boundary

Engine files contain:
- Model graph
- Compiled CUDA tactics
- Plugin invocation pointers
- Etc. (state that `IRuntime::deserializeCudaEngine` instantiates inside the process)

> Deserializing an engine is **equivalent** to executing untrusted native code on the GPU.

### Error Handling and Logging

TensorRT top-level interfaces (builder, runtime, ...) require our own implementation of the `Logger` interface (it must be thread-safe). The logger will propagate down and be used downstream:
`runtime logger -> creates execution context -> enqueue` will use the upper logger.

[Python Error Recorder interface](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ErrorRecorder.html): CUDA errors are generally asynchronous; the main problem is that some CUDA errors are **sticky** and contagious, meaning other execution contexts from different engines could show the same unrelated error. Refer to the [Cross-Context CUDA Error Isolation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html#cross-context-cuda-error-isolation).

### Memory

TensorRT uses quite a lot of GPU memory as opposed to host memory.

#### The Build Phase

TRT allocates device memory for timing layer implementations (large amounts of temporary memory for large tensors); the maximum allowed temporary memory can be controlled through the `memory pool limits` of the builder config. The `workspace size` defaults to the full size of the device's global memory but can also be restricted at will. Timing requires creating buffers for input, output, and weights; TRT directly handles out-of-memory (OOM) operating system errors coming from such allocations.

During the build phase, at least **2 copies of the model weights** are in host memory: (1) the original network and (2) those in the engine as we build it. Weight combination, e.g., `convolution + batch normalization`, requires some extra host memory as well.

> All this affects mainly host memory and can be monitored via `trtexec --monitorMemory`.

#### The Runtime Phase

Opposite of before, now host memory is barely used, but it can use considerable device memory.

- Memory allocation on device, model weights after deserialization of the engine
- Statistics about the engine can be retrieved via `ICudaEngine::getEngineStat()`
    - `kTOTAL_WEIGHT_SIZE`: total size in bytes of the weights
    - `kSTRIPPED_WEIGHT_SIZE`: size in bytes of stripped weights for stripped-built engines
- `ExecutionContext` memory usage
    - Some layer implementations require persistent memory, e.g., convolutions with edge masks, and these cannot be shared across contexts because their size is input-dependent. Memory is allocated at the creation of the context and lasts for its lifetime.
    - Enqueue memory holds intermediate results while processing the network: intermediate tensors (activation memory), temporary storage (scratch memory, bounded via `setMemoryPoolLimit()`). TRT does optimize this memory usage by:
        - Sharing a block of device memory across activation tensors with disjoint lifetimes
        - Allowing transient (scratch) tensors to occupy unused activation memory where feasible
    - Enqueue memory is a very rich area of TensorRT:
        - Memory range `[total activation memory, total activation memory + scratch memory]`
        - `ICudaEngine::createExecutionContextWithoutDeviceMemory()`: no enqueue memory configured, use that memory for something else

> By default, TensorRT allocates device memory directly from CUDA. However, you can attach an implementation of TensorRT's `IGpuAllocator` interface (C++, Python) to the builder or runtime and manage device memory yourself.

#### CUDA Lazy Loading

Reduces peak GPU and host memory usage of TRT and speeds up initialization (they say < 1% performance impact). Environment variable `CUDA_MODULE_LOADING=LAZY`.

#### L2 Persistent Cache Management

For NVIDIA architectures later than Ampere, L2 cache persistence is supported. It allows for L2 cache lines to be retained when a line is chosen for eviction, so you can choose what you wish to retain to reduce DRAM traffic and power consumption. Cache allocation is per-execution context (`setPersistentCacheLimit`), and the total persistent cache among all contexts should not exceed `cudaDeviceProp::persistingL2CacheMaxSize`.

#### Threading

Shared objects: the runtime deserializes the engines, and each engine can create an execution context, each on a thread with its own network state. From there we can distinguish two types of operations:
- Thread-safe: non-modifying access to runtime/engine, deserializing an engine, creating an execution context, and registering/deregistering plugins
- Not thread-safe: concurrent access to shared objects from different threads, and using the same execution context from different threads

![threading](https://docs.nvidia.com/deeplearning/tensorrt/latest/_images/trt-threading-model.svg)

This is a delicate topic, as there are operations that are thread-safe, like using multiple builders, but in spite of that, they can still interfere with each other; e.g., the builder's timing interface will not yield the best kernel possible if the GPU is being utilized by something else.

#### Determinism

> The TensorRT builder uses timing to find the fastest kernel to implement a given layer. Timing kernels are subject to noise, such as other work running on the GPU and GPU clock speed fluctuations.

> The Editable Timing Cache mechanism allows you to force the builder to pick a particular implementation for a given layer. Use this to ensure the builder picks the same kernels from run to run.

#### Runtime Options

This is useful for C++ applications (link to the appropriate library): default, lean (lighter), and dispatch. For Python applications, the same can be accessed via different packages: `tensorrt`, `tensorrt_lean`, and `tensorrt_dispatch`.

## End-to-End Naive Tutorial

**Requirements:**
1. NVIDIA TensorRT 11.2.1 (`trtexec` tool on PATH)
2. CUDA Toolkit 13.x

```bash
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx \
    -O models/resnet50.onnx

trtexec --onnx=./models/resnet50.onnx --saveEngine=resnet50.engine

trtexec --loadEngine=resnet50.engine --shapes=data:1x3x224x224

python onnix_python.py
```

Pull a trained model from somewhere else, compile and build a GPU-specific engine, and finally, the runtime can load and execute that engine — a `minimum viable inference loop` setup.

## Torch-TensorRT Compiler Architecture

Consists of three phases:

### 1. Lowering the TorchScript Module

> TorchScript is an intermediate representation of a PyTorch model that allows for serialization, optimization, and deployment in high-performance environments like C++ without requiring a Python runtime.

Torch-TensorRT lowers the TorchScript module, simplifying implementations of common operations into representations that map more directly to TensorRT. This lowering pass does not affect the graph itself.

### 2. Conversion

Torch-TensorRT identifies TensorRT-compatible subgraphs and translates them to TensorRT operations:
- Nodes with static values are evaluated and mapped to constants
- Tensor computations can be converted to one or more TensorRT layers
- Incompatible nodes stay in TorchScript and fall back to JIT (just-in-time) compilation

![conversion example](https://developer-blogs.nvidia.com/wp-content/uploads/2021/12/transforming-conv2d-layer-to-tensorrt-engine.png)

### 3. Execution

The compiled module is executed transparently to the user, who doesn't know whether the PyTorch op is being executed by the TensorRT engine or by usual PyTorch.

![example execution](https://developer-blogs.nvidia.com/wp-content/uploads/2021/11/Image4-4.png)

## Torch-TensorRT Features

- Support for INT8 via (1) post-training quantization and (2) quantization-aware training
- Sparsity: maximum throughput exploiting sparsity (tensor cores), e.g., in convolution and fully connected layers

## Benchmarking

Turning a TensorRT model into trustworthy numbers (latency, throughput, per-layer cost).

> Without a stable measurement baseline, every optimization you try is a guess.

## Torch-TensorRT Explained

### Dynamo Frontend

Default frontend for Torch-TensorRT, based on the Dynamo compiler stack from PyTorch.

> The PyTorch compiler stack centers on TorchDynamo, a Python-level JIT compiler that hooks into CPython's frame evaluation API to dynamically rewrite bytecode and extract tensor operations into an FX Graph.

In a nutshell, Dynamo is the one to blame when `torch.compile()` doesn't work. In the end, Dynamo is a tracer: given a function, it executes it and stores a linear sequence of instructions (without control flow) into a graph (FX graph), which is nothing but a list of function calls. Since there is no control flow, the trace of a function clearly depends on the inputs, as the control flow is resolved by them, and we can store the linear instructions at execution time, when we know, e.g., `fn(x, 2)`.

An important property of Dynamo is that it does know how to treat dynamic shapes! This way, even though training might be done with, for example, a fixed batch size, the compiled forward function can be reused without extra compilation.

> To summarize: 1. Dynamo is a Python tracer. 2. Given some inputs, it returns an FX graph with the PyTorch functions that were executed. 3. It can also trace integers if it detects that they changed between calls. 4. It specializes any other value that is not a tensor or a scalar.

### `torch.compile` (Just-In-Time)

JIT compiler stack; compilation is deferred until first use. More flexibility, but limits serialization.
Under the hood, `torch.compile` delegates the subgraphs it believes can be lowered to Torch-TensorRT, and then those parts will be directly executed as TensorRT engines (when possible), and the rest of the subgraphs will stay in PyTorch as a hybrid subgraph.

### AOT Compilation

Ahead-Of-Time compilation is done by `torch_tensorrt.dynamo.compile` (models compiled in an explicit compilation phase); same as `torch.compile`, but now the AOT and hybrid amalgamation of TensorRT engine and PyTorch subgraphs can be serialized and reloaded in the future directly.

> There are also other **legacy** frontends, like TorchScript: `torch_tensorrt.ts.compile`, which can still be used by choosing the compiler path setting `ir` (intermediate representation) to the appropriate value:
> - `torch.compile`: JIT first-use compilation
> - `dynamo`: compiled result can be run directly or saved for later with `torch.export.export` or `torch_tensorrt.save`
> - `torchscript`: same as dynamo, but legacy frontend; input module must be scriptable
> - `fx`: `torch.fx` stack

## References

1. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
2. https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html#best-practices
4. https://docs.pytorch.org/TensorRT/user_guide/torch_tensorrt_explained.html

## TODO
- [ ] Implement weight combination of layers optimization, e.g., `convolution + batch norm`
