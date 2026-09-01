"""Python API Overview.

Basic usage of the API via tensorrt module.
"""
import numpy as np

import tensorrt as trt
from cuda.bindings import runtime as cudart


def _check(err):
    if isinstance(err, tuple):
        err = err[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA err: {err}")

print(f"tensorrt version: {trt.__version__}")


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        print(f"severity: {severity}, msg: {msg}")


if __name__ == "__main__":

    # build phase
    # optimize a model and produce and engine

    # 1. create a logger (stdout)
    logger = trt.Logger(trt.Logger.INFO)

    logger.log(trt.Logger.INFO, "Hi I'm the logger")

    my_logger = MyLogger()
    my_logger.log(trt.Logger.ERROR, "omg an error!")

    # 2. create the builder
    builder = trt.Builder(logger)

    # 3. create a network definition
    # if more than one flag is to be combined OR'd them together
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    # To this network we could manually add layers and create from scratch or
    # import it form somewhere else like below

    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file("./models/resnet50.onnx")

    print(f"ONNX parse success: {success}")
    print(f"Number of network layers: {network.num_layers}")
    print(f"Number of inputs: {network.num_inputs}")
    print(f"Number of outputs: {network.num_outputs}")

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        print("ERROR: handle it here...")
        exit(1)

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"Input {i}:")
        print(f"  name: {tensor.name}")
        print(f"  shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}")
    # status = parser.load_model_proto(model)
    # assert status

    # status = parser.load_initializer(name, data, dataSize)
    # assert status

    # status = parser.parse_model_proto()

    # 4. Building an Engine
    # create a build configuration specifying how tensorRT should optimize the
    # model
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 16 << 30
    )

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile = builder.create_optimization_profile()

    profile.set_shape(
        input_name,
        (1, 3, 224, 224),   # MIN
        (1, 3, 224, 224),   # OPT
        (1, 3, 224, 224)    # MAX
    )

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(
        network, config
    )

    if serialized_engine is None:
        print("ERROR: failed to build serialized_engine")
        exit(1)

    with open("foo.engine", "wb") as f:
        f.write(serialized_engine)

    # 5. inference
    dummy_array = np.random.rand(1, 3, 224, 224)

    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()

    ## within the context we work with the usual cuda style
    ### allocate
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, dummy_array.shape)

    output_shape = tuple(context.get_tensor_shape(output_name))
    h_input = np.ascontiguousarray(dummy_array, dtype=np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    err_t, d_input = cudart.cudaMalloc(h_input.nbytes); _check(err_t)
    err_t, d_output = cudart.cudaMalloc(h_output.nbytes); _check(err_t)
    err_t, stream = cudart.cudaStreamCreate(); _check(err_t)

    ### host to device
    err_t = cudart.cudaMemcpyAsync(
        d_input, h_input.ctypes.data, h_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
    ); _check(err_t)

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    ### launch kernel
    context.execute_async_v3(stream)

    ### device to host
    err_t = cudart.cudaMemcpyAsync(
        h_output.ctypes.data, d_output, h_output.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
    ); _check(err_t)

    ### all along checking errors and appropriate synchronization
    err_t = cudart.cudaStreamSynchronize(stream); _check(err_t)

    ### free gpu allocated memory
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    print("End of the program!")












