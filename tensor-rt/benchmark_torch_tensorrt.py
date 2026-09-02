import numpy as np

import torch
import torch.cuda.profiler as profiler
import torch_tensorrt

from transformers import ViTForImageClassification


def benchmark_tensorrt(trt_compiled, input_tensor, num_warmup=10, num_runs=100):
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            trt_compiled(input_tensor)
        torch.cuda.synchronize()

        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        for i in range(num_runs):
            start_events[i].record()
            trt_compiled(input_tensor)
            end_events[i].record()
        torch.cuda.synchronize()
        latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        latencies = np.array(latencies)
        print(f"\nLatency over {num_runs} runs (ms):")
        print(f"  Mean:   {np.mean(latencies):.2f}")
        print(f"  Median: {np.median(latencies):.2f}")
        print(f"  Min:    {np.min(latencies):.2f}")
        print(f"  Max:    {np.max(latencies):.2f}")
        print(f"  P90:    {np.percentile(latencies, 90):.2f}")
        print(f"  P95:    {np.percentile(latencies, 95):.2f}")
        print(f"  P99:    {np.percentile(latencies, 99):.2f}")


def compile_tensorrt(model, dryrun=False, optimization_level=None):
    inputs = [torch.rand((128, 3, 224, 224), dtype=torch.float16).cuda()]

    with torch_tensorrt.dynamo.Debugger(log_level="error"):
        trt_compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            truncate_double=True,
            decompose_attention=True,
            dryrun=dryrun,
            optimization_level=optimization_level,
            workspace_size=128 << 30,
        )
    return trt_compiled


if __name__ == "__main__":
    print("INFO:")
    print("INFO: pulling model from hugging face")
    print("INFO:")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224", hidden_act="gelu_fast"
    ).eval().half().cuda()

    print("INFO:")
    print("INFO: compiling model")
    print("INFO:")

    trt_compiled = compile_tensorrt(model, dryrun=True)

    input_tensor = torch.rand((128, 3, 224, 224), dtype=torch.float16).cuda()

    trt_compiled = compile_tensorrt(
        model,
        optimization_level=None,
    )

    print("INFO:")
    print("INFO: benchmarking model")
    print("INFO:")

    with profiler.profile():
        benchmark_tensorrt(trt_compiled, input_tensor)

    print("INFO:")
    print("INFO: finish")
    print("INFO:")

