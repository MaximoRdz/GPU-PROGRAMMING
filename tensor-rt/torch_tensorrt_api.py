import torch
import torch_tensorrt as trt

from typing import Any


class Model(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        x = self.relu(x)
        y = self.relu(y)

        add = x + y

        return torch.mean(add)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"INFO: using device {DEVICE}")

sample_inputs = [torch.rand((5, 7)).to(DEVICE), torch.rand((5, 7)).to(DEVICE)]

model = Model().to(DEVICE).eval()

torch._logging.set_logs(graph_code=True)

opt_model = torch.compile(model)

with torch.no_grad():
    opt_model(*sample_inputs)

torch._dynamo.reset()

print("##################################################")
print("##################################################")

opt_model = torch.compile(
    model,
    backend="torch_tensorrt",
    dynamic=False,
    options={
        "min_block_size": 4, # model forward has 4 operators so the minimum must be lower from default 5
    }
)

with torch.no_grad():
    opt_model(*sample_inputs)

print("##################################################")
print("##################################################")

opt_model = trt.compile(
    model,
    ir="dynamo",
    inputs=sample_inputs,
    min_block_size=4, # model forward has 4 operators so the minimum must be lower from default 5
)

with torch.no_grad():
    opt_model(*sample_inputs)


