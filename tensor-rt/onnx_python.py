import onnx

model = onnx.load("./models/resnet50.onnx")

print(
    *[
        (i.name, [
            d.dim_value for d in i.type.tensor_type.shape.dim
        ]) for i in model.graph.input
    ], sep="\n"
)

