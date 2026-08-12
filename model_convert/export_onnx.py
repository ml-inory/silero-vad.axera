import torch
from onnxsim import simplify
import onnx
from silero import SileroVADModelforExport

jit_model = torch.jit.load("./silero_vad.jit")
jit_model.eval()
state_dict = jit_model.state_dict()
state_dict['_model.stft.forward_basis_buffer.weight'] = state_dict['_model.stft.forward_basis_buffer']

batch_size = 1
sr = 16000
hidden_size = 128
context_size = 64 if sr == 16000 else 32
context = torch.zeros(batch_size, context_size)
state = torch.zeros(2, batch_size, hidden_size)
num_samples = 512 if sr == 16000 else 256
padding = 64

model = SileroVADModelforExport()
model.eval()
model.load_state_dict(state_dict, strict=False)

input_tensor = torch.rand(1, num_samples + context_size + padding)
stft_tensor = torch.rand(1, 129, 4)

# model(input_tensor, state, context)

onnx_model = "silero_vad.onnx"
torch.onnx.export(
    model,
    (input_tensor, state),
    onnx_model,
    export_params=True,
    opset_version=16,
    # do_constant_folding=True,
    input_names=["data", "state"],
    output_names=["output", "next_state"],
    dynamic_axes=None,
    verbose=False,
)

raw_model = onnx.load(onnx_model)
shape_map = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(raw_model.graph.value_info) + list(raw_model.graph.input)}
for node in raw_model.graph.node:
    # torch 2.13 导出器会给 Split 附加 opset<18 的 num_outputs 属性，
    # 新版 onnxruntime 不再识别；Pulsar2 又要求显式 split 尺寸，
    # 这里统一改为显式等分（语义不变）。
    if node.op_type == "Split":
        in_shape = shape_map.get(node.input[0], [])
        n_out = len(node.output)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        dim = in_shape[axis] if axis < len(in_shape) else 0
        if dim and dim % n_out == 0:
            split = [dim // n_out] * n_out
        else:
            split = []
        attrs = [a for a in node.attribute if a.name not in ("num_outputs", "split")]
        if not any(a.name == "axis" for a in attrs):
            attrs.append(onnx.helper.make_attribute("axis", 1))
        del node.attribute[:]
        node.attribute.extend(attrs)
        if split and len(node.input) < 2:
            split_name = f"{node.name}_split"
            node.input.append(split_name)
            split_init = onnx.helper.make_tensor(
                split_name, onnx.TensorProto.INT64, [n_out], split,
            )
            raw_model.graph.initializer.append(split_init)
onnx.save(raw_model, onnx_model)
sim_model, _ = simplify(onnx_model)
onnx.save(sim_model, onnx_model)
print(f"Save to {onnx_model}")
