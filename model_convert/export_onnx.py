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

sim_model, _ = simplify(onnx_model)
onnx.save(sim_model, onnx_model)
print(f"Save to {onnx_model}")