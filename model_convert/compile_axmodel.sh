#!/usr/bin/env bash
# 用 Pulsar2 将 silero_vad.onnx 编译为两个芯片的 axmodel：
#   AX650  (NPU3) -> src/silero_vad_axera/data/silero_vad_ax650.axmodel
#   AX620E (NPU2) -> src/silero_vad_axera/data/silero_vad_ax630c.axmodel
# （板子型号 AX630C 使用的是 AX620E 芯片）
#
# 前置：
#   1. cd model_convert && python export_onnx.py   # 生成 silero_vad.onnx
#   2. python generate_data.py                      # 生成 calibration_dataset/*.tar.gz
#   3. 本机有 docker + pulsar2 镜像（默认 pulsar2:7.0），
#      或设置 PULSAR2_CMD 指向本机 pulsar2 可执行文件
#
# 用法：
#   PULSAR2_IMAGE=pulsar2:7.0 ./compile_axmodel.sh
#   PULSAR2_CMD=/opt/pulsar2/bin/pulsar2 ./compile_axmodel.sh
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd .. && pwd)"
CALIB_DIR="$PWD/calibration_dataset"
OUT_DIR="$REPO_ROOT/src/silero_vad_axera/data"

for f in "$PWD/silero_vad.onnx" "$CALIB_DIR/data.tar.gz" "$CALIB_DIR/state.tar.gz"; do
  [ -f "$f" ] || { echo "缺少 $f（先运行 export_onnx.py / generate_data.py）" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"

gen_config() {
  local target=$1 hw=$2 npu=$3 out=$4 prefix=$5
  python3 - "$target" "$hw" "$npu" "$out" "$prefix" > "${target}_pulsar2_config.json" <<'PY'
import json, sys
target, hw, npu, out, prefix = sys.argv[1:]

if target == "ax650":
    # AX650 全部保持 FP32（仓库原始配置，精度最好）
    layer_configs = [{"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "FP32"}]
else:
    # AX620E(NPU2) 需显式 U16 量化（参考 rnnoise-ax620e 成功配置）；
    # 全 FP32 会让 STFT 首层卷积量化模拟出 NaN。
    u16_ops = ["Conv", "Gemm", "Add", "Mul", "Pow", "Sqrt", "Relu", "Sigmoid", "Tanh",
               "Concat", "Slice", "Split", "Squeeze", "Unsqueeze", "Gather"]
    layer_configs = []
    for op in u16_ops:
        lc = {"op_type": op, "data_type": "U16", "output_data_type": "U16"}
        if op in ("Conv", "Gemm"):
            lc["weight_data_type"] = "S8"
        layer_configs.append(lc)

cfg = {
    "model_type": "ONNX",
    "target_hardware": hw,
    "npu_mode": npu,
    "input": f"{prefix}/model_convert/silero_vad.onnx",
    "output_dir": f"{prefix}/src/silero_vad_axera/data",
    "output_name": out,
    "work_dir": f"{prefix}/model_convert/work_{target}",
    "input_shapes": "data:1x640;state:2x1x128",
    "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": False, "model_check": True},
    "quant": {
        "input_configs": [
            {"tensor_name": "data", "calibration_dataset": f"{prefix}/model_convert/calibration_dataset/data.tar.gz",
             "calibration_size": 30, "calibration_format": "Numpy"},
            {"tensor_name": "state", "calibration_dataset": f"{prefix}/model_convert/calibration_dataset/state.tar.gz",
             "calibration_size": 30, "calibration_format": "Numpy"},
        ],
        "calibration_method": "MinMax",
        "precision_analysis": True,
        "precision_analysis_method": "EndToEnd",
        "highest_mix_precision": False,
        "layer_configs": layer_configs,
    },
    "input_processors": [
        {"tensor_name": "data", "src_dtype": "FP32"},
        {"tensor_name": "state", "src_dtype": "FP32"},
    ],
    "compiler": {"check": 2},
}
print(json.dumps(cfg, indent=2, ensure_ascii=False))
PY
}

for entry in "ax650 AX650 NPU3 silero_vad_ax650.axmodel" "ax620e AX620E NPU2 silero_vad_ax630c.axmodel"; do
  set -- $entry
  gen_config "$1" "$2" "$3" "$4" "/workspace"
  if [ -n "${PULSAR2_CMD:-}" ]; then
    sed -i "s|/workspace|$REPO_ROOT|g" "${1}_pulsar2_config.json"
    $PULSAR2_CMD build --config "$PWD/${1}_pulsar2_config.json"
  else
    IMG="${PULSAR2_IMAGE:-pulsar2:7.0}"
    docker run --rm -v "$REPO_ROOT:/workspace" "$IMG" -lc \
      "PATH=/opt/pulsar2:\$PATH pulsar2 build --config /workspace/model_convert/${1}_pulsar2_config.json"
  fi
  echo "已生成 ${OUT_DIR}/${4}"
done
