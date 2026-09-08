#!/usr/bin/env bash
# 一键交叉编译 C++ SDK（AX650 / AX620E）：
#   工具链缺失时自动下载（见 scripts/download_bsp.sh），随后 cmake 配置并编译。
#
# 用法：
#   bash cpp/scripts/build.sh
#   TOOLCHAIN_ROOT=/path/to/gcc-arm-... bash cpp/scripts/build.sh   # 复用已装工具链
set -euo pipefail
cd "$(dirname "$0")/.."   # 到 cpp/

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$PWD/third_party/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
AX_RUNTIME_ROOT="${AX_RUNTIME_ROOT:-$PWD/axrt}"
ROOT="$(cd .. && pwd)"
BUILD_DIR="$ROOT/cpp/build-aarch64"

if [ ! -x "$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++" ]; then
    echo "[build] toolchain not found at $TOOLCHAIN_ROOT, downloading ..."
    TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" bash "$ROOT/cpp/scripts/download_bsp.sh"
fi

MAKE="$(command -v make || command -v gmake || true)"
if [ -z "$MAKE" ]; then
    echo "ERROR: make not found" >&2
    exit 1
fi

echo "[build] cmake configure (toolchain=$TOOLCHAIN_ROOT runtime=$AX_RUNTIME_ROOT)"
cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cpp/toolchain-aarch64.cmake" \
    -DCMAKE_MAKE_PROGRAM="$MAKE" \
    -DTOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
    -DAX_RUNTIME_ROOT="$AX_RUNTIME_ROOT"

echo "[build] compiling ..."
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

echo "[build] OK: $BUILD_DIR/silero_vad_example"
file "$BUILD_DIR/silero_vad_example"
