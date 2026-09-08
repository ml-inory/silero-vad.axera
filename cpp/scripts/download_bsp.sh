#!/usr/bin/env bash
# 一键下载 C++ 交叉编译所需 BSP 工具链：Arm GNU AArch64 GCC 9.2
# （aarch64-none-linux-gnu，AX650/AX620E 通用；AX runtime 头文件/库已随仓库
#   放在 cpp/axrt/，无需再下载 MSP SDK）。
#
# 用法：
#   bash cpp/scripts/download_bsp.sh                 # 官方源
#   TOOLCHAIN_URL=<镜像地址> bash cpp/scripts/download_bsp.sh
#   TOOLCHAIN_ROOT=/path/to/gcc-arm-... bash cpp/scripts/download_bsp.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # 到 cpp/

ARCHIVE="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
URL="${TOOLCHAIN_URL:-https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${ARCHIVE}}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$PWD/third_party/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
GXX="$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++"

if [ -x "$GXX" ]; then
    echo "toolchain already exists: $TOOLCHAIN_ROOT"
    exit 0
fi

mkdir -p "$(dirname "$TOOLCHAIN_ROOT")"
TMP="$TOOLCHAIN_ROOT.tar.xz"
echo "downloading $URL"
echo "  -> $TMP"
curl -fL --retry 3 -C - -o "$TMP" "$URL"
echo "extracting ..."
tar -xJf "$TMP" -C "$(dirname "$TOOLCHAIN_ROOT")"
rm -f "$TMP"

if [ ! -x "$GXX" ]; then
    echo "ERROR: toolchain missing after extraction: $GXX" >&2
    exit 1
fi
echo "toolchain ready: $TOOLCHAIN_ROOT"
"$GXX" --version | head -1
