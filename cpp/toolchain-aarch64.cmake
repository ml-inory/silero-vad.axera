# AArch64 交叉编译工具链（用于 AX650/AX630/AX620E 板端）
# 用法：cmake -S cpp -B cpp/build-aarch64 \
#   -DCMAKE_TOOLCHAIN_FILE=cpp/toolchain-aarch64.cmake \
#   -DTOOLCHAIN_ROOT=/path/to/aarch64-none-linux-gnu \
#   -DAX_RUNTIME_ROOT=/path/to/axrt
# 工具链：arm-none aarch64-none-linux-gnu（GCC 9.2，爱芯/ARM 官方包均可）
IF(NOT DEFINED TOOLCHAIN_ROOT)
    SET(TOOLCHAIN_ROOT "/opt/aarch64-none-linux-gnu")
ENDIF()
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(CMAKE_C_COMPILER ${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
