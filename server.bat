@echo off
setlocal enabledelayedexpansion

if not exist "obj" mkdir obj
if not exist "obj\core" mkdir obj\core

echo [1/2] Checking CUDA Kernels...
if not exist "obj\core\cuda_ops.obj" (
    echo Compiling CUDA Kernels...
    nvcc -allow-unsupported-compiler -c src/core/cuda_ops.cu -o obj/core/cuda_ops.obj -O3 -arch=sm_86
)

echo [2/2] Compiling Server...
set CPP_FILES=src/server.cpp src/network.cpp src/core/matrix.cpp src/core/utils.cpp src/layers/dense.cpp src/layers/conv2d.cpp src/layers/batchnorm.cpp src/layers/pooling.cpp src/layers/softmax.cpp src/io/data.cpp src/activation.cpp src/layers/dropout.cpp

nvcc -allow-unsupported-compiler %CPP_FILES% obj/core/cuda_ops.obj -o server.exe -O3 -I./include -lcublas -Xcompiler "/openmp" -arch=sm_86

if %errorlevel% neq 0 (
    echo [!] Build Failed.
    exit /b %errorlevel%
)

echo [!] Build Successful! Created server.exe