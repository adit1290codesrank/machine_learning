@echo off
setlocal enabledelayedexpansion

if not exist "obj" mkdir obj
if not exist "obj\core" mkdir obj\core

echo [1/3] Compiling CUDA Kernels...
nvcc -allow-unsupported-compiler -c src/core/cuda_ops.cu -o obj/core/cuda_ops.obj -O3
if %errorlevel% neq 0 (
    echo [!] CUDA Compile Failed.
    exit /b %errorlevel%
)

echo [2/3] Compiling C++ Files...
set CPP_FILES=src/main.cpp src/network.cpp src/core/matrix.cpp src/core/utils.cpp src/layers/dense.cpp src/layers/conv2d.cpp src/layers/batchnorm.cpp src/layers/pooling.cpp src/layers/softmax.cpp src/io/data.cpp src/activation.cpp

nvcc -allow-unsupported-compiler %CPP_FILES% obj/core/cuda_ops.obj -o main.exe -O3 -I./include -lcublas -Xcompiler "/openmp"

if %errorlevel% neq 0 (
    echo [!] Build Failed.
    exit /b %errorlevel%
)

echo [3/3] Build Successful! Run main.exe to start.