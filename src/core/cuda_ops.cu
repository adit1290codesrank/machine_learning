#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cstdio> 

void check_cuda(cudaError_t result, const char* msg) 
{
    if (result != cudaSuccess) {
        printf("\n[CUDA ERROR] %s\n", msg);
        printf("Details: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

void check_cublas(cublasStatus_t result, const char* msg) 
{
    if (result != CUBLAS_STATUS_SUCCESS) {
        printf("\n[CUBLAS ERROR] %s (Code: %d)\n", msg, (int)result);
        exit(1);
    }
}

cublasHandle_t handle = nullptr;

// --- STATIC BUFFERS (To avoid slow mallocs) ---
static double *d_A_buf = nullptr, *d_B_buf = nullptr, *d_C_buf = nullptr;
static size_t A_cap = 0, B_cap = 0, C_cap = 0;

void ensure_capacity(double** ptr, size_t* current_cap, size_t needed_bytes) {
    if (*current_cap < needed_bytes) {
        if (*ptr) cudaFree(*ptr);
        check_cuda(cudaMalloc(ptr, needed_bytes), "Buffer Re-alloc");
        *current_cap = needed_bytes;
    }
}

extern "C" void launch_matmul(double* h_A, double* h_B, double* h_C, int m, int k, int n) 
{
    if (handle == nullptr) check_cublas(cublasCreate(&handle), "cublasCreate Failed");

    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);

    ensure_capacity(&d_A_buf, &A_cap, size_A);
    ensure_capacity(&d_B_buf, &B_cap, size_B);
    ensure_capacity(&d_C_buf, &C_cap, size_C);

    check_cuda(cudaMemcpy(d_A_buf, h_A, size_A, cudaMemcpyHostToDevice), "Memcpy A");
    check_cuda(cudaMemcpy(d_B_buf, h_B, size_B, cudaMemcpyHostToDevice), "Memcpy B");

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B_buf, n, d_A_buf, k, &beta, d_C_buf, n), "GEMM"); 

    check_cuda(cudaMemcpy(h_C, d_C_buf, size_C, cudaMemcpyDeviceToHost), "Memcpy C");
}

__global__ void hadamard_kernel(const double* A, const double* B, double* C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) C[index] = A[index] * B[index];
}

extern "C" void launch_hadamard(double* h_A, double* h_B, double* h_C, int size)
{
    size_t bytes = size * sizeof(double);
    ensure_capacity(&d_A_buf, &A_cap, bytes);
    ensure_capacity(&d_B_buf, &B_cap, bytes);
    ensure_capacity(&d_C_buf, &C_cap, bytes);

    check_cuda(cudaMemcpy(d_A_buf, h_A, bytes, cudaMemcpyHostToDevice), "Hadamard Copy A");
    check_cuda(cudaMemcpy(d_B_buf, h_B, bytes, cudaMemcpyHostToDevice), "Hadamard Copy B");

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hadamard_kernel<<<blocks, threads>>>(d_A_buf, d_B_buf, d_C_buf, size);

    check_cuda(cudaMemcpy(h_C, d_C_buf, bytes, cudaMemcpyDeviceToHost), "Hadamard Copy C");
}

__global__ void conv2dkernel(const double* input, const double* kernel, double* output, int batch_size, int h, int w, int d, int oh, int ow, int num_filters, int k_size)
{
    int total=batch_size*num_filters*oh*ow;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=total) return;
    int j=idx%ow;
    int i=(idx/ow)%oh;
    int f=(idx/(ow*oh))%num_filters;
    int b=idx/(ow*oh*num_filters);
    double sum=0.0;
    for (int depth=0;depth<d;depth++) {
        for (int ki=0; ki < k_size; ki++) {
            for (int kj=0;kj<k_size;kj++) {
                int row_in=i+ki;
                int col_in=j+kj;
                int input_idx=b*(d*h*w)+ depth*(h*w)+row_in*w+col_in;
                int kernel_idx = f*(d*k_size*k_size)+depth*(k_size*k_size)+ki*k_size+kj;
                sum+=input[input_idx]*kernel[kernel_idx];
            }
        }
    }
    output[idx] = sum;
}

extern "C" void launch_conv2d_lean(const double* d_input, const double* d_kernel, double* d_output, int batch_size, int in_h, int in_w, int in_d, int out_h, int out_w, int num_filters, int k_size)
{
    int output_size = batch_size * num_filters * out_h * out_w;
    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;
    conv2dkernel<<<blocks, threads>>>(d_input, d_kernel, d_output, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
}

__global__ void conv2d_backward_weights_kernel(const double* input, const double* delta, double* d_kernel, double* d_bias, int batch_size, int in_h, int in_w, int in_d, int out_h, int out_w, int num_filters, int k_size)
{
    int total_elements = batch_size * num_filters * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int j = idx % out_w;
    int i = (idx / out_w) % out_h;
    int f = (idx / (out_w * out_h)) % num_filters;
    int b = idx / (out_w * out_h * num_filters);
    double d_val = delta[idx];
    atomicAdd(&d_bias[f], d_val);
    for (int d = 0; d < in_d; d++) {
        for (int ki = 0; ki < k_size; ki++) {
            for (int kj = 0; kj < k_size; kj++) {
                int row_in = i + ki;
                int col_in = j + kj;
                int input_idx = b * (in_d * in_h * in_w) + d * (in_h * in_w) + row_in * in_w + col_in;
                int kernel_idx = f * (in_d * k_size * k_size) + d * (k_size * k_size) + ki * k_size + kj;
                atomicAdd(&d_kernel[kernel_idx], input[input_idx] * d_val);
            }
        }
    }
}

__global__ void conv2d_backward_input_kernel(const double* delta, const double* kernel, double* d_input, int batch_size, int in_h, int in_w, int in_d, int out_h, int out_w, int num_filters, int k_size)
{
    int total_elements = batch_size * num_filters * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int j = idx % out_w;
    int i = (idx / out_w) % out_h;
    int f = (idx / (out_w * out_h)) % num_filters;
    int b = idx / (out_w * out_h * num_filters);
    double d_val = delta[idx];
    for (int d = 0; d < in_d; d++) {
        for (int ki = 0; ki < k_size; ki++) {
            for (int kj = 0; kj < k_size; kj++) {
                int kernel_idx = f * (in_d * k_size * k_size) + d * (k_size * k_size) + ki * k_size + kj;
                int row_in = i + ki;
                int col_in = j + kj;
                int input_idx = b * (in_d * in_h * in_w) + d * (in_h * in_w) + row_in * in_w + col_in;
                atomicAdd(&d_input[input_idx], d_val * kernel[kernel_idx]);
            }
        }
    }
}

extern "C" void launch_conv2d_backward_lean(const double* d_input, const double* d_delta, const double* d_kernel, double* d_dk, double* d_db, double* d_prev_delta, int batch_size, int in_h, int in_w, int in_d, int out_h, int out_w, int num_filters, int k_size)
{
    int delta_size = batch_size * num_filters * out_h * out_w;
    int kernel_size = num_filters * in_d * k_size * k_size;
    int input_size = batch_size * in_d * in_h * in_w;
    cudaMemset(d_dk, 0, kernel_size * sizeof(double));
    cudaMemset(d_db, 0, num_filters * sizeof(double));
    cudaMemset(d_prev_delta, 0, input_size * sizeof(double));
    int threads = 256;
    int blocks = (delta_size + threads - 1) / threads;
    conv2d_backward_weights_kernel<<<blocks, threads>>>(d_input, d_delta, d_dk, d_db, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
    conv2d_backward_input_kernel<<<blocks, threads>>>(d_delta, d_kernel, d_prev_delta, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
}

extern "C" void gpu_alloc(double** ptr, size_t size) { check_cuda(cudaMalloc(ptr, size), "gpu_alloc"); }
extern "C" void gpu_free(double* ptr) { cudaFree(ptr); }
extern "C" void gpu_memcpy_h2d(double* dest, const double* src, size_t size) { check_cuda(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice), "gpu_memcpy_h2d"); }
extern "C" void gpu_memcpy_d2h(double* dest, const double* src, size_t size) { check_cuda(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost), "gpu_memcpy_d2h"); }