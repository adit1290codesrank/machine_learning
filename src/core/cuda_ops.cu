#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cstdio> 

void check_cuda(cudaError_t result, const char* msg) 
{
    if (result != cudaSuccess) 
    {
        printf("\n[CUDA ERROR] %s\n", msg);
        printf("Details: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

void check_cublas(cublasStatus_t result, const char* msg) 
{
    if (result != CUBLAS_STATUS_SUCCESS) 
    {
        printf("\n[CUBLAS ERROR] %s (Code: %d)\n", msg, (int)result);
        exit(1);
    }
}

cublasHandle_t handle = nullptr;

extern "C" void cuda_matmul(const double* A, const double* B, double* C, int m, int k, int n) 
{
    if (m <= 0 || k <= 0 || n <= 0) 
    {
        printf("\n[CRASH PREVENTED] Invalid Dimensions: %dx%d * %dx%d\n", m, k, k, n);
        exit(1);
    }
    if (A == nullptr || B == nullptr || C == nullptr) 
    {
        printf("\n[CRASH PREVENTED] Null Pointer passed to GPU!\n");
        exit(1);
    }

    if (handle == nullptr) 
    {
        printf("[GPU] Initializing cuBLAS... ");
        check_cublas(cublasCreate(&handle), "cublasCreate Failed");
        printf("Success!\n");
    }

    double *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc((void**)&d_A, m * k * sizeof(double)), "Malloc A Failed");
    check_cuda(cudaMalloc((void**)&d_B, k * n * sizeof(double)), "Malloc B Failed");
    check_cuda(cudaMalloc((void**)&d_C, m * n * sizeof(double)), "Malloc C Failed");

    check_cuda(cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice), "Memcpy A (Host->GPU) Failed");
    check_cuda(cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice), "Memcpy B (Host->GPU) Failed");
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,          
                             &alpha,
                             d_B, n,           
                             d_A, k,           
                             &beta,
                             d_C, n), "GEMM Math Execution Failed"); 

    check_cuda(cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost), "Memcpy C (GPU->Host) Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void conv2dkernel(const double* input,const double* kernel,double* output,int batch_size,int h,int w,int d,int oh, int ow,int num_filters,int k_size)
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
        for (int ki=0; ki < k_size; ki++) 
        {
            for (int kj=0;kj<k_size;kj++) 
            {
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

extern "C" void launch_conv2d_lean(const double* d_input, const double* d_kernel, double* d_output,int batch_size, int in_h, int in_w, int in_d,int out_h, int out_w, int num_filters, int k_size)
{
    int output_size = batch_size * num_filters * out_h * out_w;
    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    conv2dkernel<<<blocks, threads>>>(d_input, d_kernel, d_output, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
}

__global__ void conv2d_backward_weights_kernel(const double* input, const double* delta, double* d_kernel, double* d_bias,int batch_size, int in_h, int in_w, int in_d,int out_h, int out_w, int num_filters, int k_size)
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

    for (int d = 0; d < in_d; d++) 
    {
        for (int ki = 0; ki < k_size; ki++) 
        {
            for (int kj = 0; kj < k_size; kj++) 
            {
                int row_in = i + ki;
                int col_in = j + kj;
                
                int input_idx = b * (in_d * in_h * in_w) + d * (in_h * in_w) + row_in * in_w + col_in;
                double val = input[input_idx];
                
                int kernel_idx = f * (in_d * k_size * k_size) + d * (k_size * k_size) + ki * k_size + kj;
                
                atomicAdd(&d_kernel[kernel_idx], val * d_val);
            }
        }
    }
}

__global__ void conv2d_backward_input_kernel(const double* delta, const double* kernel, double* d_input,int batch_size, int in_h, int in_w, int in_d,int out_h, int out_w, int num_filters, int k_size)
{
    int total_elements = batch_size * num_filters * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int j = idx % out_w;
    int i = (idx / out_w) % out_h;
    int f = (idx / (out_w * out_h)) % num_filters;
    int b = idx / (out_w * out_h * num_filters);

    double d_val = delta[idx];

    for (int d = 0; d < in_d; d++) 
    {
        for (int ki = 0; ki < k_size; ki++) 
        {
            for (int kj = 0; kj < k_size; kj++) 
            {
                
                int kernel_idx = f * (in_d * k_size * k_size) + d * (k_size * k_size) + ki * k_size + kj;
                double weight = kernel[kernel_idx];

                int row_in = i + ki;
                int col_in = j + kj;
                int input_idx = b * (in_d * in_h * in_w) + d * (in_h * in_w) + row_in * in_w + col_in;
                atomicAdd(&d_input[input_idx], d_val * weight);
            }
        }
    }
}

extern "C" void launch_conv2d_backward_lean(const double* d_input, const double* d_delta, const double* d_kernel, double* d_dk, double* d_db, double* d_prev_delta,int batch_size, int in_h, int in_w, int in_d,int out_h, int out_w, int num_filters, int k_size)
{
    int delta_size = batch_size * num_filters * out_h * out_w;
    int kernel_size = num_filters * in_d * k_size * k_size;
    int bias_size = num_filters;
    int input_size = batch_size * in_d * in_h * in_w;

    cudaMemset(d_dk, 0, kernel_size * sizeof(double));
    cudaMemset(d_db, 0, bias_size * sizeof(double));
    cudaMemset(d_prev_delta, 0, input_size * sizeof(double));

    int threads = 256;
    int blocks = (delta_size + threads - 1) / threads;

    conv2d_backward_weights_kernel<<<blocks, threads>>>(d_input, d_delta, d_dk, d_db, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
    conv2d_backward_input_kernel<<<blocks, threads>>>(d_delta, d_kernel, d_prev_delta, batch_size, in_h, in_w, in_d, out_h, out_w, num_filters, k_size);
}

extern "C" void gpu_alloc(double** ptr, size_t size) { cudaMalloc(ptr, size); }
extern "C" void gpu_free(double* ptr) { cudaFree(ptr); }
extern "C" void gpu_memcpy_h2d(double* dest, const double* src, size_t size) { cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice); }
extern "C" void gpu_memcpy_d2h(double* dest, const double* src, size_t size) { cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost); }