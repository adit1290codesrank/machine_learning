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

extern "C" void launch_conv2d(const double* h_input, const double* h_kernel, double* h_output,int batch_size, int in_h, int in_w, int in_d,int out_h, int out_w, int num_filters, int k_size)
{
    double *d_input, *d_kernel, *d_output;
    int input_size = batch_size * in_d * in_h * in_w;
    int kernel_size_total = num_filters * in_d * k_size * k_size;
    int output_size = batch_size * num_filters * out_h * out_w;

    check_cuda(cudaMalloc((void**)&d_input, input_size * sizeof(double)), "Malloc Conv Input");
    check_cuda(cudaMalloc((void**)&d_kernel, kernel_size_total * sizeof(double)), "Malloc Conv Kernel");
    check_cuda(cudaMalloc((void**)&d_output, output_size * sizeof(double)), "Malloc Conv Output");

    check_cuda(cudaMemcpy(d_input, h_input, input_size * sizeof(double), cudaMemcpyHostToDevice), "Copy Input");
    check_cuda(cudaMemcpy(d_kernel, h_kernel, kernel_size_total * sizeof(double), cudaMemcpyHostToDevice), "Copy Kernel");

    int total_threads = output_size;
    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2dkernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_kernel, d_output,batch_size, in_h, in_w, in_d,out_h, out_w, num_filters, k_size);
    
    check_cuda(cudaGetLastError(), "Conv2D Kernel Launch");
    check_cuda(cudaDeviceSynchronize(), "Conv2D Kernel Sync");

    check_cuda(cudaMemcpy(h_output, d_output, output_size * sizeof(double), cudaMemcpyDeviceToHost), "Copy Output");
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}