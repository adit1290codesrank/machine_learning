#ifndef GMATRIX_H
#define GMATRIX_H

#include <cuda_runtime.h>
#include "matrix.h"
#include <iostream>

class GMatrix
{
    public:
        double* data;
        int rows,cols;
        
        GMatrix(int rows,int cols):rows(rows),cols(cols)
        {
            cudaError_t error = cudaMalloc(&data,(size_t)rows*(size_t)cols*sizeof(double));
            if(error!=cudaSuccess) std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(error) << std::endl;
        }

        ~GMatrix()
        {
            if(data) cudaFree(data);
        }

        void upload(const Matrix& host)
        {
            cudaMemcpy(data,host.data,(size_t)rows*(size_t)cols*sizeof(double),cudaMemcpyHostToDevice);
        }

        void download(Matrix& host)
        {
            cudaMemcpy(host.data,data,(size_t)rows*(size_t)cols*sizeof(double),cudaMemcpyDeviceToHost);
        }

        void clear()
        {
            cudaMemset(data,0,(size_t)rows*(size_t)cols*sizeof(double));
        }

        GMatrix(const GMatrix&)=delete;
        GMatrix& operator=(const GMatrix&)=delete;
};

#endif