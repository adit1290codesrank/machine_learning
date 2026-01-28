#include "../../include/layers/conv2d.h"
#include <iostream>
#include <random>
#include <fstream>
#include <omp.h>

Conv2D::Conv2D(int h,int w,int d,int f,int k):h(h),w(w),d(d),f(f),k(k)
{
    oh = h-k+1;
    ow = w-k+1;
    init();
}

void Conv2D::init()
{
    // He Init for ReLU
    std::default_random_engine re;
    double std = std::sqrt(2.0/(k*k*d));
    std::normal_distribution<double> dist(0.0,std);

    mk.resize(f);vk.resize(f);
    mb.resize(f,0.0);vb.resize(f,0.0);
    kernels.resize(f);
    b.resize(f);
    for(int i=0;i<f;i++)
    {
        kernels[i].resize(d);
        mk[i].resize(d);vk[i].resize(d);
        b[i]=0.0;
        for(int j=0;j<d;j++)
        {
            kernels[i][j]=Matrix(k,k);
            mk[i][j]=Matrix::zeros(k,k);
            vk[i][j]=Matrix::zeros(k,k);
            for(int k_=0;k_<k;k_++) for(int k__=0;k__<k;k__++) kernels[i][j](k_,k__)=dist(re);
        }
    }
}

Matrix Conv2D::forward_pass(const Matrix& input)
{
    this->input=input;
    allocate_gpu_memory(input.rows);
    Matrix output(input.rows,f*oh*ow);
    std::vector<double> flat_kernels;
    flat_kernels.reserve(f*d*k*k);
    for(int i = 0; i < f; i++) 
    {          
        for(int j = 0; j < d; j++) 
        {      
            Matrix& mat = kernels[i][j];
            int size = k * k;
            for(int m = 0; m < size; m++) flat_kernels.push_back(mat.data[m]);
        }
    }
    gpu_memcpy_h2d(d_kernels, flat_kernels.data(), flat_kernels.size() * sizeof(double));
    gpu_memcpy_h2d(d_input, input.data, input.rows * input.cols * sizeof(double));
    launch_conv2d_lean(d_input, d_kernels, d_output, input.rows, h, w, d, oh, ow, f, k);
    gpu_memcpy_d2h(output.data, d_output, output.rows * output.cols * sizeof(double));
    #pragma omp parallel for
    for(int i = 0; i < input.rows; i++) for(int j = 0; j < f; j++) for(int pixel=0;pixel<oh*ow;pixel++) output(i,j*oh*ow+pixel)+=b[j];
    return output;
}

Matrix Conv2D::backward_pass(const Matrix& delta, double learning_rate)
{
    Matrix prev_delta(input.rows, h*w*d); 
    
    std::vector<double> flat_kernels;
    flat_kernels.reserve(f*d*k*k);
    for(int i=0; i<f; i++) for(int j=0; j<d; j++) for(int m=0; m<k*k; m++) flat_kernels.push_back(kernels[i][j].data[m]);
    std::vector<double> flat_dk(f*d*k*k, 0.0);
    std::vector<double> flat_db(f, 0.0);

    gpu_memcpy_h2d(d_delta, delta.data, delta.rows * delta.cols * sizeof(double));

    launch_conv2d_backward_lean(d_input, d_delta, d_kernels, d_dk, d_db, d_prev_delta, input.rows, h, w, d, oh, ow, f, k);

    gpu_memcpy_d2h(flat_dk.data(), d_dk, flat_dk.size() * sizeof(double));
    gpu_memcpy_d2h(flat_db.data(), d_db, flat_db.size() * sizeof(double));
    gpu_memcpy_d2h(prev_delta.data, d_prev_delta, prev_delta.rows * prev_delta.cols * sizeof(double));

    t++;
    m = 1.0 - std::pow(b1, t);
    v = 1.0 - std::pow(b2, t);

    int dk_idx = 0;
    for(int j = 0; j < f; j++) 
    {
        mb[j] = b1 * mb[j] + (1 - b1) * flat_db[j];
        vb[j] = 0.999 * vb[j] + (0.001) * flat_db[j] * flat_db[j];
        b[j] -= learning_rate * (mb[j] / m) / ((std::sqrt(vb[j] / v) + e));

        for(int depth = 0; depth < d; depth++) 
        {
            for(int r = 0; r < k; r++) 
            {
                for(int c = 0; c < k; c++) 
                {
                    double grad = flat_dk[dk_idx++];
                    
                    mk[j][depth](r,c) = b1 * mk[j][depth](r,c) + (1.0 - b1) * grad;
                    vk[j][depth](r,c) = b2 * vk[j][depth](r,c) + (1.0 - b2) * grad * grad;
                    kernels[j][depth](r, c) -= learning_rate * (mk[j][depth](r,c) / m) / (std::sqrt(vk[j][depth](r,c) / v) + e);
                }
            }
        }
    }

    return prev_delta;
}

void Conv2D::save(std::ofstream& file) 
{
    for(int i=0; i<f; i++)for(int j=0; j<d; j++)kernels[i][j].save(file);
    file.write((char*)b.data(),f*sizeof(double));
}

void Conv2D::load(std::ifstream& file) 
{
    for(int i=0; i<f; i++)for(int j=0; j<d; j++)kernels[i][j].load(file);
    file.read((char*)b.data(),f*sizeof(double));
}

void Conv2D::allocate_gpu_memory(int batch_size) 
{
    if (d_kernels && batch_size <= this->allocated_batch_size) return;
    if (d_kernels && batch_size > this->allocated_batch_size)
    {
        gpu_free(d_input);
        gpu_free(d_output);
        gpu_free(d_delta);
        gpu_free(d_prev_delta);
    }
    if (!d_kernels) 
    {
        gpu_alloc(&d_kernels, f * d * k * k * sizeof(double));
        gpu_alloc(&d_db, f * sizeof(double));
        gpu_alloc(&d_dk, f * d * k * k * sizeof(double));
    }
    gpu_alloc(&d_input, batch_size * d * h * w * sizeof(double));
    gpu_alloc(&d_output, batch_size * f * oh * ow * sizeof(double));
    gpu_alloc(&d_delta, batch_size * f * oh * ow * sizeof(double));
    gpu_alloc(&d_prev_delta, batch_size * d * h * w * sizeof(double));
    this->allocated_batch_size = batch_size;
}

Conv2D::~Conv2D() 
{
    gpu_free(d_kernels); gpu_free(d_input); gpu_free(d_output);
    gpu_free(d_delta); gpu_free(d_dk); gpu_free(d_db); gpu_free(d_prev_delta);
}