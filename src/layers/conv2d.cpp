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
    Matrix output(input.rows,f*oh*ow);
    std::vector<double> flat_kernels(f*d*k*k);

    for(int i = 0; i < f; i++) 
    {          
        for(int j = 0; j < d; j++) 
        {      
            Matrix& mat = kernels[i][j];
            int size = k * k;
            for(int m = 0; m < size; m++) flat_kernels.push_back(mat.data[m]);
        }
    }
    launch_conv2d(input.data, flat_kernels.data(), output.data,input.rows, h, w, d, oh, ow, f, k);
    #pragma omp parallel for
    for(int i = 0; i < input.rows; i++) for(int j = 0; j < f; j++) for(int pixel=0;pixel<oh*ow;pixel++) output(i,j*oh*ow)+=b[j];
    return output;
}

Matrix Conv2D::backward_pass(const Matrix& delta,double learning_rate)
{
    Matrix prev_delta=Matrix::zeros(input.rows,input.cols);
    std::vector<double> db(f, 0.0);
    std::vector<std::vector<Matrix>> dk(f, std::vector<Matrix>(d));
    for(int i=0; i<f; i++)for(int j=0; j<d; j++) dk[i][j] = Matrix::zeros(k, k);
    for(int r=0;r<delta.rows;r++)
    {
        for(int filter=0;filter<f;filter++)
        {
            for(int i=0;i<oh;i++)
            {
                for(int j=0;j<ow;j++)
                {
                    double d_val = delta(r, filter*oh*ow + i*ow + j);
                    db[filter] += d_val;

                    for(int depth=0;depth<d;depth++)
                    {
                        for(int ker_i=0;ker_i<k;ker_i++)
                        {
                            for(int ker_j=0;ker_j<k;ker_j++)
                            {
                                int input_idx = depth*h*w + (i+ker_i)*w + (j+ker_j);
                                double val = input(r, input_idx);
                                
                                dk[filter][depth](ker_i,ker_j) += val * d_val;
                                prev_delta(r, input_idx) += d_val * kernels[filter][depth](ker_i,ker_j);
                            }
                        }                       
                    }
                }
            }
        }
    }
    t++;
    m=1.0-std::pow(b1,t);v=1.0-std::pow(b2,t);
    for(int j=0; j<f; j++) 
    {
        mb[j]=b1*mb[j]+(1-b1)*db[j];
        vb[j]=0.999*vb[j]+(0.001)*db[j]*db[j];

        b[j]-=learning_rate*(mb[j]/m)/((std::sqrt(vb[j]/v)+e));
        for(int depth=0; depth<d; depth++) 
        {
            for(int r=0; r<k; r++) 
            {
                for(int c=0; c<k; c++) 
                {
                    mk[j][depth](r,c)=b1*mk[j][depth](r,c)+(1.0-b1)*dk[j][depth](r,c);
                    vk[j][depth](r,c)=b2*vk[j][depth](r,c)+(1.0-b2)*dk[j][depth](r,c)*dk[j][depth](r,c);
                    kernels[j][depth](r, c)-=learning_rate*(mk[j][depth](r,c)/m)/(std::sqrt(vk[j][depth](r,c)/v)+e);
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