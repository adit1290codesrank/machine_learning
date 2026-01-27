#include "../../include/layers/pooling.h"
#include <iostream>
#include <algorithm>

Pooling::Pooling(int h,int w,int d,int pool_size,int stride):h(h),w(w),d(d),pool_size(pool_size),stride(stride)
{
    oh=(h-pool_size)/stride+1;
    ow=(w-pool_size)/stride+1;
}

Matrix Pooling::forward_pass(const Matrix& input)
{   
    this->input = input;
    Matrix output(input.rows,oh*ow*d);
    max_cache.assign(input.rows,std::vector<int>(oh*ow*d));
    
    #pragma omp parallel for
    for(int r=0;r<input.rows;r++)
    {
        for(int depth=0;depth<d;depth++)
        {
            for(int i=0;i<oh;i++)
            {
                for(int j=0;j<ow;j++)
                {
                    double max_=-DBL_MAX;
                    int index=-1;
                    for(int p_i=0;p_i<pool_size;p_i++)
                    {
                        for(int p_j=0;p_j<pool_size;p_j++)
                        {
                            if(i*stride+p_i<h && j*stride+p_j<w)
                            {
                                if(input(r,depth*h*w+(i*stride+p_i)*w+(j*stride+p_j))>max_)
                                {
                                    max_=input(r,depth*h*w+(i*stride+p_i)*w+(j*stride+p_j));
                                    index=depth*h*w+(i*stride+p_i)*w+(j*stride+p_j);
                                }
                            }
                        }
                    }
                    output(r,depth*oh*ow+i*ow+j)=max_;
                    max_cache[r][depth*oh*ow+i*ow+j]=index;
                }
            }
        }
    }
    return output;
}

Matrix Pooling::backward_pass(const Matrix& delta,double learning_rate)
{
    Matrix prev_delta=Matrix::zeros(input.rows,d*h*w);
    #pragma omp parallel for
    for(int i=0;i<input.rows;i++)for(int j=0;j<delta.cols;j++)prev_delta(i,max_cache[i][j])+=delta(i,j);
    return prev_delta;
}
