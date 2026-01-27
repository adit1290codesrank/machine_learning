#include "../../include/layers/zeropad.h"

ZeroPad::ZeroPad(int h,int w,int d,int pad):h(h),w(w),d(d),pad(pad) 
{
    oh=h+2*pad;
    ow=w+2*pad;
}

Matrix ZeroPad::forward_pass(const Matrix& input)
{
    Matrix output=Matrix::zeros(input.rows,d*oh*ow);

    #pragma omp parallel for
    for(int r=0;r<input.rows;r++)for(int depth=0;depth<d;depth++)for(int i=0;i<h;i++)for(int j=0;j<w;j++)output(r,depth*oh*ow + (i+pad)*ow+j+pad)=input(r,depth*h*w+i*w+j);
    return output;
}

Matrix ZeroPad::backward_pass(const Matrix& delta,double learning_rate)
{
    Matrix prev_delta(input.rows,d*h*w);
    
    #pragma omp parallel for
    for(int r=0;r<delta.rows;r++)for(int depth=0;depth<d;depth++)for(int i=0;i<h;i++)for(int j=0;j<w;j++)prev_delta(r,depth*oh*ow + (i+pad)*ow+j+pad)=input(r,depth*h*w+i*w+j);
    return prev_delta;
}
