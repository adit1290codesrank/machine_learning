#include "../include/layers/softmax.h"
#include <iostream>

Softmax::Softmax(){}

Matrix Softmax::forward_pass(const Matrix& input)
{
    this->input=input;
    Matrix output(input.rows,input.cols);
    for(int i=0;i<input.rows;i++)
    {
        double max_=-1e9;
        for(int j=0;j<input.cols;j++) if(input(i,j) > max_) max_=input(i,j);
        double sum=0.0;
        for(int j=0;j<input.cols;j++) 
        {
            output(i,j)=std::exp(input(i,j)-max_);
            sum+=output(i,j);
        }
        for(int j=0;j<input.cols;j++) output(i,j)/=sum;
    }
    return output;
}

Matrix Softmax::backward_pass(const Matrix& delta, double learning_rate)
{
    return delta;
}