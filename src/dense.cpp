#include "../include/dense.h"
#include <iostream>

    Dense::Dense(int input_size,int output_size)
    {
        w = Matrix::random(input_size,output_size);
        b = Matrix::zeros(1,output_size);
    }

    Matrix Dense::forward_pass(const Matrix& input)
    {
        this->input=input;
        Matrix output = input*w;
        for(int i=0; i < output.rows; i++) for(int j=0; j < output.cols; j++) output(i,j) += b(0,j);
        return output;
    }

    Matrix Dense::backward_pass(const Matrix& delta,double learning_rate)
    {
        Matrix dw = input.transpose()*delta;
        Matrix db = delta.sum_rows();
        Matrix prev_delta = delta*w.transpose();
        w = w - dw * learning_rate;
        b = b - db * learning_rate;
        return prev_delta;
    }

