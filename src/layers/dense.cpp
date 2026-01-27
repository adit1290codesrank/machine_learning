#include "../include/layers/dense.h"
#include <iostream>
#include <cmath>

    Dense::Dense(int input_size,int output_size):mw(input_size,output_size),vw(input_size,output_size),mb(1,output_size),vb(1,output_size),t(0)
    {
        w = Matrix(input_size, output_size);
        b = Matrix(1, output_size);
        double scale = std::sqrt(2.0 / input_size);
        for (int i = 0; i < input_size; i++) 
        {
            for (int j = 0; j < output_size; j++) 
            {
                double r = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
                w(i, j) = r * scale; 
            }
        }

        for (int j = 0; j < output_size; j++) b(0, j) = 0.0;
        init();
    }

    void Dense::init()
    {
        mw = Matrix::zeros(mw.rows,mw.cols);
        vw = Matrix::zeros(vw.rows,vw.cols);
        mb = Matrix::zeros(mb.rows,mb.cols);
        vb = Matrix::zeros(vb.rows,vb.cols);
        t=0;
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
        Matrix delta_prev = delta*w.transpose();
        //Adam optimizer
        t++;
        m=1.0-std::pow(b1,t);v=1.0-std::pow(b2,t);
        for(int i=0;i<w.rows;i++)
        {
            for(int j=0;j<w.cols;j++)
            {
                mw(i,j)=b1*mw(i,j)+(1-b1)*dw(i,j);
                vw(i,j)=b2*vw(i,j)+(1-b2)*dw(i,j)*dw(i,j);
                w(i,j)-=learning_rate*(mw(i,j)/m)/(std::sqrt(vw(i,j)/v)+e);
            }
        }

        for(int i=0;i<b.cols;i++)
        {
            mb(0,i)=b1*mb(0,i)+(1-b1)*db(0,i);
            vb(0,i)=b2*vb(0,i)+(1-b2)*db(0,i)*db(0,i);
            b(0,i)-=learning_rate*(mb(0,i)/m)/(std::sqrt(vb(0,i)/v)+e);
        }
        return delta_prev;
    }

    void Dense::save(std::ofstream& file) 
    {
        w.save(file);
        b.save(file);
    }

    void Dense::load(std::ifstream& file) 
    {
        w.load(file);
        b.load(file);
    }

