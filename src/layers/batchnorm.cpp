#include "../../include/layers/batchnorm.h"
#include <iostream>
#include <cmath>

BatchNorm::BatchNorm(int features):features(features)
{
    g=Matrix(1,features);
    for(int i=0;i<features;i++) g(0,i)=1.0;
    b=Matrix::zeros(1,features);
    mean = Matrix::zeros(1,features);
    var = Matrix::zeros(1,features);
    mg = Matrix::zeros(1,features);
    vg = Matrix::zeros(1,features);
    mb = Matrix::zeros(1,features);
    vb = Matrix::zeros(1,features);
}

Matrix BatchNorm::forward_pass(const Matrix& input)
{
    Matrix output(input.rows,features);
    x_=Matrix(input.rows,features);
    std_inv=Matrix(1,features);
    if(this->is_training)
    {
         Matrix mean_=Matrix::zeros(1,features),var_=Matrix::zeros(1,features);
        for(int i=0;i<input.rows;i++)for(int j=0;j<features;j++)mean_(0,j)+=input(i,j);
        for(int i=0;i<features;i++)mean_(0,i)/=input.rows;
        for(int i=0;i<input.rows;i++)
        {
            for(int j=0;j<features;j++)
            {
                x_(i,j)=input(i,j)-mean_(0,j);
                var_(0,j)+=x_(i,j)*x_(i,j);
            }
        }
        for(int i=0;i<features;i++) var_(0,i)/=input.rows;
        for(int i=0;i<features;i++)
        {
            mean(0,i)=momentum*mean(0,i)+(1.0-momentum)*mean_(0,i);
            var(0,i)=momentum*var(0,i)+(1.0-momentum)*var_(0,i);
            std_inv(0,i)=1.0/std::sqrt(var_(0,i)+e);
        }
    }
    else
    {
        for(int i=0; i<input.rows; i++)for(int j=0; j<features; j++) x_(i, j) = input(i, j) - mean(0, j);
        for(int i=0; i<features; i++) std_inv(0, i) = 1.0 / std::sqrt(var(0, i) + e);
    }
    #pragma omp parallel for
    for(int i=0;i<input.rows;i++)for(int j=0;j<features;j++)output(i,j)=g(0,j)*x_(i,j)*std_inv(0,j)+b(0,j);
    return output;
}

Matrix BatchNorm::backward_pass(const Matrix& delta,double learning_rate)
{
    Matrix prev_delta(delta.rows,features);
    Matrix dg=Matrix::zeros(1,features),db=Matrix::zeros(1,features);
    
    for(int i=0;i<delta.rows;i++)
    {
        for(int j=0;j<features;j++)
        {
            db(0,j)+=delta(i,j);
            dg(0,j)+=delta(i,j)*x_(i,j)*std_inv(0,j);
        }
    }

    #pragma omp parallel for
    for(int i=0;i<delta.rows;i++)for(int j=0;j<features;j++)prev_delta(i,j)=(g(0,j)*std_inv(0,j)/delta.rows)*(delta.rows*delta(i,j)-db(0,j)-x_(i,j)*std_inv(0,j)*dg(0,j));
    t++;
    m=1.0-std::pow(b1,t);v=1.0-std::pow(b2,t);
    for(int i=0;i<features;i++)
    {
        mb(0,i)=b1*mb(0,i)+(1.0-b1)*db(0,i);
        vb(0,i)=b2*vb(0,i)+(1.0-b2)*db(0,i)*db(0,i);
        b(0,i)-=learning_rate*(mb(0,i)/m)/(std::sqrt(vb(0,i)/v)+e);
        mg(0,i)=b1*mg(0,i)+(1.0-b1)*dg(0,i);
        vg(0,i)=b2*vg(0,i)+(1.0-b2)*dg(0,i)*dg(0,i);
        g(0,i)-=learning_rate*(mg(0,i)/m)/(std::sqrt(vg(0,i)/v)+e);
    }
    return prev_delta;
}

void BatchNorm::save(std::ofstream& file) 
{
    g.save(file);
    b.save(file);
    mean.save(file);
    var.save(file);
}

void BatchNorm::load(std::ifstream& file) 
{
    g.load(file);
    b.load(file);
    mean.load(file);
    var.load(file);
}