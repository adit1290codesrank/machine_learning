#include "../include/layers/recurrent.h"
#include "../include/core/utils.h"
#include <iostream>>

Recurrent::Recurrent(int input_size,int hidden_size):input_size(input_size), hidden_size(hidden_size)
{
    //Xavier init
    double scale=sqrt(1.0/hidden_size);
    Wax=Matrix::random(hidden_size,input_size,-scale,scale);
    Waa=Matrix::random(hidden_size,hidden_size,-scale,scale);
    ba =Matrix::zeros(hidden_size,1);

    dWax=Matrix::zeros(hidden_size,input_size);
    dWaa=Matrix::zeros(hidden_size,hidden_size);
    dba =Matrix::zeros(hidden_size,1);

    mWax = Matrix::zeros(hidden_size,input_size);vWax = Matrix::zeros(hidden_size,input_size);
    mWaa = Matrix::zeros(hidden_size,hidden_size);vWaa = Matrix::zeros(hidden_size,hidden_size);
    mba  = Matrix::zeros(hidden_size,1);vba  = Matrix::zeros(hidden_size,1);
    t = 0;
}

std::vector<Matrix> Recurrent::forward_pass(const std::vector<Matrix>& input)
{
    x_cache=input;
    a_cache.clear();
    std::vector<Matrix> output;

    Matrix a_t_1=Matrix::zeros(hidden_size,1);
    for(const auto& x_t:input)
    {
        Matrix z_t=(Wax*x_t)+(Waa*a_t_1)+ba;
        Matrix a_t=z_t.apply(tanh_);
        output.push_back(a_t);
        a_cache.push_back(a_t);
        a_t_1=a_t;
    }
    return output;
}

std::vector<Matrix> Recurrent::backward_pass(const std::vector<Matrix>& delta)
{
    int time=delta.size();
    dWax=Matrix::zeros(hidden_size,input_size);
    dWaa=Matrix::zeros(hidden_size,hidden_size);
    dba=Matrix::zeros(hidden_size,1);

    Matrix delta_t=Matrix::zeros(hidden_size,1);
    std::vector<Matrix> prev_delta(time);

    for(int t=time-1;t>=0;t--)
    {  
        Matrix da=delta[t]+delta_t;
        Matrix dz=da.Hadamard(a_cache[t].apply(dtanh));
        dWax=dWax+(dz*x_cache[t].transpose());
        if(t!=0) dWaa=dWaa+(dz*a_cache[t-1].transpose());
        dba=dba+dz;
        delta_t=Waa.transpose()*dz;
        prev_delta[t]=Wax.transpose()*dz;
    }
    return prev_delta;
}

void Recurrent::update(double learning_rate)
{
    t++;
    auto adam=[&](Matrix& w,Matrix& dw,Matrix& m,Matrix& v)
    {
        double b1=0.9,b2=0.999,e=1e-8;
        m=(m*b1)+(dw*(1.0-b1));
        v=(v*b2)+(dw.Hadamard(dw*(1-b2)));

        double m_=1/(1.0-std::pow(b1,t)),v_=1/(1.0-std::pow(b2,t));
        Matrix m_hat=m*m_;
        Matrix v_hat=v*v_;

        for(int i=0;i<w.rows;i++)for(int j=0;j<w.cols;j++)w(i,j)-=learning_rate*m_hat(i,j)/(std::sqrt(v_hat(i,j))+e);

    };
    adam(Wax,dWax,mWax,vWax);
    adam(Waa,dWaa,mWaa,vWaa);
    adam(ba,dba,mba,vba);
}
