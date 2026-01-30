#include "../include/layers/lstm.h"
#include "../include/core/utils.h"

LSTM::LSTM(int input_size,int hidden_size):input_size(input_size),hidden_size(hidden_size)
{
    //Xavier init
    double s=std::sqrt(1.0/hidden_size);

    Wfx=Matrix::random(hidden_size,input_size,-s,s);
    Wfa=Matrix::random(hidden_size,hidden_size,-s,s);
    bf=Matrix::ones(hidden_size,1);

    Wux=Matrix::random(hidden_size,input_size,-s,s);
    Wua=Matrix::random(hidden_size,hidden_size,-s,s);
    bu=Matrix::zeros(hidden_size,1);

    Wcx=Matrix::random(hidden_size,input_size,-s,s);
    Wca=Matrix::random(hidden_size,hidden_size,-s,s);
    bc=Matrix::zeros(hidden_size,1);

    Wox=Matrix::random(hidden_size,input_size,-s,s);
    Woa=Matrix::random(hidden_size,hidden_size,-s,s);
    bo=Matrix::zeros(hidden_size,1);

    dWfx=Matrix::zeros(hidden_size,input_size); dWfa=Matrix::zeros(hidden_size,hidden_size); dbf=Matrix::zeros(hidden_size,1);
    dWux=Matrix::zeros(hidden_size,input_size); dWua=Matrix::zeros(hidden_size,hidden_size); dbu=Matrix::zeros(hidden_size,1);
    dWcx=Matrix::zeros(hidden_size,input_size); dWca=Matrix::zeros(hidden_size,hidden_size); dbc=Matrix::zeros(hidden_size,1);
    dWox=Matrix::zeros(hidden_size,input_size); dWoa=Matrix::zeros(hidden_size,hidden_size); dbo=Matrix::zeros(hidden_size,1);

    mWfx=Matrix::zeros(hidden_size,input_size); vWfx=Matrix::zeros(hidden_size,input_size);
    mWfh=Matrix::zeros(hidden_size,hidden_size); vWfh=Matrix::zeros(hidden_size,hidden_size);
    mbf=Matrix::zeros(hidden_size,1); vbf=Matrix::zeros(hidden_size,1);

    mWix=Matrix::zeros(hidden_size,input_size); vWix=Matrix::zeros(hidden_size,input_size);
    mWih=Matrix::zeros(hidden_size,hidden_size); vWih=Matrix::zeros(hidden_size,hidden_size);
    mbi=Matrix::zeros(hidden_size,1); vbi=Matrix::zeros(hidden_size,1);

    mWcx=Matrix::zeros(hidden_size,input_size); vWcx=Matrix::zeros(hidden_size,input_size);
    mWch=Matrix::zeros(hidden_size,hidden_size); vWch=Matrix::zeros(hidden_size,hidden_size);
    mbc=Matrix::zeros(hidden_size,1); vbc=Matrix::zeros(hidden_size,1);

    mWox=Matrix::zeros(hidden_size,input_size); vWox=Matrix::zeros(hidden_size,input_size);
    mWoh=Matrix::zeros(hidden_size,hidden_size); vWoh=Matrix::zeros(hidden_size,hidden_size);
    mbo=Matrix::zeros(hidden_size,1); vbo=Matrix::zeros(hidden_size,1);

    t=0;
}

std::vector<Matrix> LSTM::forward_pass(const std::vector<Matrix>& input)
{
    x_cache=input;
    cache.clear();
    std::vector<Matrix> outputs;

    Matrix a_prev=Matrix::zeros(hidden_size,1);
    Matrix c_prev=Matrix::zeros(hidden_size,1);

    for(const auto& x:input)
    {
        Cache s;
        s.f=((Wfx*x)+(Wfa*a_prev)+bf).apply(sigmoid);
        s.u=((Wux*x)+(Wua*a_prev)+bu).apply(sigmoid);
        s.c_=((Wcx*x)+(Wca*a_prev)+bc).apply(tanh_);
        s.o=((Wox*x)+(Woa*a_prev)+bo).apply(sigmoid);

        s.c=s.f.Hadamard(c_prev)+s.u.Hadamard(s.c_);
        
        Matrix tanh_c=s.c.apply(std::tanh);
        s.a=s.o.Hadamard(tanh_c);

        outputs.push_back(s.a);
        cache.push_back(s);

        a_prev=s.a;
        c_prev=s.c;
    }
    return outputs;
}

std::vector<Matrix> LSTM::backward_pass(const std::vector<Matrix>& delta)
{
    int steps=delta.size();

    dWfx=Matrix::zeros(hidden_size,input_size); dWfa=Matrix::zeros(hidden_size,hidden_size); dbf=Matrix::zeros(hidden_size,1);
    dWux=Matrix::zeros(hidden_size,input_size); dWua=Matrix::zeros(hidden_size,hidden_size); dbu=Matrix::zeros(hidden_size,1);
    dWcx=Matrix::zeros(hidden_size,input_size); dWca=Matrix::zeros(hidden_size,hidden_size); dbc=Matrix::zeros(hidden_size,1);
    dWox=Matrix::zeros(hidden_size,input_size); dWoa=Matrix::zeros(hidden_size,hidden_size); dbo=Matrix::zeros(hidden_size,1);

    Matrix da_next=Matrix::zeros(hidden_size,1);
    Matrix dc_next=Matrix::zeros(hidden_size,1);
    std::vector<Matrix> dx(steps);

    for(int i=steps-1;i>=0;--i)
    {
        Cache s=cache[i];
        Matrix c_prev=(i>0)?cache[i-1].c:Matrix::zeros(hidden_size,1);
        Matrix x=x_cache[i];

        Matrix da=delta[i]+da_next;
        Matrix tanh_c=s.c.apply(std::tanh);

        Matrix do_=da.Hadamard(tanh_c).Hadamard(s.o.apply(dsigmoid));
        
        Matrix dc=da.Hadamard(s.o).Hadamard(tanh_c.apply(dtanh))+dc_next;
        
        Matrix dc_=dc.Hadamard(s.u).Hadamard(s.c_.apply(dtanh));
        Matrix du=dc.Hadamard(s.c_).Hadamard(s.u.apply(dsigmoid));
        Matrix df=dc.Hadamard(c_prev).Hadamard(s.f.apply(dsigmoid));

        dWfx=dWfx+(df*x.transpose());
        dWux=dWux+(du*x.transpose());
        dWcx=dWcx+(dc_*x.transpose());
        dWox=dWox+(do_*x.transpose());
        
        dbf=dbf+df; dbu=dbu+du; dbc=dbc+dc_; dbo=dbo+do_;

        if(i>0)
        {
            Matrix a_prev_T=cache[i-1].a.transpose();
            dWfa=dWfa+(df*a_prev_T);
            dWua=dWua+(du*a_prev_T);
            dWca=dWca+(dc_*a_prev_T);
            dWoa=dWoa+(do_*a_prev_T);
        }

        dc_next=dc.Hadamard(s.f);
        da_next=(Wfa.transpose()*df)+(Wua.transpose()*du)+(Wca.transpose()*dc_)+(Woa.transpose()*do_);
        
        dx[i]=(Wfx.transpose()*df)+(Wux.transpose()*du)+(Wcx.transpose()*dc_)+(Wox.transpose()*do_);
    }
    return dx;
}

void LSTM::update(double learning_rate)
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

    adam(Wfx,dWfx,mWfx,vWfx); adam(Wfa,dWfa,mWfh,vWfh); adam(bf,dbf,mbf,vbf);
    adam(Wux,dWux,mWix,vWix); adam(Wua,dWua,mWih,vWih); adam(bu,dbu,mbi,vbi);
    adam(Wcx,dWcx,mWcx,vWcx); adam(Wca,dWca,mWch,vWch); adam(bc,dbc,mbc,vbc);
    adam(Wox,dWox,mWox,vWox); adam(Woa,dWoa,mWoh,vWoh); adam(bo,dbo,mbo,vbo);
}
