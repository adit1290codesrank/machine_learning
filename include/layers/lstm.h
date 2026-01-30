#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include "../core/matrix.h"

class LSTM
{
    public:
        int input_size;
        int hidden_size;

        Matrix Wfx,Wfa,bf;
        Matrix Wux,Wua,bu;
        Matrix Wcx,Wca,bc;
        Matrix Wox,Woa,bo;

        Matrix dWfx,dWfa,dbf;
        Matrix dWux,dWua,dbu;
        Matrix dWcx,dWca,dbc;
        Matrix dWox,dWoa,dbo;

        Matrix mWfx, vWfx; Matrix mWfh, vWfh; Matrix mbf, vbf;
        Matrix mWix, vWix; Matrix mWih, vWih; Matrix mbi, vbi;
        Matrix mWcx, vWcx; Matrix mWch, vWch; Matrix mbc, vbc;
        Matrix mWox, vWox; Matrix mWoh, vWoh; Matrix mbo, vbo;

    int t;

        struct Cache
        {
            Matrix f,u,c_,o;
            Matrix c,a;
        };
        std::vector<Cache> cache; 
        std::vector<Matrix> x_cache;

        LSTM(int input_size,int hidden_size);

        std::vector<Matrix> forward_pass(const std::vector<Matrix>& input);
        std::vector<Matrix> backward_pass(const std::vector<Matrix>& delta);
        void update(double learning_rate);
};

#endif