#ifndef RECURRENT_H
#define RECURRENT_H

#include <vector>
#include "../core/matrix.h"
#include <cmath>

class Recurrent
{
    public:
        int input_size;
        int hidden_size;

        Matrix Wax,Waa,ba;
        Matrix dWax,dWaa,dba;
        Matrix mWax,vWax,mWaa,vWaa,mba,vba;
        int t;

        std::vector<Matrix> a_cache; 
        std::vector<Matrix> x_cache;

        Recurrent(int input_size,int hidden_size);
        std::vector<Matrix> forward_pass(const std::vector<Matrix>& input);
        std::vector<Matrix> backward_pass(const std::vector<Matrix>& delta);

        void update(double learning_rate);
};

#endif