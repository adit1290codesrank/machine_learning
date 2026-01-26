#ifndef DENSE_H
#define DENSE_H

#include "layer.h"
#include "../core/matrix.h"

class Dense:public Layer
{
    public:
        Dense(int input_size,int output_size);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;
        Matrix w;
        Matrix b;

    private:
        Matrix mw,vw;
        Matrix mb,vb;
        int t;

        void init();
};

#endif