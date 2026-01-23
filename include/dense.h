#ifndef DENSE_H
#define DENSE_H

#include "layer.h"
#include "matrix.h"

class Dense:public Layer
{
    public:
        Dense(int input_size,int output_size);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;

    private:
        Matrix w;
        Matrix b;
        Matrix input;
};

#endif