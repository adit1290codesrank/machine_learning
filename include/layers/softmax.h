#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "./layer.h"
#include "../core/matrix.h"
#include <cmath>

class Softmax:public Layer
{
    public:
        Softmax();
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;
};

#endif