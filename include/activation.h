#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"
#include "matrix.h"
#include <functional>

typedef double (*Activate)(double);

class Activation:public Layer
{
    public:
        Activation(Activate f, Activate df);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;

    private:
        Activate f;
        Activate df;
};

#endif