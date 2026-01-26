#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer
{
    public:
        virtual ~Layer() = default;
        virtual Matrix forward_pass(const Matrix& input)=0;
        virtual Matrix backward_pass(const Matrix& output,double learning_rate)=0;
    
    protected:
        Matrix input;
};

#endif