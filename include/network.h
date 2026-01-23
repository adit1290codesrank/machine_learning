#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "layer.h"
#include "matrix.h"

class Network
{
    public:
        ~Network();
        void add(Layer* layer);
        Matrix predict(const Matrix& input);
        void fit(const Matrix& X,const Matrix& y,int epochs,double learning_rate);

    private:
        std::vector<Layer*> layers;
};

#endif