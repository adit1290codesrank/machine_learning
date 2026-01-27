#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "./layers/layer.h"
#include "./core/matrix.h"

class Network
{
    public:
        ~Network();
        void add(Layer* layer);
        Matrix predict(const Matrix& input);
        void fit(const Matrix& X,const Matrix& y,int epochs,double learning_rate);
        void save(const std::string& filename);
        void load(const std::string& filename);
        
    private:
        std::vector<Layer*> layers;
};

#endif