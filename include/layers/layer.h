#ifndef LAYER_H
#define LAYER_H

#include "../core/matrix.h"
#include <fstream>

class Layer
{
    public:
        virtual ~Layer() = default;
        virtual Matrix forward_pass(const Matrix& input)=0;
        virtual Matrix backward_pass(const Matrix& output,double learning_rate)=0;
        virtual void save(std::ofstream& file){};
        virtual void load(std::ifstream& file){};
    protected:
        Matrix input;
};

#endif