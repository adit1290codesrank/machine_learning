#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"
#include "../core/matrix.h"

class Dropout : public Layer {
public:
    double x;     
    Matrix mask;  

    Dropout(double x);
    
    Matrix forward_pass(const Matrix& input) override;
    Matrix backward_pass(const Matrix& delta, double learning_rate) override;

    void save(std::ofstream& file) override {}
    void load(std::ifstream& file) override {}
};

#endif