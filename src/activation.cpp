#include "../include/activation.h"
#include <iostream>

Activation::Activation(Activate f, Activate df):f(f),df(df){}

Matrix Activation::forward_pass(const Matrix& input) 
{
    this->input=input;
    Matrix output;
    output=input.apply(f);
    return input.apply(f);
}

Matrix Activation::backward_pass(const Matrix& delta, double learning_rate) 
{
    return delta.Hadamard(input.apply(df));
}
