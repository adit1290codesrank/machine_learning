#include "../include/network.h"
#include "../include/utils.h"
#include <iostream>

Network::~Network()
{
    for(auto layer:layers) delete layer;
}

void Network::add(Layer* layer)
{
    layers.push_back(layer);
}

Matrix Network::predict(const Matrix& input)
{
    Matrix output=input;
    for(auto layer:layers) output=layer->forward_pass(output);
    return output;
}

void Network::fit(const Matrix& X,const Matrix& y, int epochs,double learning_rate)
{
    int m=layers.size();
    for(int i=0;i<epochs;i++)
    {
        Matrix output=predict(X);
        Matrix delta=output-y;
        for(int j=m-1;j>=0;j--) delta=layers[j]->backward_pass(delta,learning_rate);
        //if(i%1000==0) std::cout<<"Epoch "<<i<<", Loss: "<<mse(y,output)<<std::endl;
    }
}

