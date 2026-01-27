#include "../include/network.h"
#include "../include/core/utils.h"
#include <iostream>
#include <fstream>

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
    }
}

void Network::save(const std::string& filename) 
{
    std::ofstream file(filename,std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for saving." << std::endl;
        return;
    }
    for(Layer* layer : layers) layer->save(file);
    file.close();
    std::cout << "Model successfully saved to " << filename << std::endl;
}

void Network::load(const std::string& filename) 
{
    std::ifstream file(filename,std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for loading." << std::endl;
        return;
    }
    for(Layer* layer : layers)layer->load(file);
    file.close();
    std::cout << "Model successfully loaded from " << filename << std::endl;
}

