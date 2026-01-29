#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include "../include/network.h"
#include "../include/core/matrix.h"
#include "../include/layers/dense.h"
#include "../include/layers/conv2d.h"
#include "../include/layers/pooling.h"
#include "../include/layers/batchnorm.h"
#include "../include/layers/dropout.h"
#include "../include/layers/softmax.h"
#include "../include/activation.h"
#include "../include/core/utils.h"

char get_emnist_char(int index) 
{
    const std::string mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
    if (index >= 0 && index < 47) return mapping[index];
    return '?';
}

int argmax(const Matrix& m) 
{
    double max_val = -1e9; 
    int max_idx = 0;
    for(int i=0; i < m.cols; i++) 
    {
        if(m(0, i) > max_val) 
        {
            max_val = m(0, i);
            max_idx = i;
        }
    }
    return max_idx;
}

int main()
{
    Network nn;
    nn.add(new Conv2D(28,28,1,32,3)); 
    nn.add(new BatchNorm(26*26*32));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Pooling(26,26,32,2,2));

    nn.add(new Conv2D(13,13,32,64,3)); 
    nn.add(new BatchNorm(11*11*64));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Pooling(11,11,64,2,2));

    nn.add(new Dense(1600, 512));
    nn.add(new BatchNorm(512));
    nn.add(new Activation(leaky_relu, dleaky_relu));

    nn.add(new Dropout(0.5));

    nn.add(new Dense(512, 128));
    nn.add(new BatchNorm(128));
    nn.add(new Activation(leaky_relu, dleaky_relu));

    nn.add(new Dropout(0.25));

    nn.add(new Dense(128, 47));
    nn.add(new Softmax());

    std::cerr << " [C++] Loading Model Weights" << std::endl;
    nn.load("emnist_model.bin");
    std::cerr << " [C++] Model Ready! Listening for input" << std::endl;

    std::string line;
    while(std::getline(std::cin,line))
    {
        if(line=="exit") break;
        if(line.empty()) continue;

        Matrix input(1,784);
        std::stringstream ss(line);
        for(int i=0;i<784;i++) ss >> input(0,i);
        
        std::cerr << "\n[C++] Model Input View:" << std::endl;
        for(int r = 0; r < 28; r++) {
            for(int c = 0; c < 28; c++) {
                double pixel = input(0, r * 28 + c);
                if(pixel > 0.5) std::cerr << "# ";
                else if(pixel > 0.2) std::cerr << ". ";
                else std::cerr << "  ";
            }
            std::cerr << "\n";
        }
        std::cerr << "------------------------" << std::endl;

        Matrix output=nn.predict(input);
        int prediction=argmax(output);

        std::cout<<get_emnist_char(prediction)<<std::endl;
    }
    return 0;
}

