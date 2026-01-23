#include <iostream>
#include <vector>
#include "../include/matrix.h"
#include "../include/utils.h"
#include "../include/layer.h"
#include "../include/dense.h"
#include "../include/activation.h"
#include "../include/network.h"

int main()
{
    Matrix X(4,2);
    X(0,0) = 0.0; X(0,1) = 0.0;
    X(1,0) = 0.0; X(1,1) = 1.0;
    X(2,0) = 1.0; X(2,1) = 0.0;
    X(3,0) = 1.0; X(3,1) = 1.0;
    Matrix y(4,1);
    y(0,0) = 0.0;
    y(1,0) = 1.0;
    y(2,0) = 1.0;
    y(3,0) = 0.0;
    
    Network nn;
    nn.add(new Dense(2,3));
    nn.add(new Activation(sigmoid,dsigmoid));
    nn.add(new Dense(3,1));
    nn.add(new Activation(sigmoid,dsigmoid));
    std::cout << "Network Built." << std::endl;
    nn.fit(X,y,10000,0.1);
    Matrix pred = nn.predict(X);
    std::cout << "Input: 0, 0 | Target: 0 | Pred: " << pred(0,0) << std::endl;
    std::cout << "Input: 0, 1 | Target: 1 | Pred: " << pred(1,0) << std::endl;
    std::cout << "Input: 1, 0 | Target: 1 | Pred: " << pred(2,0) << std::endl;
    std::cout << "Input: 1, 1 | Target: 0 | Pred: " << pred(3,0) << std::endl;
    return 0;
}