#include <iostream>
#include <vector>
#include "../include/core/matrix.h"
#include "../include/core/utils.h"
#include "../include/layers/layer.h"
#include "../include/layers/dense.h"
#include "../include/activation.h"
#include "../include/network.h"
#include "../include/io/data.h"
#include "../include/layers/softmax.h"
#include <algorithm>
#include <chrono>

char get_emnist_char(int index) 
{
    const std::string mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
    if (index >= 0 && index < 47) return mapping[index];
    return '?';
}

int argmax(const Matrix& m, int row) 
{
    double max_val = -1e9; 
    int max_idx = 0;
    for(int i=0; i < m.cols; i++) 
    {
        if(m(row, i) > max_val) 
        {
            max_val = m(row, i);
            max_idx = i;
        }
    }
    return max_idx;
}

double get_accuracy(Network& nn, Matrix& X, Matrix& Y) 
{
    std::cout << "Calculating predictions..." << std::endl;
    
    int correct = 0;
    int batch_size = 1024;
    for(int i=0; i < X.rows; i += batch_size) 
    {
        int end = std::min(i + batch_size, X.rows);
        Matrix X_batch = X.slice(i, end);
        Matrix Y_batch = Y.slice(i, end);
        Matrix predictions = nn.predict(X_batch);
        for(int j=0; j < predictions.rows; j++) 
        {
            int pred_idx = argmax(predictions, j);
            int true_idx = argmax(Y_batch, j);
            if(pred_idx == true_idx) correct++;
        }
    }
    return (double)correct / X.rows * 100.0;
}

int main()
{
    std::srand(std::time(0));
    std::cout << "Loading Train Set..." << std::endl;
    Matrix X_train = DataLoader::load_images("./data/emnist-balanced-train-images-idx3-ubyte");
    Matrix Y_train = DataLoader::load_labels("./data/emnist-balanced-train-labels-idx1-ubyte");
    std::cout << "Loading Test Set..." << std::endl;
    Matrix X_test = DataLoader::load_images("./data/emnist-balanced-test-images-idx3-ubyte");
    Matrix Y_test = DataLoader::load_labels("./data/emnist-balanced-test-labels-idx1-ubyte");
    Network nn;
    nn.add(new Dense(784,512));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Dense(512,256));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Dense(256,128));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Dense(128,64));
    nn.add(new Activation(leaky_relu,dleaky_relu));
    nn.add(new Dense(64,47));
    nn.add(new Softmax());
    
    int epochs = 5;
    int batch_size = 64; 
    double learning_rate = 0.001;

    std::cout << "Starting Training (" << epochs << " epochs, Batch Size: " << batch_size << ")..." << std::endl;
    for(int epoch=1;epoch<=epochs;++epoch)
    {
        std::cout << "Epoch " << epoch << "/" << epochs << " ";
        for(int i=0;i<X_train.rows;i+=batch_size)
        {
            int end = std::min(i+batch_size,X_train.rows);
            Matrix X_batch = X_train.slice(i,end);
            Matrix Y_batch = Y_train.slice(i,end);
            nn.fit(X_batch,Y_batch,1,learning_rate);
            if ((i / batch_size) % 100 == 0) std::cout << ".";
        }
        std::cout << std::endl;
        double acc = get_accuracy(nn, X_test, Y_test);
        std::cout << "  - Test Accuracy: " << acc << "%" << std::endl;
    }
    double train_acc = get_accuracy(nn, X_train, Y_train);
    std::cout << "Training Accuracy: " << train_acc << "%" << std::endl;

    // Check Test Accuracy (Can it generalize?)
    double test_acc = get_accuracy(nn, X_test, Y_test);
    std::cout << "Test Accuracy:     " << test_acc << "%" << std::endl;
    return 0;

}

