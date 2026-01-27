#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include "../include/core/matrix.h"
#include "../include/core/utils.h"
#include "../include/layers/dense.h"
#include "../include/layers/softmax.h"
#include "../include/layers/conv2d.h"
#include "../include/layers/pooling.h"
#include "../include/layers/batchnorm.h"
#include "../include/activation.h"
#include "../include/network.h"
#include "../include/io/data.h"

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

    nn.add(new Dense(512, 128));
    nn.add(new BatchNorm(128));
    nn.add(new Activation(leaky_relu, dleaky_relu));

    nn.add(new Dense(128, 47));
    nn.add(new Softmax());

    int epochs=10;
    int batch_size=32;
    double learning_rate=0.001;
    std::cout << "Starting CNN Training..." << std::endl;

    for(int epoch=1;epoch<=epochs;epoch++)
    {
        if(epoch == 6) 
        {
            std::cout << "[!] Scheduler: Dropping Learning Rate to 0.0001" << std::endl;
            learning_rate = 0.0001;
        }
        for(int i=0;i<X_train.rows;i+=batch_size)
        {
            int end=std::min(i+batch_size,X_train.rows);
            Matrix X_batch=X_train.slice(i,end);
            Matrix Y_batch=Y_train.slice(i,end);
            nn.fit(X_batch,Y_batch,1,learning_rate);
            if((i / batch_size) % 100 == 0) std::cout << ".";
        }
        std::cout<<std::endl;
        std::cout << "Epoch"<<epoch<<"-Accuracy: " << get_accuracy(nn, X_test, Y_test) << "%" << std::endl;
    }
    std::cout << "Evaluating on Test Set..." << std::endl;
    int test_correct = 0;
    int test_total = X_test.rows;
    int test_batch_size = 1000;

    for(int i = 0; i < test_total; i += test_batch_size)
    {
        int end = std::min(i + test_batch_size, test_total);
        Matrix X_test_batch = X_test.slice(i, end);
        Matrix Y_test_batch = Y_test.slice(i, end);
        Matrix preds = nn.predict(X_test_batch);
        for(int k = 0; k < preds.rows; k++)if(argmax(preds, k)==argmax(Y_test_batch, k))test_correct++;
    }
    double final_acc = (double)test_correct / test_total * 100.0;
    std::cout << "Test Accuracy: " << final_acc << "%" << std::endl;
    std::cout << "Saving model..." << std::endl;
    nn.save("emnist_model.bin");
    return 0;
}

