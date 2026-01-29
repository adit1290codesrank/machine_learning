#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>
#include <chrono>
#include "../include/core/matrix.h"
#include "../include/core/utils.h"
#include "../include/layers/dense.h"
#include "../include/layers/softmax.h"
#include "../include/layers/conv2d.h"
#include "../include/layers/pooling.h"
#include "../include/layers/batchnorm.h"
#include "../include/layers/dropout.h"
#include "../include/activation.h"
#include "../include/network.h"
#include "../include/io/data.h"
#include <iomanip>

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
            if(argmax(predictions, j) == argmax(Y_batch, j)) correct++;
        }
    }
    return (double)correct / X.rows * 100.0;
}

double calculate_loss(const Matrix& predictions, const Matrix& targets) 
{
    double loss = 0;
    int n = predictions.rows;
    for (int i = 0; i < n; i++) for (int j = 0; j < predictions.cols; j++) if (targets(i, j) > 0.5) loss -= std::log(std::max(predictions(i, j), 1e-12));     
    return loss / n;
}

int main()
{
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

    nn.add(new Dropout(0.5));

    nn.add(new Dense(512, 128));
    nn.add(new BatchNorm(128));
    nn.add(new Activation(leaky_relu, dleaky_relu));

    nn.add(new Dropout(0.25));

    nn.add(new Dense(128, 47));
    nn.add(new Softmax());

    int epochs = 10;
    int batch_size = 128;
    double learning_rate = 0.001;

    std::vector<int> indices(X_train.rows);
    std::iota(indices.begin(), indices.end(), 0); 
    std::random_device rd;
    std::mt19937 g(rd());

    std::cout << "Starting CNN Training..." << std::endl;
    
    for(int epoch=1; epoch<=epochs; epoch++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), g);
        double epoch_loss=0.0;
        int num_batches = (X_train.rows+batch_size-1)/batch_size;
        if(epoch == 6) 
        {
            std::cout << "[!] Scheduler: Dropping Learning Rate to 0.0001" << std::endl;
            learning_rate = 0.0001;
        }

        for(int i=0; i < X_train.rows; i += batch_size)
        {
            int end = std::min(i + batch_size, X_train.rows);
            int current_batch_size = end - i;

            Matrix X_batch(current_batch_size, X_train.cols);
            Matrix Y_batch(current_batch_size, Y_train.cols);

            for(int b = 0; b < current_batch_size; b++) 
            {
                int idx = indices[i + b];
                for(int c=0; c < X_train.cols; c++) X_batch(b, c) = X_train(idx, c);
                for(int c=0; c < Y_train.cols; c++) Y_batch(b, c) = Y_train(idx, c);
            }
            

            Matrix output=nn.predict(X_batch);
            epoch_loss+=calculate_loss(output,Y_batch);
            nn.fit(X_batch, Y_batch, 1, learning_rate);
            
            int batch_idx = i / batch_size;
            double running_loss = epoch_loss / (batch_idx + 1); 

            if (batch_idx % 50 == 0) 
            { 
                float progress = (float)batch_idx / num_batches;
                int bar_width = 30;
                int pos = bar_width * progress;
                std::cout << "\rEpoch " << epoch << " [";
                for (int b = 0; b < bar_width; ++b) 
                {
                    if (b < pos) std::cout << "=";
                    else if (b == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << "% " << "| Loss: " << std::fixed << std::setprecision(4) << running_loss << " " << std::flush;
            }
        }
        std::cout << "] 100%" << std::endl;
        double acc = get_accuracy(nn, X_test, Y_test);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        int minutes = (int)elapsed.count() / 60;
        int seconds = (int)elapsed.count() % 60;

        std::cout << "Epoch " << epoch << " Result: " << acc << "% Accuracy" 
                  << " | Time: " << minutes << "m " << seconds << "s" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    std::cout << "Test Accuracy: " << get_accuracy(nn, X_test, Y_test) << "%" << std::endl;
    nn.save("emnist_model.bin");
    return 0;
}