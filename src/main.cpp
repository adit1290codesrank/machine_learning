#include "../include/logistic_regression.h"
#include "../include/linear_regression.h"
#include "../include/matrix.h"
#include "../include/utils.h"
#include "../include/data_frame.h"
#include <bits/stdc++.h>
#include <chrono>

int main()
{
    DataFrame df;
    try
    {
        df.read_csv("data.csv",true);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error reading file: " << e.what() << std::endl;
        return 1;
    }
    df.info();
    Matrix X_raw = df.select(2, 31);
    Matrix Y = df.get_column_encode("diagnosis");
    std::cout << "X Shape: " << X_raw.rows << "x" << X_raw.cols << std::endl;
    std::cout << "Y Shape: " << Y.rows << "x" << Y.cols << std::endl;

    int train_size = (int)(X_raw.rows * 0.8);
    int test_size = X_raw.rows - train_size;

    Matrix X_train_raw(train_size, X_raw.cols);
    Matrix Y_train(train_size, 1);
    Matrix X_test_raw(test_size, X_raw.cols);
    Matrix Y_test(test_size, 1);

    for(int i=0;i<train_size;i++)
    {
        for(int j=0;j<X_raw.cols;j++) X_train_raw(i,j)=X_raw(i,j);
        Y_train(i,0)=Y(i,0);
    }

    for(int i=0;i<test_size;i++)
    {
        for(int j=0;j<X_raw.cols;j++) X_test_raw(i,j)=X_raw(i+train_size,j);
        Y_test(i,0)=Y(i+train_size,0);
    }

    Normalization_with_min_max norm = min_max_scale(X_train_raw);
    Matrix X_train = norm.matrix;

    Matrix X_test(test_size, X_raw.cols);
    for(int i = 0; i < test_size; i++) 
    {
        for(int j = 0; j < X_raw.cols; j++) 
        {
            double range = norm.max(0, j) - norm.min(0, j);
            if(range == 0) range = 1.0;
            X_test(i, j) = (X_test_raw(i, j) - norm.min(0, j)) / range;
        }
    }

    LogisticRegression model(0.1, 1000);
    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X_train, Y_train);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Training completed in " << duration << " ms" << std::endl;
    double train_acc = model.score(X_train, Y_train);
    double test_acc = model.score(X_test, Y_test);std::cout << "---------------------------------" << std::endl;
    std::cout << "Training Time:  " << duration << " ms" << std::endl;
    std::cout << "Train Accuracy: " << train_acc * 100 << "%" << std::endl;
    std::cout << "Test Accuracy:  " << test_acc * 100 << "%" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    return 0;
}