#include "../include/linear_regression.h"
#include "../include/matrix.h"
#include <iostream>
#include <chrono>

LinearRegression::LinearRegression(double learning_rate, int n_iterations)
    : learning_rate(learning_rate), n_iterations(n_iterations) {}

LinearRegression::~LinearRegression() {}

Matrix LinearRegression::predict(const Matrix& X) const
{
    return (X * w) + b;
}

double LinearRegression::score(const Matrix& X, const Matrix& Y) const
{
    Matrix Y_pred = predict(X);
    double ss_total = 0.0;
    double ss_residual = 0.0;
    double y_mean = 0.0;
    int n = Y.rows;
    for(int i=0;i<n;i++) y_mean += Y(i,0);
    y_mean /= n;
    for(int i=0;i<n;i++)
    {
        ss_total += (Y(i,0)-y_mean)*(Y(i,0)-y_mean);
        ss_residual += (Y(i,0)-Y_pred(i,0))*(Y(i,0)-Y_pred(i,0));
    }
    return 1.0-(ss_residual/ss_total);
}

void LinearRegression::fit(const Matrix& X,const Matrix& Y)
{
    int n = X.rows;
    int m = X.cols;
    w = Matrix::random(m,1);
    b = 0.0;
    Matrix Xt = X.transpose();
    for(int i=0;i<n_iterations;i++)
    {
        Matrix Y_pred = predict(X);
        Matrix dw = (Xt*(Y_pred-Y))*(1.0/n);
        double db = 0.0;
        for(int j=0;j<n;j++) db += (Y_pred(j,0)-Y(j,0));
        db /= n;
        w = w - (dw * learning_rate);
        b = b - (db * learning_rate);
        if(i%100==0) std::cout << "Iteration " << i << ": Score = " << score(X,Y) << std::endl;
    }
}