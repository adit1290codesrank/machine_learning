#include "../include/logistic_regression.h"
#include "../include/utils.h"
#include <iostream>
#include <cmath>

LogisticRegression::LogisticRegression(double learning_rate, int n_iterations)
    : learning_rate(learning_rate), n_iterations(n_iterations) {}

LogisticRegression::~LogisticRegression() {}

Matrix LogisticRegression::predict_proba(const Matrix& X) const
{
    Matrix linear = (X*w)+b;
    return linear.apply(sigmoid);
}

Matrix LogisticRegression::predict(const Matrix& X) const
{
    Matrix probs = predict_proba(X);
    Matrix ans(probs.rows, probs.cols);
    for(int i=0;i<probs.rows;++i) ans(i,0)= (probs(i,0)>=0.5)?1.0:0.0;
    return ans;
}

void LogisticRegression::fit(const Matrix& X,const Matrix& Y)
{
    int m = X.rows;
    int n = X.cols;

    w=Matrix::zeros(n,1);
    b=0.0;
    Matrix Xt = X.transpose();
    for(int i=0;i<n_iterations;i++)
    {
        Matrix y_pred=predict_proba(X);
        Matrix dw=(Xt*(y_pred - Y))*(1.0/m);
        double db=0.0;
        for(int j=0;j<m;j++) db+=(y_pred(j,0)-Y(j,0));
        db/=m;
        w=w-(dw*learning_rate);
        b=b-(db*learning_rate);
        if(i%100==0) std::cout<<"Iteration "<<i<<": Loss = "<<loss(X,Y)<<std::endl;
    }
}

double LogisticRegression::loss(const Matrix& X, const Matrix& Y) const
{
    Matrix Y_pred = predict_proba(X);
    int m = Y.rows;
    double loss = 0.0;
    for(int i=0;i<m;i++)
    {
        double y = Y(i,0);
        double y_pred = Y_pred(i,0);
        //add small value to avoid log(0)
        loss+=-(y*std::log(y_pred+1e-15)+(1-y)*std::log(1-y_pred+1e-15));
    }
    return loss/m;
}

double LogisticRegression::score(const Matrix& X, const Matrix& Y) const
{
    Matrix Y_pred=predict(X);
    int correct=0;
    for(int i=0;i<Y.rows;i++) if(Y(i,0)==Y_pred(i,0)) correct++;
    return (double)correct/Y.rows;
}