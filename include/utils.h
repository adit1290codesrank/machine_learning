#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"
#include<cmath>

struct TrainTestSplit
{
    Matrix X_train;
    Matrix X_test;
    Matrix y_train;
    Matrix y_test;
};

struct Normalization_with_mean_std
{
    Matrix matrix;
    Matrix mean;
    Matrix std;
};

struct Normalization_with_min_max
{
    Matrix matrix;
    Matrix min;
    Matrix max;
};

TrainTestSplit train_test_split(const Matrix& X, const Matrix& y, double test_size=0.2, bool shuffle=true);
Normalization_with_mean_std normalize(const Matrix& X);
Normalization_with_min_max min_max_scale(const Matrix& X);

inline double sigmoid(double x) {return 1.0/(1.0+std::exp(-x));} 

#endif