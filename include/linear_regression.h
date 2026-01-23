#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include "matrix.h"

class LinearRegression
{
    public:
        LinearRegression(double learning_rate=0.01, int n_iterations=1000);
        ~LinearRegression();
        void fit(const Matrix& X, const Matrix& Y);
        double score(const Matrix& X, const Matrix& Y) const;
        Matrix predict(const Matrix& X) const;
    private:
        double learning_rate;
        int n_iterations;
        Matrix w;
        double b;
};
#endif // LINEAR_REGRESSION_H