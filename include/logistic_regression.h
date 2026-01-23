#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"

class LogisticRegression
{
    public:
        LogisticRegression(double learning_rate=0.01, int n_iterations=1000);
        ~LogisticRegression();
        void fit(const Matrix& X, const Matrix& Y);
        double score(const Matrix& X, const Matrix& Y) const;
        double loss(const Matrix& X, const Matrix& Y) const;
        Matrix predict(const Matrix& X) const;
        Matrix predict_proba(const Matrix& X) const;
    private:
        double learning_rate;
        int n_iterations;
        Matrix w;
        double b;
};

#endif