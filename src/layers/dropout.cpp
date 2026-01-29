#include "../include/layers/dropout.h"
#include <random>
#include <omp.h>

Dropout::Dropout(double x) : x(x) {}

Matrix Dropout::forward_pass(const Matrix& input)
{
    if (!this->is_training) return input * (1.0 - x);
    mask = Matrix(input.rows, input.cols);
    Matrix output(input.rows, input.cols);

    #pragma omp parallel 
    {
        std::default_random_engine gen(std::random_device{}() ^ omp_get_thread_num());
        std::bernoulli_distribution dist(1.0 - x);

        #pragma omp for collapse(2)
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                double m = dist(gen) ? 1.0 : 0.0;
                mask(i, j) = m;
                output(i, j) = input(i, j) * m;
            }
        }
    }
    return output;
}

Matrix Dropout::backward_pass(const Matrix& delta, double learning_rate)
{
    
    return delta.Hadamard(mask);
}