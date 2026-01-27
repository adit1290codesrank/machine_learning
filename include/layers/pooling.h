#ifndef POOLING_H
#define POOLING_H

#include "layer.h"
#include "../core/matrix.h"
#include <vector>
#include <cfloat>

class Pooling:public Layer
{
    public:
        Pooling(int h,int w,int d,int pool_size=2,int stride=2);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta, double learning_rate) override;
    
    private:
        int h,w,d,pool_size,stride,oh,ow;
        std::vector<std::vector<int>> max_cache;
};

#endif