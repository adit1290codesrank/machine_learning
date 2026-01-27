#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"
#include "../core/matrix.h"
#include <vector>

class BatchNorm:public Layer
{
    public:
        BatchNorm(int features);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;
        void save(std::ofstream& file) override;
        void load(std::ifstream& file) override;
        Matrix g,b,mean,var;
    private:
        int features;
        double e=1e-8,momentum=0.9,b1=0.9,b2=0.999,m,v;
        Matrix mg,vg,mb,vb;
        int t=0;
        Matrix x_,std_inv;
};

#endif