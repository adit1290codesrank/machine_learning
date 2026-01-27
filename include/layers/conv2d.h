#ifndef CONV2D_H
#define CONV2D_H

#include "layer.h"
#include "../core/matrix.h"
#include <vector>
#include <cmath>

class Conv2D:public Layer
{
    public:
        Conv2D(int h,int w,int d,int f,int k);
        Matrix forward_pass(const Matrix& input) override;
        Matrix backward_pass(const Matrix& delta,double learning_rate) override;
        void save(std::ofstream& file) override;
        void load(std::ifstream& file) override;
        void init();
    private:
        int h,w,d,f,k; 
        int oh,ow;
        std::vector<std::vector<Matrix>> kernels;
        std::vector<double> b;
        Matrix input;
        std::vector<std::vector<Matrix>> mk,vk;
        std::vector<double> mb,vb;
        int t=0;
        double b1=0.9,b2=0.999,e=1e-8,m,v;

};

#endif