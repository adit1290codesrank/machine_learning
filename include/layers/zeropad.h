#ifndef ZEROPAD_H
#define ZEROPAD_H
#include "layer.h"
#include "../core/matrix.h"

class ZeroPad : public Layer 
{
public:
    ZeroPad(int h, int w, int d, int pad);
    Matrix forward_pass(const Matrix& input) override;
    Matrix backward_pass(const Matrix& delta, double learning_rate) override;
private:
    int h, w, d, pad;
    int oh, ow;
};

#endif