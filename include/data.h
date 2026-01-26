#ifndef DATA_H
#define DATA_H

#include "matrix.h"
#include <string>
#include <vector>

class DataLoader
{
    public:
        static Matrix load_images(const std::string& filepath);
        static Matrix load_labels(const std::string& filepath);
};

#endif