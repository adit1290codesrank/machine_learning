#include "../include/io/data.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

uint32_t swap_endian(uint32_t val)
{
    val=((val<<8)&0xFF00FF00)|((val>>8)&0xFF00FF);
    return (val<<16)|(val>>16);
}

Matrix DataLoader::load_images(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        exit(1);
    }

    uint32_t magic=0,n_images=0,rows=0,cols=0;
    file.read((char*)&magic, sizeof(magic));
    file.read((char*)&n_images, sizeof(n_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    magic=swap_endian(magic);
    n_images=swap_endian(n_images);
    rows=swap_endian(rows);
    cols=swap_endian(cols);
    std::cout << "Loading " << n_images << " images (" << rows << "x" << cols << ")" << std::endl;

    Matrix X(n_images, rows*cols);
    for(int i=0;i<n_images;i++)
    {
        for(int j=0;j<rows;j++)
        {
            for(int k=0;k<cols;k++)
            {
                unsigned char pixel=0;
                file.read((char*)&pixel, sizeof(pixel));
                X(i, k*rows + j) = (double)pixel / 255.0;
            }
        }
    }
    return X;
}

Matrix DataLoader::load_labels(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        exit(1);
    }
    uint32_t magic=0, n_labels=0;
    file.read((char*)&magic, sizeof(magic));
    file.read((char*)&n_labels, sizeof(n_labels));
    magic = ((magic << 8) & 0xFF00FF00) | ((magic >> 8) & 0xFF00FF);
    magic = (magic << 16) | (magic >> 16);
    n_labels = ((n_labels << 8) & 0xFF00FF00) | ((n_labels >> 8) & 0xFF00FF);
    n_labels = (n_labels << 16) | (n_labels >> 16);
    std::cout << "Loading " << n_labels << " labels..." << std::endl;
    Matrix Y = Matrix::zeros(n_labels, 47);
    for(int i=0; i<n_labels; i++) 
    {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        if(label < 47) Y(i, (int)label) = 1.0;
    }
    return Y;
}