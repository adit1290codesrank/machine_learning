#ifndef MATRIX_H
#define MATRIX_H

#include <functional>
#include <fstream>
#include <vector>

extern "C" 
{
    void launch_matmul(double* d_A, double* d_B, double* d_C, int m, int k, int n);
    void launch_hadamard(double* dA, double* dB, double* dC, int size);
    void gpu_alloc(double** ptr, size_t size);
    void gpu_free(double* ptr);
    void gpu_memcpy_h2d(double* dest, const double* src, size_t size);
    void gpu_memcpy_d2h(double* dest, const double* src, size_t size);
    void launch_conv2d_lean(const double* d_in, const double* d_k, double* d_out, int b, int h, int w, int d, int oh, int ow, int f, int k);
    void launch_conv2d_backward_lean(const double* d_in, const double* d_del, const double* d_k, double* d_dk, double* d_db, double* d_prev, int b, int h, int w, int d, int oh, int ow, int f, int k);
}

class Matrix
{
    friend class Conv2D;
    public:
        int rows;
        int cols;

        //Constructors and Destructor
        Matrix();
        Matrix(int r, int c);
        /*
        c++ default copy constructor does shallow copy which causes double free error. (both share same memory)  
        copy constructor needs to do deep copy.
        allocate new memory and copy values over.
        */
        Matrix(const Matrix& matrix);
        ~Matrix();

        //Operations
        double& operator()(int r, int c);
        double operator()(int r, int c) const;
        Matrix& operator=(const Matrix& matrix);
        bool operator==(const Matrix& matrix) const;
        bool operator!=(const Matrix& matrix) const;
        Matrix operator+(const Matrix& matrix) const;
        Matrix operator+(double scalar) const;
        Matrix operator-(const Matrix& matrix) const;
        Matrix operator-(double scalar) const;
        /*
        Loop order ikj for better cache performance because it stays constant for the inner loop
        */
        Matrix operator*(const Matrix& matrix) const;
        Matrix operator*(double scalar) const;

        //Utilities
        Matrix slice(int start,int end);
        Matrix transpose() const;
        Matrix sum_rows() const;
        Matrix Hadamard(const Matrix& matrix) const;
        static Matrix identity(int size);
        static Matrix zeros(int r, int c);
        static Matrix ones(int r, int c);
        static Matrix random(int r, int c, double min=-1.0, double max=1.0);
        Matrix apply(double (*function)(double)) const;
        void save(std::ofstream& file) const;
        void load(std::ifstream& file);
    private:
        /*
        1-D array used because only one pointer is needed which makes memory contiguous.
        2-D arrays has array of pointers which adds overhead and makes memory non-contiguous.
        */
        double* data;
};

#endif