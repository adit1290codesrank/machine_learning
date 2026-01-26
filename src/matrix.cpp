#include "../include/matrix.h"
#include <omp.h>
#include <stdexcept>
#include <functional>
#include <iostream>

Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

Matrix::Matrix(int r,int c) : rows(r), cols(c)
{
    data = new double[rows*cols];
    for (int i=0; i<rows*cols; ++i) data[i] = 0.0;
}

Matrix::Matrix(const Matrix& matrix) : rows(matrix.rows), cols(matrix.cols)
{
    data = new double[rows*cols];
    for (int i=0; i<rows*cols; ++i) data[i] = matrix.data[i];
}

Matrix::~Matrix()
{
    delete[] data;
}

Matrix Matrix::transpose() const
{
    Matrix ans(cols, rows);
    for(int i=0;i<rows;i++) for(int j= 0; j<cols;j++)ans.data[j*rows+i] = data[i*cols+j]; 
    return ans;
}

double& Matrix::operator()(int r, int c)
{
    if(r<0||r>=rows||c<0||c>=cols) throw std::out_of_range("Matrix indices out of range");
    return data[r*cols+c];
}

double Matrix::operator()(int r, int c) const
{
    if(r<0||r>=rows||c<0||c>=cols) throw std::out_of_range("Matrix indices out of range");
    return data[r*cols+c];
}

Matrix& Matrix::operator=(const Matrix& matrix)
{
    if(this!=&matrix)
    {
        delete[] data;
        rows=matrix.rows;
        cols=matrix.cols;
        data=new double[rows*cols];
        for(int i=0;i<rows*cols;++i)data[i]=matrix.data[i];
    }
    return *this;
}

bool Matrix::operator==(const Matrix& matrix) const
{
    if(rows!=matrix.rows||cols!=matrix.cols) return false;
    for(int i=0;i<rows*cols;++i)if(data[i]!=matrix.data[i]) return false;
    return true;
}

bool Matrix::operator!=(const Matrix& matrix) const
{
    return !(*this==matrix);
}

Matrix Matrix::operator+(const Matrix& matrix) const
{
    if(rows!=matrix.rows||cols!=matrix.cols) throw std::invalid_argument("Dimension mismatch");;
    Matrix ans(rows,cols);
    for(int i=0;i<rows*cols;++i)ans.data[i]=data[i]+matrix.data[i];
    return ans;
}

Matrix Matrix::operator+(double scalar) const 
{
    Matrix ans(rows, cols);
    for(int i = 0; i < rows * cols; ++i)ans.data[i] = data[i] + scalar;
    return ans;
}

Matrix Matrix::operator-(double scalar) const 
{
    Matrix ans(rows, cols);
    for(int i = 0; i < rows * cols; ++i)ans.data[i] = data[i] - scalar;
    return ans;
}

Matrix Matrix::operator-(const Matrix& matrix) const
{
    if(rows!=matrix.rows||cols!=matrix.cols) throw std::invalid_argument("Dimension mismatch");;
    Matrix ans(rows,cols);
    for(int i=0;i<rows*cols;++i)ans.data[i]=data[i]-matrix.data[i];
    return ans;
}

Matrix Matrix::operator*(const Matrix& matrix) const
{
    if(cols!=matrix.rows) throw std::invalid_argument("Dimension mismatch");;
    Matrix ans(rows,matrix.cols);
    #pragma omp parallel for
    for(int i=0;i<rows;i++) for(int k=0;k<cols;k++) for(int j=0;j<matrix.cols;j++)
        ans.data[i*matrix.cols+j]+=data[i*cols+k]*matrix.data[k*matrix.cols+j];
    //ikj loop order for better cache performance because it stays constant for the inner loop
    return ans;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix ans(rows,cols);
    for(int i=0;i<rows*cols;++i)ans.data[i]=data[i]*scalar;
    return ans;
}

Matrix Matrix::identity(int size)
{
    Matrix id(size,size);
    for(int i=0;i<size;++i)id.data[i*size+i]=1.0;
    return id;
}

Matrix Matrix::zeros(int r, int c)
{
    return Matrix(r,c);
}

Matrix Matrix::ones(int r, int c)
{
    Matrix one(r,c);
    for(int i=0;i<r*c;++i)one.data[i]=1.0;
    return one;
}

Matrix Matrix::random(int r, int c, double min, double max)
{
    Matrix matrix(r,c);
    for(int i=0;i<r*c;++i)
    {
        double f = (double)rand() / RAND_MAX;
        matrix.data[i] = min + f * (max - min);
    }
    return matrix;
}

Matrix Matrix::apply(double (*function)(double)) const
{
    Matrix result(rows,cols);
    for(int i=0;i<rows*cols;++i)result.data[i]=function(data[i]);
    return result;
}

Matrix Matrix::sum_rows() const
{
    Matrix result(1,cols);
    for(int j=0;j<cols;++j)
    {
        double sum=0.0;
        for(int i=0;i<rows;++i)sum+=data[i*cols+j];
        result.data[j]=sum;
    }
    return result;
}

Matrix Matrix::Hadamard(const Matrix& matrix) const
{
    if(rows!=matrix.rows||cols!=matrix.cols) throw std::invalid_argument("Dimension mismatch");
    Matrix ans(rows,cols);
    for(int i=0;i<rows*cols;++i)ans.data[i]=data[i]*matrix.data[i];
    return ans;
}

Matrix Matrix::slice(int start,int end)
{
    if(start<0||end>rows||start>=end) throw std::out_of_range("Slice indices out of range");
    Matrix result(end-start,cols);
    for(int i=start;i<end;++i) for(int j=0;j<cols;++j) result.data[(i-start)*cols+j]=data[i*cols+j];
    return result;
}