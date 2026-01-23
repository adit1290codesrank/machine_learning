#ifndef MATRIX_H
#define MATRIX_H

class Matrix
{
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
        Matrix operator=(const Matrix& matrix);
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
        Matrix transpose() const;
        static Matrix identity(int size);
        static Matrix zeros(int r, int c);
        static Matrix ones(int r, int c);
        static Matrix random(int r, int c, double min=-1.0, double max=1.0);
        Matrix apply(double (*function)(double)) const;
    private:
        /*
        1-D array used because only one pointer is needed which makes memory contiguous.
        2-D arrays has array of pointers which adds overhead and makes memory non-contiguous.
        */
        double* data;
};

#endif