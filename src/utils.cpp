#include "../include/utils.h"
#include <stdexcept>
#include <cmath>

TrainTestSplit train_test_split(const Matrix& X, const Matrix& y, double test_size, bool shuffle)
{
    if(X.rows!=y.rows) throw std::invalid_argument("Number of samples in X and y must be the same");
    int rows=X.rows;
    int test_rows=(int)(rows*test_size);
    int train_rows=rows - test_rows;

    int* index = new int[rows];
    for(int i=0;i<rows;++i)index[i]=i;
    for(int i=rows-1;i>0;--i)
    {
        int j=rand()%(i+1);
        std::swap(index[i],index[j]);
    }
    TrainTestSplit split;
    split.X_train=Matrix(train_rows,X.cols);
    split.X_test=Matrix(test_rows,X.cols);
    split.y_train=Matrix(train_rows,y.cols);
    split.y_test=Matrix(test_rows,y.cols);
    
    for(int i=0;i<train_rows;i++)
    {
        int idx=index[i];
        for(int j=0;j<X.cols;j++) split.X_train(i,j)=X(idx,j);
        for(int j=0;j<y.cols;j++) split.y_train(i,j)=y(idx,j);
    }

    for(int i=0;i<test_rows;i++)
    {
        int idx=index[i+train_rows];
        for(int j=0;j<X.cols;j++) split.X_test(i,j)=X(idx,j);
        for(int j=0;j<y.cols;j++) split.y_test(i,j)=y(idx,j);
    }
    
    delete[] index;
    return split;
}

Normalization_with_mean_std normalize(const Matrix& X)
{
    int m=X.rows;
    int n=X.cols;
    Normalization_with_mean_std norm;
    norm.matrix=Matrix(m,n);
    norm.mean=Matrix(1,n);
    norm.std=Matrix(1,n);
    for(int i=0;i<n;i++)
    {
        double sum=0.0;
        for(int j=0;j<m;j++) sum+=X(j,i);
        double mean=sum/m;
        norm.mean(0,i)=mean;
        double sq_sum=0.0;
        for(int j=0;j<m;j++) sq_sum+=(X(j,i)-mean)*(X(j,i)-mean);
        double std=sqrt(sq_sum/m);
        norm.std(0,i)=std;
        for(int j=0;j<m;j++)
        {
            if(std!=0)norm.matrix(j,i)=(X(j,i)-mean)/std;
            else norm.matrix(j,i)=0.0;
        }
    }
    return norm;
}

Normalization_with_min_max min_max_scale(const Matrix& X)
{
    int m=X.rows;
    int n=X.cols;
    Normalization_with_min_max norm;
    norm.matrix=Matrix(m,n);
    norm.min=Matrix(1,n);
    norm.max=Matrix(1,n);
    for(int i=0;i<n;i++)
    {
        double min_val=X(0,i);
        double max_val=X(0,i);
        for(int j=1;j<m;j++)
        {
            if(X(j,i)<min_val) min_val=X(j,i);
            if(X(j,i)>max_val) max_val=X(j,i);
        }
        norm.min(0,i)=min_val;
        norm.max(0,i)=max_val;
        double range=max_val - min_val;
        for(int j=0;j<m;j++)
        {
            if(range!=0) norm.matrix(j,i)=(X(j,i)-min_val)/range;
            else norm.matrix(j,i)=0.0;
        }
    }
    return norm;
}

double mse(const Matrix& y_true, const Matrix& y_pred)
{
    if(y_true.rows!=y_pred.rows || y_true.cols!=y_pred.cols) throw std::invalid_argument("Dimensions of y_true and y_pred must be the same");
    double sum=0.0;
    int m=y_true.rows;
    int n=y_true.cols;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            double diff=y_true(i,j)-y_pred(i,j);
            sum+=diff*diff;
        }
    }
    return sum/(2*m*n);
}

Matrix dmse(const Matrix& y_true, const Matrix& y_pred)
{
    if(y_true.rows!=y_pred.rows || y_true.cols!=y_pred.cols) throw std::invalid_argument("Dimensions of y_true and y_pred must be the same");
    int m=y_true.rows;
    int n=y_true.cols;
    Matrix gradient=Matrix::zeros(m,n);
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            gradient(i,j)=(y_pred(i,j)-y_true(i,j))/(m*n);
        }
    }
    return gradient;
}