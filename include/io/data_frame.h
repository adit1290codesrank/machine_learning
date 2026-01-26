#ifndef DATAFRAME_H
#define DATAFRAME_H

#include "../core/matrix.h"
#include <string>
#include <vector>

class DataFrame
{
    public:
        DataFrame();
        void read_csv(const std::string& filename,bool has_header=true,char delimiter=',');

        void head(int n=5) const;
        void info() const;

        Matrix select(const std::vector<std::string>& columns) const;
        Matrix select(int column_index,int end_column_index) const;
        Matrix get_column(const std::string& column_name) const;
        Matrix get_column_encode(const std::string& column_name) const;
        int get_column_index(const std::string& column_name) const;
    
    private:
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> data;
        int rows;
        int cols;
};

#endif