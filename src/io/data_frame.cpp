#include "../include/io/data_frame.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>

DataFrame::DataFrame() : rows(0), cols(0) {}

void DataFrame::read_csv(const std::string& filename, bool has_header, char delimiter)
{
    std::ifstream file(filename);
    if (!file.is_open())  throw std::runtime_error("Could not open file: " + filename);
    
    std::string line;
    rows=0;
    data.clear();
    column_names.clear();
    if(has_header && std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string col_name;
        while (std::getline(ss, col_name, delimiter)) 
        {
            col_name.erase(std::remove(col_name.begin(), col_name.end(), '\r'), col_name.end());
            col_name.erase(std::remove(col_name.begin(), col_name.end(), '\"'), col_name.end());
            column_names.push_back(col_name);
        }
        cols=column_names.size();
    }

    while(std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, delimiter)) 
        {
            cell.erase(std::remove(cell.begin(), cell.end(), '\r'), cell.end());
            cell.erase(std::remove(cell.begin(), cell.end(), '\"'), cell.end());
            row.push_back(cell);
        }
        if(cols==0) cols=row.size();
        data.push_back(row);
        rows++;
    }
    std::cout << "Loaded " << rows << " rows, " << cols << " columns." << std::endl;
}

void DataFrame::head(int n) const
{
    n = std::min(n, rows);
    if(!column_names.empty())
    {
        for(const auto& name : column_names) std::cout << name << "\t";
        std::cout << std::endl;
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<cols;j++) std::cout << data[i][j] << "\t";
        std::cout << std::endl;
    }
}

int DataFrame::get_column_index(const std::string& column_name) const
{
    auto it = std::find(column_names.begin(), column_names.end(), column_name);
    if(it == column_names.end()) throw std::invalid_argument("Column not found: " + column_name);
    return std::distance(column_names.begin(), it);
}

Matrix DataFrame::select(const std::vector<std::string>& columns) const
{
    Matrix result(rows, columns.size());
    for(size_t j=0;j<columns.size();j++)
    {
        int col_index = get_column_index(columns[j]);
        for(int i=0;i<rows;i++) 
        {
            try {
                if(data[i][col_index].empty()) result(i,j)=0.0;
                else result(i,j)=std::stod(data[i][col_index]);
            } catch (...) {
                result(i,j)=0.0;
            }
        }
    }
    return result;
}

Matrix DataFrame::select(int column_index, int end_column_index) const
{
    if(column_index<0||end_column_index>=cols||column_index>end_column_index) throw std::out_of_range("Column index out of range");
    int selected_cols = end_column_index - column_index + 1;
    Matrix result(rows, selected_cols);
    for(int j=0;j<selected_cols;j++) 
    {
        for(int i=0;i<rows;i++) 
        {
            try {
                if(data[i][column_index + j].empty()) result(i,j)=0.0;
                else result(i,j)=std::stod(data[i][column_index + j]);
            } catch (...) {
                result(i,j)=0.0;
            }
        }
    }
    return result;
}

Matrix DataFrame::get_column(const std::string& column_name) const
{
    int col_index = get_column_index(column_name);
    Matrix result(rows, 1);
    for(int i=0;i<rows;i++) 
    {
        try {
            if(data[i][col_index].empty()) result(i,0)=0.0;
            else result(i,0)=std::stod(data[i][col_index]);
        } catch (...) {
            result(i,0)=0.0;
        }
    }
    return result;
}

Matrix DataFrame::get_column_encode(const std::string& column_name) const
{
    int col_index = get_column_index(column_name);
    std::map<std::string, int> encoding_map;
    int code = 0;
    for(int i=0;i<rows;i++)
    {
        const std::string& value = data[i][col_index];
        if(encoding_map.find(value) == encoding_map.end())
        {
            encoding_map[value] = code++;
        }
    }
    Matrix result(rows, 1);
    for(int i=0;i<rows;i++) result(i,0)=encoding_map[data[i][col_index]];
    return result;
}

void DataFrame::info() const
{
    std::cout << "DataFrame Info:" << std::endl;
    std::cout << "Number of rows: " << rows << std::endl;
    std::cout << "Number of columns: " << cols << std::endl;
    if(!column_names.empty())
    {
        std::cout << "Columns:" << std::endl;
        for(const auto& name : column_names) std::cout << "- " << name << std::endl;
    }
}