#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class Timeseries
{
public:
    Timeseries() : Timeseries(nullptr, 0) {}
    Timeseries(const float *data, size_t size) : _data(data), _size(size) {}
    Timeseries(const std::vector<float> &vec)
        : Timeseries(vec.data(), vec.size())
    {
    }

    const float *data() const { return _data; }
    size_t size() const { return _size; }

protected:
    const float *_data;
    size_t _size;
};

class Dataset
{
public:
    std::vector<Timeseries> timeseries;
    size_t n_rows;

    Dataset() : n_rows(0) {}

    Dataset(const std::string &path)
    {
        std::ifstream ifs(path);
        std::string line;

        while (ifs >> line) {
            std::stringstream ss(line);
            std::string cell;

            for (auto i = 0; std::getline(ss, cell, ','); i++) {
                // Read header
                if (is_header) {
                    columns.push_back(std::vector<float>());
                    continue;
                }

                // Read body
                columns[i].push_back(std::stof(cell));
            }

            if (is_header) {
                is_header = false;
                continue;
            }
            n_rows++;
        }

        for (const auto &column : columns) {
            timeseries.push_back(Timeseries(column));
        }
    }

protected:
    std::vector<std::vector<float>> columns;
    bool is_header = true;
};

#endif
