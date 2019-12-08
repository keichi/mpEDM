#ifndef __DATASET_H__
#define __DATASET_H__

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

    Dataset(const std::string &path);

protected:
    std::vector<std::vector<float>> columns;
    bool is_header = true;
};

#endif
