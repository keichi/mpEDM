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

    const float& operator[](size_t i) const { return _data[i]; };

protected:
    const float *_data;
    size_t _size;
};

class Dataset
{
public:
    std::vector<Timeseries> timeseries;
    size_t n_rows;

    Dataset() : n_rows(0), is_header(true) {}
    Dataset(const std::string &path);

protected:
    std::vector<std::vector<float>> columns;
    bool is_header;
};

#endif
