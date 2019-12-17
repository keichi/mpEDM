#ifndef __DATASET_H__
#define __DATASET_H__

#include <string>
#include <vector>

class Timeseries
{
public:
    Timeseries() : Timeseries(nullptr, 0) {}
    Timeseries(const float *data, uint32_t size) : _data(data), _size(size) {}
    explicit Timeseries(const std::vector<float> &vec)
        : Timeseries(vec.data(), vec.size())
    {
    }

    const float *data() const { return _data; }
    uint32_t size() const { return _size; }

    const float &operator[](uint32_t i) const { return _data[i]; };

protected:
    const float *_data;
    uint32_t _size;
};

class Dataset
{
public:
    std::vector<Timeseries> timeseries;

    Dataset() : _n_rows(0), is_header(true) {}

    void load(const std::string &path);
    uint32_t n_rows() const { return _n_rows; }
    uint32_t n_cols() const { return timeseries.size(); }

protected:
    std::vector<std::vector<float>> columns;
    uint32_t _n_rows;
    bool is_header;

    void load_csv(const std::string &path);
#ifdef ENABLE_HDF5_READER
    void load_hdf5(const std::string &path);
#endif
    bool ends_with(const std::string &full, const std::string &ending);
};

#endif
