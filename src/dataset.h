#ifndef __DATASET_H__
#define __DATASET_H__

#include <stdexcept>
#include <string>
#include <vector>

class Timeseries
{
public:
    Timeseries() : Timeseries(nullptr, 0) {}
    Timeseries(const float *data, size_t size) : _data(data), _size(size) {}
    explicit Timeseries(const std::vector<float> &vec)
        : Timeseries(vec.data(), vec.size())
    {
    }

    const float *data() const { return _data; }
    size_t size() const { return _size; }

    Timeseries slice(size_t start, size_t end) const;
    Timeseries slice(size_t start) const;

    const float &operator[](size_t i) const { return _data[i]; };

protected:
    const float *_data;
    size_t _size;
};

class Dataset
{
public:
    std::vector<Timeseries> timeseries;

    Dataset() : _n_rows(0), _n_cols(0) {}

    void load(const std::string &path);
    const float *data() const { return _data.data(); }
    size_t n_rows() const { return _n_rows; }
    size_t n_cols() const { return _n_cols; }

protected:
    // raw data stored in column-major
    std::vector<float> _data;
    size_t _n_rows;
    size_t _n_cols;

    void load_csv(const std::string &path);
#ifdef ENABLE_HDF5_READER
    void load_hdf5(const std::string &path);
#endif
    void create_timeseries();
    bool ends_with(const std::string &full, const std::string &ending) const;
};

#endif
