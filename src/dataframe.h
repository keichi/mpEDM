#ifndef __DATAFRAME_H__
#define __DATAFRAME_H__

#include <stdexcept>
#include <string>
#include <vector>

class Series
{
public:
    Series() : Series(nullptr, 0) {}
    Series(const float *data, size_t size) : _data(data), _size(size) {}
    explicit Series(const std::vector<float> &vec)
        : Series(vec.data(), vec.size())
    {
    }

    const float *data() const { return _data; }
    size_t size() const { return _size; }

    Series slice(size_t start, size_t end) const;
    Series slice(size_t start) const;

    const float &operator[](size_t i) const { return _data[i]; };

protected:
    const float *_data;
    size_t _size;
};

class DataFrame
{
public:
    std::vector<Series> columns;

    DataFrame() : _n_rows(0), _n_cols(0) {}

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
