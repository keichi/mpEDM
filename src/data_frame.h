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

    DataFrame() : _n_rows(0), _n_columns(0) {}
    DataFrame(int n_rows, int n_columns)
        : _n_rows(n_rows), _n_columns(n_columns)
    {
        create_timeseries();
    }
    DataFrame(const std::vector<float> &data, int n_rows, int n_columns)
        : _data(data), _n_rows(n_rows), _n_columns(n_columns)
    {
        create_timeseries();
    }

    const float *data() const { return _data.data(); }
    size_t n_rows() const { return _n_rows; }
    size_t n_columns() const { return _n_columns; }

    void load_csv(const std::string &path);
    void load_hdf5(const std::string &path, const std::string &dataset);

protected:
    // raw data stored in column-major
    std::vector<float> _data;
    size_t _n_rows;
    size_t _n_columns;

    void create_timeseries();
};

#endif
