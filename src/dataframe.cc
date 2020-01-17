#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "dataframe.h"

Series Series::slice(size_t start, size_t end) const
{
    if (end < start) {
        throw std::invalid_argument("Invald slice: end < start");
    } else if (start >= _size) {
        throw std::invalid_argument("Invald slice: start is out of boundf");
    } else if (end >= _size) {
        throw std::invalid_argument("Invald slice: end is out of boundf");
    }

    return Series(_data + start, end - start);
}

Series Series::slice(size_t start) const
{
    return Series(_data + start, _size - start);
}

void DataFrame::load(const std::string &path)
{
    if (ends_with(path, ".csv")) {
        load_csv(path);
    } else if (ends_with(path, ".hdf5") || ends_with(path, ".h5")) {
        load_hdf5(path);
    } else {
        throw std::invalid_argument("Unknown file type " + path);
    }

    create_timeseries();
}

void DataFrame::load_csv(const std::string &path)
{
    std::ifstream ifs(path);
    std::string line;
    std::vector<std::vector<float>> columns;

    if (!ifs) {
        throw std::invalid_argument("Failed to open file " + path);
    }

    _n_rows = 0;
    _n_columns = 0;

    auto is_header = true;

    while (ifs >> line) {
        std::stringstream ss(line);
        std::string cell;

        for (auto i = 0; std::getline(ss, cell, ','); i++) {
            // Read header
            if (is_header) {
                columns.push_back(std::vector<float>());
                _n_columns++;
                continue;
            }

            // Read body
            columns[i].push_back(std::stof(cell));
        }

        if (is_header) {
            is_header = false;
            continue;
        }
        _n_rows++;
    }

    _data.resize(_n_rows * _n_columns);
    for (auto i = 0u; i < _n_columns; i++) {
        std::copy(columns[i].begin(), columns[i].end(),
                  _data.begin() + i * _n_rows);
    }
}

void DataFrame::load_hdf5(const std::string &path)
{
    const HighFive::File file(path, HighFive::File::ReadOnly);
    const auto dataset = file.getDataSet("/values");
    const auto shape = dataset.getDimensions();

    _n_columns = shape[0];
    _n_rows = shape[1];

    _data.resize(_n_rows * _n_columns);
    dataset.read(_data.data());
}

void DataFrame::create_timeseries()
{
    columns.resize(_n_columns);
    for (auto i = 0u; i < _n_columns; i++) {
        columns[i] = Series(_data.data() + i * _n_rows, _n_rows);
    }
}

bool DataFrame::ends_with(const std::string &str,
                          const std::string &suffix) const
{
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#endif
