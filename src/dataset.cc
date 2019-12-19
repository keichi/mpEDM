#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_HDF5_READER
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#endif

#include "dataset.h"

void Dataset::load(const std::string &path)
{
    if (ends_with(path, ".csv")) {
        load_csv(path);
    }
#ifdef ENABLE_HDF5_READER
    else if (ends_with(path, ".hdf5") || ends_with(path, ".h5")) {
        load_hdf5(path);
    }
#endif
    else {
        throw std::invalid_argument("Unknown file type: " + path);
    }

    create_timeseries();
}

void Dataset::load_csv(const std::string &path)
{
    std::ifstream ifs(path);
    std::string line;
    std::vector<std::vector<float>> columns;

    _n_rows = 0;
    _n_cols = 0;

    while (ifs >> line) {
        std::stringstream ss(line);
        std::string cell;

        for (auto i = 0; std::getline(ss, cell, ','); i++) {
            // Read header
            if (is_header) {
                columns.push_back(std::vector<float>());
                _n_cols++;
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

    _data.resize(_n_rows * _n_cols);
    for (auto i = 0u; i < _n_cols; i++) {
        std::copy(columns[i].begin(), columns[i].end(),
                  _data.begin() + i * _n_rows);
    }
}

#ifdef ENABLE_HDF5_READER
void Dataset::load_hdf5(const std::string &path)
{
    const HighFive::File file(path, HighFive::File::ReadOnly);
    const auto dataset = file.getDataSet("/values");
    const auto shape = dataset.getDimensions();

    _n_cols = shape[0];
    _n_rows = shape[1];

    _data.resize(_n_rows * _n_cols);
    dataset.read(_data.data());
}
#endif

void Dataset::create_timeseries()
{
    timeseries.resize(_n_cols);
    for (auto i = 0u; i < _n_cols; i++) {
        timeseries[i] = Timeseries(_data.data() + i * _n_rows, _n_rows);
    }
}

bool Dataset::ends_with(const std::string &str, const std::string &suffix) const
{
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#endif
