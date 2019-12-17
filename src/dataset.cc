#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_HDF5_READER
#include <highfive/H5Easy.hpp>
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
}

void Dataset::load_csv(const std::string &path)
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
        _n_rows++;
    }

    std::transform(columns.begin(), columns.end(),
                   std::back_inserter(timeseries),
                   [](const std::vector<float> &c) { return Timeseries(c); });
}

#ifdef ENABLE_HDF5_READER
void Dataset::load_hdf5(const std::string &path)
{
    HighFive::File file(path, HighFive::File::ReadOnly);

    const auto shape = H5Easy::getShape(file, "/values");

    _n_rows = shape[1];

    columns = H5Easy::load<std::vector<std::vector<float>>>(file, "/values");

    std::transform(columns.begin(), columns.end(),
                   std::back_inserter(timeseries),
                   [](const std::vector<float> &c) { return Timeseries(c); });
}
#endif

bool Dataset::ends_with(const std::string &str, const std::string &suffix)
{
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#endif
