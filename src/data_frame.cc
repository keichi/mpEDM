#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "data_frame.h"

Series Series::slice(size_t start, size_t end) const
{
    if (end < start) {
        throw std::invalid_argument("Invald slice: end < start");
    } else if (start >= _size) {
        throw std::invalid_argument("Invald slice: start is out of bounds");
    } else if (end >= _size) {
        throw std::invalid_argument("Invald slice: end is out of bounds");
    }

    return Series(_data + start, end - start);
}

Series Series::slice(size_t start) const
{
    return Series(_data + start, _size - start);
}

void DataFrame::load_csv(const std::string &path)
{
    std::ifstream ifs(path);
    std::string line;
    // cppcheck-suppress shadowVariable
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

    create_timeseries();
}

void DataFrame::load_hdf5(const std::string &path, const std::string &ds_name)
{
    const HighFive::File file(path, HighFive::File::ReadOnly);
    const auto dataset = file.getDataSet(ds_name);
    const auto shape = dataset.getDimensions();

    _n_rows = shape[0];
    _n_columns = shape[1];

    const size_t MAX_CHUNK_SIZE = 100;

    std::vector<float> rows(MAX_CHUNK_SIZE * _n_columns);
    _data.resize(_n_rows * _n_columns);

    for (auto i = 0u; i < _n_rows; i += MAX_CHUNK_SIZE) {
        const auto chunk_size = std::min(MAX_CHUNK_SIZE, _n_rows - i);

        dataset.select({i, 0}, {chunk_size, _n_columns}).read(rows.data());

        for (auto j = 0u; j < chunk_size; j++) {
            for (auto k = 0u; k < _n_columns; k++) {
                _data[k * _n_rows + (i + j)] = rows[j * _n_columns + k];
            }
        }
    }

    create_timeseries();
}

void DataFrame::create_timeseries()
{
    columns.resize(_n_columns);
    for (auto i = 0u; i < _n_columns; i++) {
        columns[i] = Series(_data.data() + i * _n_rows, _n_rows);
    }
}

#endif
