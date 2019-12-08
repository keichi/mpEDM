#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dataset.h"

Dataset::Dataset(const std::string &path) : Dataset()
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
        n_rows++;
    }

    for (const auto &column : columns) {
        timeseries.push_back(Timeseries(column));
    }
}

#endif
