#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void load_dataset(const std::string &fname, std::vector<std::vector<float>> &ds)
{
    std::ifstream ifs(fname);

    int row_idx = -1;

    while (!ifs.eof()) {
        std::vector<float> row;
        std::string line;
        ifs >> line;

        row_idx++;

        // Skip header
        if (row_idx == 0) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        ds.push_back(row);
    }

    std::cout << ds.size() << " rows read from " << fname << std::endl;
}

void select_timeseries(const std::vector<std::vector<float>> &ds, int col_idx,
                       std::vector<float> &ts)
{
    for (const auto &row : ds) {
        ts.push_back(row[col_idx]);
    }
}

void print_timeseries(const std::vector<float> &ts)
{
    for (const auto val : ts) {
        std::cout << val << std::endl;
    }
}

void create_block(const std::vector<float> &ts, int E, int tau,
                  std::vector<const float *> &block)
{
    block.resize(E);

    for (int i = 0; i < E; i++) {
        block[i] = ts.data() + (E - i - 1) * tau;
    }
}

void print_block(const std::vector<const float *> &block, int len)
{
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < block.size(); j++) {
            std::cout << block[j][i] << ", ";
        }
        std::cout << std::endl;
    }
}

void calc_distances(const std::vector<const float *> &block, int len,
                    std::vector<std::vector<std::pair<float, int>>> &distances)
{
    distances.resize(len);

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        std::vector<std::pair<float, int>> dist_vec(len);

        for (int j = 0; j < len; j++) {
            float norm = 0.0f;

            for (int k = 0; k < block.size(); k++) {
                norm +=
                    (block[k][i] - block[k][j]) * (block[k][i] - block[k][j]);
            }

            dist_vec[j] = std::make_pair(std::sqrt(norm), j);
        }

        std::sort(dist_vec.begin(), dist_vec.end());
        distances[i] = dist_vec;
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::vector<float>> ds;

    load_dataset(argv[1], ds);

    for (int E = 20; E <= 20; E++) {
        std::vector<float> ts;

        select_timeseries(ds, 0, ts);

        std::vector<const float *> block;

        create_block(ts, E, 1, block);

        std::vector<std::vector<std::pair<float, int>>> distances;

        calc_distances(block, ts.size() - (E - 1) * 1, distances);
    }

    return 0;
}
