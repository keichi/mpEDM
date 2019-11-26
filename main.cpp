#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "timer.hpp"

class Dataset
{
public:
    std::vector<std::vector<float>> cols;
    int n_rows;
    int n_cols;

    Dataset() : n_rows(0), n_cols(0) {}

    void load_csv(const std::string &fname)
    {
        n_rows = -1;
        std::ifstream ifs(fname);

        while (!ifs.eof()) {
            std::string line;
            ifs >> line;
            n_rows++;

            std::stringstream ss(line);
            std::string cell;

            for (int i = 0; std::getline(ss, cell, ','); i++) {
                if (n_rows == 0) {
                    cols.push_back(std::vector<float>());
                    n_cols++;
                    continue;
                }

                cols[i].push_back(std::stof(cell));
            }
        }
    }
};

class Block
{
public:
    std::vector<const float *> cols;
    int E;
    int tau;
    int n;

    Block(const Dataset &ds, int col_idx, int E, int tau) : E(E), tau(tau)
    {
        cols.resize(E);

        for (int i = 0; i < E; i++) {
            cols[i] = ds.cols[col_idx].data() + (E - i - 1) * tau;
        }

        n = ds.n_rows - (E - 1) * tau;
    }

    void print()
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < E; j++) {
                std::cout << cols[j][i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void
    calc_distances(std::vector<std::vector<std::pair<float, int>>> &distances)
    {
        distances.resize(n);
        for (int i = 0; i < n; i++) {
            distances[i].resize(n);
        }

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float norm = 0.0f;

                for (int k = 0; k < E; k++) {
                    norm +=
                        (cols[k][i] - cols[k][j]) * (cols[k][i] - cols[k][j]);
                }

                distances[i][j].first = std::sqrt(norm);
                distances[i][j].second = j;
            }
        }

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            std::sort(distances[i].begin(), distances[i].end());
        }
    }
};

int main(int argc, char *argv[])
{
    Dataset ds;
    ds.load_csv(argv[1]);

    std::cout << ds.n_rows << " rows read from " << argv[1] << std::endl;

    const int tau = 1;
    const int E_max = 20;

    Timer timer;

    for (int i = 0; i < ds.n_cols; i++) {
        std::cout << "Processing column #" << i << std::endl;

        timer.start();

        for (int E = 2; E <= E_max; E++) {
            Block block(ds, i, E, tau);
            std::vector<std::vector<std::pair<float, int>>> distances;

            block.calc_distances(distances);
        }

        timer.stop();

        std::cout << "Column #" << i << " processed in " << timer.elapsed()
                  << " [ms]" << std::endl;

        timer.reset();
    }

    return 0;
}
