#ifndef __BLOCK_HPP__
#define __BLOCK_HPP__

#include <cmath>
#include <iostream>
#include <vector>

#include "dataset.hpp"

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

#endif
