#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
                // Read header
                if (n_rows == 0) {
                    cols.push_back(std::vector<float>());
                    n_cols++;
                    continue;
                }

                // Read body
                cols[i].push_back(std::stof(cell));
            }
        }
    }
};

#endif
