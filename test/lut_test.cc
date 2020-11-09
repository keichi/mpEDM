#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/lut.h"

TEST_CASE("Normalize lookup table", "[lut][cpu]")
{
    const auto input =
        uninitialized_vector<float>({0.74278091, 0.78794577, 1.20091218, //
                                     0.73450598, 0.85545997, 1.19310228, //
                                     0.78794577, 0.79144452, 1.17882891, //
                                     0.78722765, 0.85545997, 1.15747635, //
                                     0.74278091, 0.79144452, 0.80511738, //
                                     0.73450598, 0.78722765, 0.80511738});

    const auto indices = uninitialized_vector<uint32_t>({0, 1, 2, //
                                                         0, 1, 2, //
                                                         0, 1, 2, //
                                                         0, 1, 2, //
                                                         0, 1, 2, //
                                                         0, 1, 2});

    const auto normalized =
        uninitialized_vector<float>({0.403114158, 0.379333097, 0.217552745, //
                                     0.419502817, 0.355809796, 0.224687387, //
                                     0.383953331, 0.382252226, 0.233794442, //
                                     0.393425348, 0.360761525, 0.245813127, //
                                     0.350129443, 0.327925843, 0.321944713, //
                                     0.352226913, 0.327830661, 0.319942426});

    LUT lut(6, 3, input, indices);

    lut.normalize();

    for (auto i = 0u; i < lut.n_columns() * lut.n_rows(); i++) {
        REQUIRE(lut.distances[i] == Approx(normalized[i]));
    }
}
