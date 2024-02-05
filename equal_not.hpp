#include <iostream>
#include <vector>
#include <cmath>
bool areMatricesEqual(const std::vector<std::vector<std::vector<float>>> &mat1,
                      const std::vector<std::vector<std::vector<float>>> &mat2)
{
    int precision = 3;
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size() || mat1[0][0].size() != mat2[0][0].size())
    {
        return false; // Matrices must have the same dimensions
    }

    for (size_t i = 0; i < mat1.size(); ++i)
    {
        for (size_t j = 0; j < mat1[0].size(); ++j)
        {
            for (size_t k = 0; k < mat1[0][0].size(); ++k)
            {
                // Compare elements up to the specified precision
                if (std::abs(mat1[i][j][k] - mat2[i][j][k]) > std::pow(10, -precision))
                {
                    return false;
                }
            }
        }
    }

    return true;
}