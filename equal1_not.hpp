#include <iostream>
#include <vector>
#include <cmath>

bool areVectorsEqual(const std::vector<std::vector<float>> &vec1,
                     const std::vector<std::vector<float>> &vec2)
{
    int precision = 3;
    if (vec1.size() != vec2.size() || vec1[0].size() != vec2[0].size())
    {
        return false; // Vectors must have the same size and shape
    }

    for (size_t i = 0; i < vec1.size(); ++i)
    {
        for (size_t j = 0; j < vec1[0].size(); ++j)
        {
            // Compare elements up to the specified precision
            if (std::abs(vec1[i][j] - vec2[i][j]) > std::pow(10, -precision))
            {
                return false; // Elements are not equal up to precision
            }
        }
    }

    return true; // All elements are equal up to precision
}