#include <vector>

std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>> &input)
{
    std::vector<float> flattened;

    for (const auto &depthSlice : input)
    {
        for (const auto &row : depthSlice)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    return flattened;
}