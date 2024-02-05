#include <vector>
std::vector<std::vector<float>> denseLayer(const std::vector<std::vector<float>> &input,
                                           const std::vector<std::vector<float>> &weight,
                                           const std::vector<float> &bias)
{
    // Check if the input and weight matrices have compatible sizes
    if (input[0].size() != weight.size())
    {
        std::cerr << "Error: Matrix dimensions are not compatible for multiplication." << std::endl;
        exit(1);
    }

    // Initialize the resulting matrix with zeros
    std::vector<std::vector<float>> result(input.size(), std::vector<float>(bias.size(), 0.0));

    // Perform linear transformation with bias
    for (size_t i = 0; i < input.size(); ++i)
    {
        for (size_t j = 0; j < bias.size(); ++j)
        {
            result[i][j] = bias[j];
            for (size_t k = 0; k < weight.size(); ++k)
            {
                result[i][j] += input[i][k] * weight[k][j];
            }
        }
    }

    return result;
}
