#include <iostream>
#include <vector>
#include <limits> // Use <limits> for FLOAT_MIN
#include <random>
std::vector<std::vector<std::vector<float>>> maxPooling(const std::vector<std::vector<std::vector<float>>> &input, int kernelSize, int stride)
{
    int channels = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();

    int outputHeight = (inputHeight - kernelSize) / stride + 1;
    int outputWidth = (inputWidth - kernelSize) / stride + 1;

    std::vector<std::vector<std::vector<float>>> output(channels, std::vector<std::vector<float>>(outputHeight, std::vector<float>(outputWidth, 0.0f)));

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < outputHeight; ++i)
        {
            for (int j = 0; j < outputWidth; ++j)
            {
                float maxVal = -std::numeric_limits<float>::infinity(); // Use FLOAT_MIN for the minimum value
                for (int m = 0; m < kernelSize; ++m)
                {
                    for (int n = 0; n < kernelSize; ++n)
                    {
                        int inputRow = i * stride + m;
                        int inputCol = j * stride + n;
                        maxVal = std::max(maxVal, input[c][inputRow][inputCol]);
                    }
                }
                output[c][i][j] = maxVal;
            }
        }
    }

    return output;
}