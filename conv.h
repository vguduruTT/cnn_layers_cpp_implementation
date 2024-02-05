#include <iostream>
#include <vector>
using namespace std;
vector<vector<vector<float>>> convolution(const vector<vector<vector<float>>> &input, const vector<vector<vector<vector<float>>>> &filters, const vector<float> &bias)
{
    int inputChannels = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();

    int filterCount = filters.size();
    int filterChannels = filters[0].size();
    int filterSize = filters[0][0].size();

    int outputHeight = inputHeight - filterSize + 1;
    int outputWidth = inputWidth - filterSize + 1;

    vector<vector<vector<float>>> output(filterCount, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0.0)));

    // Perform convolution and add bias
    for (int f = 0; f < filterCount; ++f)
    {
        for (int i = 0; i < outputHeight; ++i)
        {
            for (int j = 0; j < outputWidth; ++j)
            {
                for (int c = 0; c < filterChannels; ++c)
                {
                    for (int m = 0; m < filterSize; ++m)
                    {
                        for (int n = 0; n < filterSize; ++n)
                        {
                            output[f][i][j] += input[c][i + m][j + n] * filters[f][c][m][n];
                        }
                    }
                }

                // Add bias
                output[f][i][j] += bias[f];
            }
        }
    }

    return output;
}
