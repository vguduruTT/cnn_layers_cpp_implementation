#include <vector>
void applyReLU(std::vector<std::vector<std::vector<float>>> &inputVector)
{
    for (auto &channel : inputVector)
    {
        for (auto &row : channel)
        {
            for (float &value : row)
            {
                if (value < 0)
                {
                    value = 0;
                }
            }
        }
    }
}
