#include <iostream>
#include <fstream>
#include "json.hpp"
using json = nlohmann::json;
using namespace std;
#include "npy.hpp"
#include "conv.h"
#include "relu.h"
#include "fla.h"
#include "den.h"
#include <bits/stdc++.h>
#include "equal_not.hpp"
#include "equal1_not.hpp"
#include "pool.hpp"

int main()
{

    try
    {
        ofstream MyFile("final_output.txt");
        std::ifstream file("input.json");
        if (!file.is_open())
        {
            std::cerr << "Error opening the file.\n";
            return 1;
        }
        json data;
        file >> data;
        for (auto it = data.begin(); it != data.end(); ++it)
        {
            const auto &layerName = it.key();
            const auto &layerData = it.value();
            string layer = layerData["layer"];
            if (layer == "conv")
            {
                int shape = layerData["shape"];
                std::string filename = layerData["input"];
                std::string weight = layerData["weight"];
                std::string bias = layerData["bias"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];

                // getting output
                std::string path1{filename};
                npy::npy_data<float> d1 = npy::read_npy<float>(path1);
                std::vector<float> data1 = d1.data;
                std::vector<std::vector<std::vector<float>>> reshaped_data(3, std::vector<std::vector<float>>(shape, std::vector<float>(shape)));
                int index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < shape; ++j)
                    {
                        for (int k = 0; k < shape; ++k)
                        {
                            reshaped_data[i][j][k] = data1[index++];
                        }
                    }
                }
                // filters
                std::string path2{weight};
                npy::npy_data<float> d2 = npy::read_npy<float>(path2);
                std::vector<float> data2 = d2.data;
                // Reshape the vector to a 3x3x3x3 multi-dimensional vector
                std::vector<std::vector<std::vector<std::vector<float>>>> filters(3,
                                                                                  std::vector<std::vector<std::vector<float>>>(3,
                                                                                                                               std::vector<std::vector<float>>(3,
                                                                                                                                                               std::vector<float>(3, 0.0))));

                index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            for (int l = 0; l < 3; ++l)
                            {
                                filters[i][j][k][l] = data2[index++];
                            }
                        }
                    }
                }

                // bias
                std::string path3{bias};
                npy::npy_data<float> d3 = npy::read_npy<float>(path3);
                std::vector<float> data3 = d3.data;
                std::vector<float> bias1(3);
                for (int i = 0; i < 3; i++)
                {
                    bias1[i] = data3[i];
                }
                clock_t start, end;
                start = clock();

                vector<vector<vector<float>>> result1 = convolution(reshaped_data, filters, bias1);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                // storing first layer output
                std::string path4{py_output};
                npy::npy_data<float> d4 = npy::read_npy<float>(path4);
                std::vector<float> data4 = d4.data;
                std::vector<std::vector<std::vector<float>>> layer1_out(3, std::vector<std::vector<float>>(shape - 2, std::vector<float>(shape - 2)));

                index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < shape - 2; ++j)
                    {
                        for (int k = 0; k < shape - 2; ++k)
                        {
                            layer1_out[i][j][k] = data4[index++];
                        }
                    }
                }

                MyFile << "layer name: convolution\n";
                MyFile << "function exection: convolution\n";
                MyFile << "time exection:" << time_taken << "\n";
                bool as = areMatricesEqual(layer1_out, result1);
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }

                {
                    std::ofstream outputFile(cpp_output);

                    if (outputFile.is_open())
                    {
                        for (const auto &matrix : result1)
                        {
                            for (const auto &row : matrix)
                            {
                                for (const auto &element : row)
                                {
                                    outputFile << element << " ";
                                }
                            }
                        }
                        outputFile.close();
                    }
                    else
                    {
                        std::cerr << " opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
            else if (layer == "relu")
            {

                std::string filename = layerData["input"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];
                int shape = layerData["shape"];

                std::ifstream inputFile(filename);
                std::vector<std::vector<std::vector<float>>> layer2_input;

                if (inputFile.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile >> value)
                    {
                        values.push_back(value);
                    }
                    size_t rows = 3;
                    size_t cols = shape;
                    size_t depth = shape;

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer2_input.push_back({});
                        for (size_t j = 0; j < cols; ++j)
                        {
                            layer2_input[i].push_back({});
                            for (size_t k = 0; k < depth; ++k)
                            {
                                layer2_input[i][j].push_back(values[index++]);
                            }
                        }
                    }

                    inputFile.close();
                }
                else
                {
                    std::cerr << "Err opening the file for reading." << std::endl;
                    return 1;
                }
                clock_t start, end;
                start = clock();
                applyReLU(layer2_input);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                MyFile << "layer name: relu\n";
                MyFile << "function exection: applyrelu\n";
                MyFile << "time exection:" << time_taken << "\n";
                std::string path5{py_output};
                npy::npy_data<float> d5 = npy::read_npy<float>(path5);
                std::vector<float> data5 = d5.data;
                std::vector<std::vector<std::vector<float>>> layer2_out(3, std::vector<std::vector<float>>(shape, std::vector<float>(shape)));

                int index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < shape; ++j)
                    {
                        for (int k = 0; k < shape; ++k)
                        {
                            layer2_out[i][j][k] = data5[index++];
                        }
                    }
                }

                bool as = areMatricesEqual(layer2_input, layer2_out);
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }
                // storing the data
                {
                    std::ofstream outputFile(cpp_output);

                    if (outputFile.is_open())
                    {
                        for (const auto &matrix : layer2_input)
                        {
                            for (const auto &row : matrix)
                            {
                                for (const auto &element : row)
                                {
                                    outputFile << element << " ";
                                }
                            }
                        }
                        outputFile.close();
                    }
                    else
                    {
                        std::cerr << "Erro opening the file for writing." << std::endl;
                        return 1;
                    }
                }
                // std::cout << "completed relu\n";
            }
            else if (layer == "conv1")
            {
                int shape = layerData["shape"];
                std::string filename = layerData["input"];
                std::string weight = layerData["weight"];
                std::string bias = layerData["bias"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];

                std::ifstream inputFile1(filename);
                std::vector<std::vector<std::vector<float>>> layer3_input;

                if (inputFile1.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile1 >> value)
                    {
                        values.push_back(value);
                    }
                    size_t rows = 3;
                    size_t cols = shape;
                    size_t depth = shape;

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer3_input.push_back({});
                        for (size_t j = 0; j < cols; ++j)
                        {
                            layer3_input[i].push_back({});
                            for (size_t k = 0; k < depth; ++k)
                            {
                                layer3_input[i][j].push_back(values[index++]);
                            }
                        }
                    }

                    inputFile1.close();
                }
                else
                {
                    std::cerr << "2Error opening the file for reading." << std::endl;
                    return 1;
                }

                // filters
                std::string path6{weight};
                npy::npy_data<float> d6 = npy::read_npy<float>(path6);
                std::vector<float> data6 = d6.data;
                // Reshape the vector to a 3x3x3x3 multi-dimensional vector
                std::vector<std::vector<std::vector<std::vector<float>>>> filters6(3,
                                                                                   std::vector<std::vector<std::vector<float>>>(3,
                                                                                                                                std::vector<std::vector<float>>(3,
                                                                                                                                                                std::vector<float>(3, 0.0))));

                int index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            for (int l = 0; l < 3; ++l)
                            {
                                filters6[i][j][k][l] = data6[index++];
                            }
                        }
                    }
                }

                // bias
                std::string path7{bias};
                npy::npy_data<float> d7 = npy::read_npy<float>(path7);
                std::vector<float> data7 = d7.data;
                std::vector<float> bias7(3);
                for (int i = 0; i < 3; i++)
                {
                    bias7[i] = data7[i];
                }
                clock_t start, end;
                start = clock();
                vector<vector<vector<float>>> result2 = convolution(layer3_input, filters6, bias7);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

                // stroing output
                std::string path8{py_output};
                npy::npy_data<float> d8 = npy::read_npy<float>(path8);
                std::vector<float> data8 = d8.data;
                std::vector<std::vector<std::vector<float>>> layer3_out(3, std::vector<std::vector<float>>(shape - 2, std::vector<float>(shape - 2)));

                index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < shape - 2; ++j)
                    {
                        for (int k = 0; k < shape - 2; ++k)
                        {
                            layer3_out[i][j][k] = data8[index++];
                        }
                    }
                }

                MyFile << "layer name: convolution\n";
                MyFile << "function exection: convolution\n";
                MyFile << "time exection:" << time_taken << "\n";
                bool as = areMatricesEqual(layer3_out, result2);
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }

                // storing the data in txt file
                {
                    std::ofstream outputFile1(cpp_output);

                    if (outputFile1.is_open())
                    {
                        for (const auto &matrix : result2)
                        {
                            for (const auto &row : matrix)
                            {
                                for (const auto &element : row)
                                {
                                    outputFile1 << element << " ";
                                }
                            }
                        }
                        outputFile1.close();
                    }
                    else
                    {
                        std::cerr << "3Error opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
            else if (layer == "flatten")
            {
                std::string filename = layerData["input"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];
                int shape = layerData["shape"];

                std::ifstream inputFile5(filename);
                std::vector<std::vector<std::vector<float>>> layer7_input;

                if (inputFile5.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile5 >> value)
                    {
                        values.push_back(value);
                    }
                    size_t rows = 3;
                    size_t cols = shape;
                    size_t depth = shape;

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer7_input.push_back({});
                        for (size_t j = 0; j < cols; ++j)
                        {
                            layer7_input[i].push_back({});
                            for (size_t k = 0; k < depth; ++k)
                            {
                                layer7_input[i][j].push_back(values[index++]);
                            }
                        }
                    }

                    inputFile5.close();
                }
                else
                {
                    std::cerr << "4Error  the file for reading." << std::endl;
                    return 1;
                }
                clock_t start, end;
                start = clock();
                std::vector<float> flattenedOutput = flatten(layer7_input);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                std::vector<std::vector<float>> inp_cov(1, std::vector<float>(3 * shape * shape, 0.0));

                for (int j = 0; j < 3 * shape * shape; ++j)
                {
                    inp_cov[0][j] = flattenedOutput[j];
                }

                // storing output
                std::string path14{py_output};
                npy::npy_data<float> d14 = npy::read_npy<float>(path14);
                std::vector<float> data14 = d14.data;
                std::vector<std::vector<float>> layer7_out(1, std::vector<float>(3 * shape * shape, 0.0));

                for (int j = 0; j < shape * shape * 3; ++j)
                {
                    layer7_out[0][j] = data14[j];
                }
                MyFile << "\nlayer name: flatten\n";
                MyFile << "function exection: flattenedOutput\n";
                MyFile << "time exection:" << time_taken << "\n";
                bool as = areVectorsEqual(inp_cov, layer7_out);
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }

                {
                    std::ofstream outputFile7(cpp_output);

                    if (outputFile7.is_open())
                    {
                        for (const auto &matrix : inp_cov)
                        {

                            for (const auto &element : matrix)
                            {
                                outputFile7 << element << " ";
                            }
                        }
                        outputFile7.close();
                    }
                    else
                    {
                        std::cerr << "5Error opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
            else if (layer == "dense")
            {
                int shape = layerData["shape"];
                std::string filename = layerData["input"];
                std::string weight = layerData["weight"];
                std::string bias = layerData["bias"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];

                std::ifstream inputFile6(filename);
                std::vector<float> layer8_input;

                if (inputFile6.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile6 >> value)
                    {
                        values.push_back(value);
                    }

                    size_t rows = shape;
                    size_t cols = 1;

                    layer8_input.resize(rows);

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer8_input[i] = values[index++];
                    }

                    inputFile6.close();
                }
                else
                {
                    std::cerr << "6Error opening the file for reading." << std::endl;
                    return 1;
                }
                std::vector<std::vector<float>> pp(1, std::vector<float>(shape, 0.0));

                for (int j = 0; j < shape; ++j)
                {
                    pp[0][j] = layer8_input[j];
                }

                std::string path15{weight};
                npy::npy_data<float> d15 = npy::read_npy<float>(path15);
                std::vector<float> data15 = d15.data;

                std::vector<std::vector<float>> weight_den1(10, std::vector<float>(shape, 0.0));

                int index = 0;
                for (int i = 0; i < 10; ++i)
                {
                    for (int j = 0; j < shape; ++j)
                    {
                        weight_den1[i][j] = data15[index++];
                    }
                }
                std::vector<std::vector<float>> reshapedWeightDen1(shape, std::vector<float>(10, 0.0));

                for (int i = 0; i < 10; ++i)
                {
                    for (int j = 0; j < shape; ++j)
                    {
                        reshapedWeightDen1[j][i] = weight_den1[i][j];
                    }
                }

                // bias

                std::string path16{bias};
                npy::npy_data<float> d16 = npy::read_npy<float>(path16);
                std::vector<float> bias16 = d16.data;
                clock_t start, end;
                start = clock();
                std::vector<std::vector<float>> den_output = denseLayer(pp, reshapedWeightDen1, bias16);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

                // storing data
                std::string path17{py_output};
                npy::npy_data<float> d17 = npy::read_npy<float>(path17);
                std::vector<float> data17 = d17.data;
                std::vector<std::vector<float>> layer8_out(1, std::vector<float>(10, 0.0));

                for (int j = 0; j < 10; ++j)
                {
                    layer8_out[0][j] = data17[j];
                }
                bool as = areVectorsEqual(layer8_out, den_output);
                MyFile << "layer name: dense layer\n";
                MyFile << "function exection: denselayer\n";
                MyFile << "time exection:" << time_taken << "\n";
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }

                {
                    std::ofstream outputFile(cpp_output);

                    if (outputFile.is_open())
                    {
                        for (const auto &row : den_output)
                        {
                            for (const auto &element : row)
                            {
                                outputFile << element << " ";
                            }
                            outputFile << "\n";
                        }
                        outputFile.close();
                    }
                    else
                    {
                        std::cerr << "7Error opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
            else if (layer == "pool")
            {
                std::string filename = layerData["input"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];
                int shape = layerData["shape"];

                std::ifstream inputFile(filename);
                std::vector<std::vector<std::vector<float>>> layer2_input;

                if (inputFile.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile >> value)
                    {
                        values.push_back(value);
                    }
                    size_t rows = 3;
                    size_t cols = shape;
                    size_t depth = shape;

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer2_input.push_back({});
                        for (size_t j = 0; j < cols; ++j)
                        {
                            layer2_input[i].push_back({});
                            for (size_t k = 0; k < depth; ++k)
                            {
                                layer2_input[i][j].push_back(values[index++]);
                            }
                        }
                    }

                    inputFile.close();
                }
                else
                {
                    std::cerr << "8Error  the file for reading." << std::endl;
                    return 1;
                }
                clock_t start, end;
                int kernelSize = 2;
                int stride = 1;
                start = clock();
                std::vector<std::vector<std::vector<float>>> pooledOutput = maxPooling(layer2_input, kernelSize, stride);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                MyFile << "layer name: pool\n";
                MyFile << "function exection: maxpooling\n";
                MyFile << "time exection:" << time_taken << "\n";
                std::string path5{py_output};
                npy::npy_data<float> d5 = npy::read_npy<float>(path5);
                std::vector<float> data5 = d5.data;
                std::vector<std::vector<std::vector<float>>> layer2_out(3, std::vector<std::vector<float>>(shape - 1, std::vector<float>(shape - 1)));

                int index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < shape - 1; ++j)
                    {
                        for (int k = 0; k < shape - 1; ++k)
                        {
                            layer2_out[i][j][k] = data5[index++];
                        }
                    }
                }

                bool as = areMatricesEqual(pooledOutput, layer2_out);
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }
                // storing the data
                {
                    std::ofstream outputFile(cpp_output);

                    if (outputFile.is_open())
                    {
                        for (const auto &matrix : pooledOutput)
                        {
                            for (const auto &row : matrix)
                            {
                                for (const auto &element : row)
                                {
                                    outputFile << element << " ";
                                }
                            }
                        }
                        outputFile.close();
                    }
                    else
                    {
                        std::cerr << "9Error opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
            else if (layer == "dense1")
            {
                int shape = layerData["shape"];
                std::string filename = layerData["input"];
                std::string weight = layerData["weight"];
                std::string bias = layerData["bias"];
                std::string py_output = layerData["py_output"];
                std::string cpp_output = layerData["this_layer"];

                std::ifstream inputFile7(filename);
                std::vector<std::vector<float>> layer9_input;

                if (inputFile7.is_open())
                {
                    std::vector<float> values;
                    float value;

                    while (inputFile7 >> value)
                    {
                        values.push_back(value);
                    }

                    size_t rows = 1;
                    size_t cols = shape;

                    size_t index = 0;

                    for (size_t i = 0; i < rows; ++i)
                    {
                        layer9_input.push_back({});
                        for (size_t j = 0; j < cols; ++j)
                        {
                            layer9_input[i].push_back(values[index++]);
                        }
                    }

                    inputFile7.close();
                }
                else
                {
                    std::cerr << "Error opening the file for reading." << std::endl;
                    return 1;
                }
                std::string path18{weight};
                npy::npy_data<float> d18 = npy::read_npy<float>(path18);
                std::vector<float> data18 = d18.data;

                std::vector<std::vector<float>> weight_den2(10, std::vector<float>(10, 0.0));

                int index = 0;
                for (int i = 0; i < 10; ++i)
                {
                    for (int j = 0; j < 10; ++j)
                    {
                        weight_den2[i][j] = data18[index++];
                    }
                }
                std::vector<std::vector<float>> reshapedWeightDen2(10, std::vector<float>(10, 0.0));

                for (int i = 0; i < 10; ++i)
                {
                    for (int j = 0; j < 10; ++j)
                    {
                        reshapedWeightDen2[j][i] = weight_den2[i][j];
                    }
                }

                // bias

                std::string path19{bias};
                npy::npy_data<float> d19 = npy::read_npy<float>(path19);
                std::vector<float> bias19 = d19.data;
                clock_t start, end;
                start = clock();
                std::vector<std::vector<float>> den_output2 = denseLayer(layer9_input, reshapedWeightDen2, bias19);
                end = clock();
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

                // storing output
                std::string path20{py_output};
                npy::npy_data<float> d20 = npy::read_npy<float>(path20);
                std::vector<float> data20 = d20.data;
                std::vector<std::vector<float>> layer9_out(1, std::vector<float>(10, 0.0));

                for (int j = 0; j < 10; ++j)
                {
                    layer9_out[0][j] = data20[j];
                }
                bool as = areVectorsEqual(layer9_out, den_output2);
                MyFile << "\nlayer name: dense layer\n";
                MyFile << "function exection: denselayer\n";
                MyFile << "time exection:" << time_taken << "\n";
                if (as)
                {
                    MyFile << "pass\n\n\n";
                }
                else
                {
                    MyFile << "fail\n\n\n";
                }
                {
                    std::ofstream outputFile(cpp_output);

                    if (outputFile.is_open())
                    {
                        for (const auto &row : den_output2)
                        {
                            for (const auto &element : row)
                            {
                                outputFile << element << " ";
                            }
                            outputFile << "\n";
                        }
                        outputFile.close();
                    }
                    else
                    {
                        std::cerr << "Error opening the file for writing." << std::endl;
                        return 1;
                    }
                }
            }
        }
        MyFile.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
