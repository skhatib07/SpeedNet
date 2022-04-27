//
// Created by Poppro on 12/3/2019.
//

#include <iostream>
#include "../src/SpeedNet.h"

void layerAddTest(){
    SpeedNet::NeuralNet nn(5);
    nn.add(SpeedNet::LayerDefinition(50, SpeedNet::LayerDefinition::activationType::relu));
    nn.add(SpeedNet::LayerDefinition(50, SpeedNet::LayerDefinition::activationType::relu));
    nn.add(SpeedNet::LayerDefinition(50, SpeedNet::LayerDefinition::activationType::relu));
    nn.add(SpeedNet::LayerDefinition(2, SpeedNet::LayerDefinition::activationType::relu));
    nn.mutate_uniform(0, 0.1);
    std::vector<double> input{0,1,2,3,4};
    std::vector<double> output = nn.predict(input);
    std::cout << "Individual Layer add test complete" << std::endl;
}

void stressTest() {
    std::vector<SpeedNet::LayerDefinition> layerDefs;
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(2, SpeedNet::LayerDefinition::activationType::relu);
    SpeedNet::NeuralNet nn(5, layerDefs);
    nn.mutate_uniform(0, 0.1);
    std::vector<double> input{0,1,2,3,4};
    std::vector<double> output = nn.predict(input);
    std::cout << "Stress test complete" << std::endl;
}

void multiThreadedStressTest() {
    int threadCount = 10;
    std::vector<SpeedNet::LayerDefinition> layerDefs;
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(2, SpeedNet::LayerDefinition::activationType::relu);
    std::vector<SpeedNet::NeuralNet> nns;
    for (int i = 0; i < threadCount; ++i) {
        nns.emplace_back(5, layerDefs);
        nns.back().mutate_uniform(0, 1);
    }
    std::vector<double> input{0,1,2,3,4};

    std::vector<std::thread> pool;
    for (int i = 0; i < threadCount; ++i)
        pool.emplace_back(&SpeedNet::NeuralNet::predict, nns[i], input);

    for (std::thread& t : pool)
        t.join();

    std::cout << "Multi threaded stress test complete" << std::endl;
}

void serializeTest() {
    std::vector<SpeedNet::LayerDefinition> layerDefs;
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);
    layerDefs.emplace_back(2, SpeedNet::LayerDefinition::activationType::relu);
    SpeedNet::NeuralNet nn(5, layerDefs);
    nn.mutate_uniform(0,2);
    std::vector<double> input{0,1,2,3,4};
    double output = nn.predict(input)[0];

    std::stringstream ss;
    ss << nn;

    nn.mutate_uniform(2,2);
    if (std::abs(nn.predict(input)[0] - output) <= 1)
        exit(EXIT_FAILURE);

    ss >> nn;
    if (std::abs(nn.predict(input)[0] - output) > 1)
        exit(EXIT_FAILURE);

    ss << nn;
    SpeedNet::NeuralNet nn2(ss);
    if (std::abs(nn2.predict(input)[0] - output) > 1)
        exit(EXIT_FAILURE);

    std::cout << "Serialize test complete" << std::endl;
}

int main() {
    layerAddTest();
    stressTest();
    multiThreadedStressTest();
    serializeTest();
    return 0;
}
