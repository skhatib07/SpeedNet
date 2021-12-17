#pragma once 
#include "Neuron.h" 
#include <vector> 
using namespace std;
class Layer{

    Neuron* neurons;
    
    public:
        Layer(vector<Neuron*> _neurons);
        Layer(int numOfNodes);
    
        int numOfNodes();

        int ReLU(int num);

        short getActivation(int idx);


};