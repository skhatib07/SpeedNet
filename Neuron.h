#pragma once
#include <vector>
using namespace std;
class Neuron {
    // Activation for the current neuron on a scale from 0-1
    short activation;

    /* This array is used to represent the connections to the next node(s). These connections can be set
        using an entirely new array, or you can change one value using an index and a weight
    */
   // Neuron* nextWeights;
   // int nextSize;
    /* This array is set once at the start and exclusively used for calculating the weight of the node
        since the variables are node pointers, when you change the weight for one, if affects the calculated
        weights of the others
    */
    Neuron* prevWeights;
    int prevSize;


    public:
        Neuron();
        Neuron(Neuron* _prevWeights, int size);
       // Neuron(Neuron* _prevWeights, int _prevSize, Neuron* _nextWeights, int _nextSize);
        Neuron(const Neuron& copy);

        Neuron& operator=(const Neuron& copy);

        bool setWeight(int idx, Neuron _weight);
        void setWeights(Neuron* _weights, int newSize);
        

        // Getters for the variables inside each neuron. Varaible returned is a constant so cannot be edited
        const float getWeight();
        const Neuron* getNextWeights();
        const Neuron* getPrevWeights();
        const int& getNextSize();
        const int& getPrevSize();

        ~Neuron();

};