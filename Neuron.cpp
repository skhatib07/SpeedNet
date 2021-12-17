#include "Neuron.h"

Neuron::Neuron(){
    prevWeights = nullptr;
    /*nextWeights = nullptr;
    nextSize = 0;*/
    prevSize = 0;
}
Neuron::Neuron(Neuron* _prevWeights, int size){

    prevSize = size;

    prevWeights = new Neuron[prevSize];

    for(unsigned int i = 0; i < prevSize; i++){
        prevWeights[i] = _prevWeights[i];
    }

    //nextWeights = nullptr;
    //nextSize = 0;

    delete[] _prevWeights;
}

/*Neuron::Neuron(Neuron* _prevWeights, int _prevSize, Neuron* _nextWeights, int _nextSize){
    
    prevSize = _prevSize;
    prevWeights = new Neuron[prevSize];

    for(unsigned int i = 0; i < prevSize; i++){
        prevWeights[i] = _prevWeights[i];
    }

    nextSize = _nextSize;
    nextWeights = new Neuron[nextSize];

    for(unsigned int i = 0; i < nextSize; i++){
        nextWeights[i] = _nextWeights[i];
    }

    delete[] _prevWeights;
    delete[] _nextWeights;
}

*/

Neuron::Neuron(const Neuron& copy){

    prevSize = copy.prevSize;
    prevWeights = new Neuron[prevSize];

    for(unsigned int i = 0; i < prevSize; i++){
        prevWeights[i] = copy.prevWeights[i];
    }

   /* nextSize = copy.nextSize;
    nextWeights = new Neuron[nextSize];

    for(unsigned int i = 0; i < nextSize; i++){
        nextWeights[i] = copy.nextWeights[i];
    }
    */
}

Neuron& Neuron::operator=(const Neuron& copy){
    delete this;

    prevSize = copy.prevSize;
    prevWeights = new Neuron[prevSize];

    for(unsigned int i = 0; i < prevSize; i++){
        prevWeights[i] = copy.prevWeights[i];
    }

    /*nextSize = copy.nextSize;
    nextWeights = new Neuron[nextSize];

    for(unsigned int i = 0; i < nextSize; i++){
        nextWeights[i] = copy.nextWeights[i];
    }
    */
    return *this;
}


/*bool Neuron::setWeight(int idx, Neuron _weight){
    if(idx > nextSize - 1)
        return false;

    nextWeights[idx] = _weight;

    return true;
}*/

/*void Neuron::setWeights(Neuron* _weights, int newSize){

    delete[] nextWeights;

    nextWeights = new Neuron[newSize];

    for(unsigned int i = 0; i < newSize; i++){
        nextWeights[i] = _weights[i];
    }
    nextSize = newSize;
}*/

const float Neuron::getWeight(){
    float weight = 0.0;

    for(unsigned int i = 0; i < prevSize; i++)
        weight += prevWeights[i].getWeight();

    return weight;
}

//const Neuron* Neuron::getNextWeights(){return nextWeights;}
const Neuron* Neuron::getPrevWeights(){return prevWeights;}
//const int& Neuron::getNextSize(){return nextSize;}
const int& Neuron::getPrevSize(){return prevSize;}

Neuron::~Neuron(){
    delete[] prevWeights;
    //delete[] nextWeights;
}