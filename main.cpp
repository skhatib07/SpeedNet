#include <iostream>
#include <vector>
#include "Neuron.h"
using namespace std;

int main()
{
    Neuron* A = new Neuron();
    Neuron* B = new Neuron(A,1);
}