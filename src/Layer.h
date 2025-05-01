//
// Created by Poppro on 12/2/2019.
//

#ifndef CPPNN_LAYER_H
#define CPPNN_LAYER_H

#include <vector>
#include <thread>

#include "Perceptron.h"
#include "activations/Activation.h"

namespace SpeedNet {
    class Layer {
    public:
        // Size of layer, size of previous layer (input to layer), activation function
        Layer(const int, const int, Activation *);

        std::vector<double> calculate(const std::vector<double> &);
        
        // Calculate the deltas (errors) for this layer during backpropagation
        std::vector<double> calculateDeltas(const std::vector<double>& nextLayerDeltas,
                                           const std::vector<std::vector<double>>& nextLayerWeights,
                                           const std::vector<double>& outputs);
        
        // Calculate the output layer deltas based on target values
        std::vector<double> calculateOutputDeltas(const std::vector<double>& targets,
                                                 const std::vector<double>& outputs);
        
        // Update the weights based on deltas, inputs, and learning rate
        void updateWeights(const std::vector<double>& deltas,
                          const std::vector<double>& inputs,
                          double learningRate);
        
        // Get the weights for a specific node (needed for backpropagation)
        std::vector<double> getWeights(int nodeIndex) const;

        void mutate_uniform(const double, const double);

        void mutate_gaussian(const double mean, const double std);

        const int size();
        
        // Get the number of inputs to this layer
        const int inputSize() const;

        friend std::ostream& operator<<(std::ostream& os, const Layer &l);

        friend std::istream& operator>>(std::istream& is, Layer &l);

    private:
        std::vector<Perceptron> nodes;
        Activation* activation;
    };
}


#endif //CPPNN_LAYER_H
