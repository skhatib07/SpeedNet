#ifndef CPPNN_LIBRARY_H
#define CPPNN_LIBRARY_H

#include <vector>

#include "Perceptron.h"
#include "LayerDefinition.h"
#include "Layer.h"
#include "activations/ActivationFactory.h"
#include "exceptions/InvalidInputSize.h"
#include "exceptions/InvalidLayer.h"
#include "exceptions/InvalidNetworkSize.h"

namespace SpeedNet {
    class NeuralNet {
    public:
        /* Empty network with no layers currently added - a minimum of 1 input layer,
         1 hidden layer, and 1 output layer are required to use the network. */
        NeuralNet(int);

        // # of input size, and layer topology (vector of layer sizes, includes output & middle)
        NeuralNet(int, const std::vector<LayerDefinition>&);

        // Load network from an istream
        NeuralNet(std::istream& is);

        // Predict output
        std::vector<double> predict(const std::vector<double>&);

        // Train the network using backpropagation
        // Returns the mean squared error of the training example
        double train(const std::vector<double>& input, const std::vector<double>& target, double learningRate);
        
        // Train the network using a batch of examples
        // Returns the average mean squared error across all examples
        double trainBatch(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          double learningRate,
                          int batchSize = 1);

        //Add new layer to network
        void add(const LayerDefinition &);

        // Mutate weightings in the neural net by an interval amount.
        // Applies to all layers by default.
        void mutate_uniform(const double lowerBound, const double upperBound, int layer = -1);

        // Mutate weightings based on normal distribution.
        // Applies to all layers by default.
        void mutate_gaussian(const double mean, const double std, int layer = -1);

        //size of public layers
        const int getInputSize();

        const int getOutputSize();

        friend std::ostream& operator<<(std::ostream& os, const NeuralNet& nn);

        friend std::istream& operator>>(std::istream& is, NeuralNet& nn);

    private:
        // Backpropagate the error through the network
        void backpropagate(const std::vector<double>& target, std::vector<std::vector<double>>& layerOutputs, double learningRate);
        
        // Update the weights based on the calculated gradients
        void updateWeights(const std::vector<std::vector<double>>& layerOutputs,
                           const std::vector<std::vector<double>>& deltas,
                           double learningRate);
        
        // Calculate the mean squared error between predicted and target values
        double calculateError(const std::vector<double>& predicted, const std::vector<double>& target);
        
        int inputSize;
        std::vector<Layer> layers;
    };
}
#endif