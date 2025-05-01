//
// Created by Poppro on 12/2/2019.
//

#include "Layer.h"

#include <condition_variable>

#include "activations/ActivationFactory.h"
#include "thread/LayerThreadPool.h"

namespace SpeedNet {
// Constructor

    Layer::Layer(const int size, const int prevSize, Activation *activation) {
        this->nodes.reserve(size);
        for (int i = 0; i < size; i++) {
            this->nodes.emplace_back(prevSize);
        }
        this->activation = activation;
    }

// Main Layer Logic

    std::vector<double> Layer::calculate(const std::vector<double> &input) {
        LayerThreadPool* layerThreadPool = LayerThreadPool::getInstance();
        int track = this->nodes.size();
        for (Perceptron &p : this->nodes) {
            LayerThreadPool::LayerQueueJob job(input, p, this->activation, track);
            layerThreadPool->addJob(job);
        }
        layerThreadPool->waitForTasks(track);
        std::vector<double> ans;
        ans.reserve(this->nodes.size());
        for (const Perceptron &p : this->nodes) {
            ans.push_back(p.getValue());
        }
        return ans;
    }

// Mutators

    void Layer::mutate_uniform(const double lower, const double upper) {
        for (Perceptron &p : this->nodes) {
            p.mutate_uniform(lower, upper);
        }
    }

    void Layer::mutate_gaussian(const double mean, const double std) {
        for (Perceptron &p : this->nodes) {
            p.mutate_gaussian(mean, std);
        }
    }

// Getters

    const int Layer::size() {
        return this->nodes.size();
    }
    
    const int Layer::inputSize() const {
        if (this->nodes.empty()) {
            return 0;
        }
        // The input size is the size of the weights vector minus 1 (for bias)
        return this->nodes[0].getWeights().size() - 1;
    }
    
    std::vector<double> Layer::calculateOutputDeltas(const std::vector<double>& targets,
                                                    const std::vector<double>& outputs) {
        // Calculate deltas for output layer: delta = (target - output) * derivative(weighted_sum)
        std::vector<double> deltas;
        deltas.reserve(this->nodes.size());
        
        for (size_t i = 0; i < this->nodes.size(); i++) {
            double error = targets[i] - outputs[i];
            double weightedSum = this->nodes[i].getWeightedSum();
            double derivative = this->activation->derivative(weightedSum);
            deltas.push_back(error * derivative);
        }
        
        return deltas;
    }
    
    std::vector<double> Layer::calculateDeltas(const std::vector<double>& nextLayerDeltas,
                                              const std::vector<std::vector<double>>& nextLayerWeights,
                                              const std::vector<double>& outputs) {
        // Calculate deltas for hidden layer
        std::vector<double> deltas;
        deltas.reserve(this->nodes.size());
        
        for (size_t i = 0; i < this->nodes.size(); i++) {
            double weightedSum = this->nodes[i].getWeightedSum();
            double derivative = this->activation->derivative(weightedSum);
            
            // Calculate weighted sum of deltas from next layer
            double sum = 0.0;
            for (size_t j = 0; j < nextLayerDeltas.size(); j++) {
                // Use the weight from the next layer that connects to this node
                // nextLayerWeights[j][i] is the weight from node i in this layer to node j in the next layer
                sum += nextLayerDeltas[j] * nextLayerWeights[j][i];
            }
            
            deltas.push_back(sum * derivative);
        }
        
        return deltas;
    }
    
    void Layer::updateWeights(const std::vector<double>& deltas,
                             const std::vector<double>& inputs,
                             double learningRate) {
        // Update weights for each node in the layer
        for (size_t i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].updateWeights(deltas[i], inputs, learningRate);
        }
    }
    
    std::vector<double> Layer::getWeights(int nodeIndex) const {
        if (nodeIndex < 0 || nodeIndex >= this->nodes.size()) {
            // Return empty vector if index is out of bounds
            return std::vector<double>();
        }
        
        return this->nodes[nodeIndex].getWeights();
    }

// Serialize

    std::ostream &operator<<(std::ostream& os, const Layer &l) {
        os << ActivationFactory::toString(l.activation->type()) << " ";
        os << l.nodes.size() << " ";
        for (const Perceptron& p : l.nodes)
            os << p << " ";
        return os;
    }

    std::istream& operator>>(std::istream& is, Layer &l) {
        ActivationFactory* activationFactory = ActivationFactory::getInstance();
        std::string activationString;
        is >> activationString;
        l.activation = activationFactory->getActivation(ActivationFactory::fromString(activationString));
        int nodesSize;
        is >> nodesSize;
        l.nodes.resize(nodesSize, Perceptron(0));
        for (Perceptron& p : l.nodes)
            is >> p;
        return is;
    }
}
