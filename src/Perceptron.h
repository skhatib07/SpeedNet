//
// Created by hunter harloff on 2019-12-01.
//

#ifndef CPPNN_PERCEPTRON_H
#define CPPNN_PERCEPTRON_H

#include <vector>
#include <sstream>

#include "activations/Activation.h"

namespace SpeedNet {
    class Perceptron {
    public:
        explicit Perceptron(int);

        void calculate(const std::vector<double> &, Activation *);

        double getValue() const;
        
        // Get the raw weighted sum before activation (needed for backpropagation)
        double getWeightedSum() const;
        
        // Update weights based on delta, inputs, and learning rate
        void updateWeights(double delta, const std::vector<double>& inputs, double learningRate);
        
        // Get the weights of this perceptron
        const std::vector<double>& getWeights() const;

        void mutate_uniform(const double, const double);

        void mutate_gaussian(const double, const double);

        friend std::ostream& operator<<(std::ostream& os, const Perceptron &p);

        friend std::istream& operator>>(std::istream& is, Perceptron &p);

    private:
        std::vector<double> weights;
        double cachedValue;
        double cachedWeightedSum; // Store the weighted sum for backpropagation
        int inputSize;
    };
}


#endif //CPPNN_PERCEPTRON_H
