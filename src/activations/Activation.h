//
// Created by hunter harloff on 2019-12-01.
//

#ifndef CPPNN_ACTIVATION_H
#define CPPNN_ACTIVATION_H

#include "../LayerDefinition.h"

namespace SpeedNet {
    class Activation {
    public:
        virtual const double activate(double) = 0;
        
        // Derivative of the activation function with respect to its input
        // This is needed for backpropagation
        virtual const double derivative(double) = 0;
        
        virtual const LayerDefinition::activationType type() = 0;
    };
}


#endif //CPPNN_ACTIVATION_H
