//
// Created on 2025-05-01.
//

#ifndef CPPNN_SELU_H
#define CPPNN_SELU_H

#include "Activation.h"

namespace SpeedNet {
    class Selu : public Activation {
    public:
        const double activate(double) override;
        const double derivative(double) override;
        const LayerDefinition::activationType type() override;
        
    private:
        // SELU parameters (from the paper)
        // "Self-Normalizing Neural Networks" by Klambauer et al.
        const double alpha = 1.6732632423543772848170429916717;
        const double scale = 1.0507009873554804934193349852946;
    };
}


#endif //CPPNN_SELU_H