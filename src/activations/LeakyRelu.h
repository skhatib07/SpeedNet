//
// Created on 2025-05-01.
//

#ifndef CPPNN_LEAKYRELU_H
#define CPPNN_LEAKYRELU_H

#include "Activation.h"

namespace SpeedNet {
    class LeakyRelu : public Activation {
    public:
        const double activate(double) override;
        const double derivative(double) override;
        const LayerDefinition::activationType type() override;
        
    private:
        // Alpha parameter for leaky part (typically 0.01)
        const double alpha = 0.01;
    };
}


#endif //CPPNN_LEAKYRELU_H