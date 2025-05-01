//
// Created on 2025-05-01.
//

#ifndef CPPNN_SOFTPLUS_H
#define CPPNN_SOFTPLUS_H

#include "Activation.h"

namespace SpeedNet {
    class Softplus : public Activation {
    public:
        const double activate(double) override;
        const double derivative(double) override;
        const LayerDefinition::activationType type() override;
    };
}


#endif //CPPNN_SOFTPLUS_H