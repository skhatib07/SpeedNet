//
// Created on 2025-05-01.
//

#ifndef CPPNN_ELU_H
#define CPPNN_ELU_H

#include "Activation.h"

namespace SpeedNet {
    class Elu : public Activation {
    public:
        const double activate(double) override;
        const double derivative(double) override;
        const LayerDefinition::activationType type() override;
        
    private:
        // Alpha parameter (typically 1.0)
        const double alpha = 1.0;
    };
}


#endif //CPPNN_ELU_H