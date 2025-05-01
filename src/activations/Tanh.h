//
// Created by Poppro on 12/3/2019.
//

#ifndef SPEEDNET_TANH_H
#define SPEEDNET_TANH_H

#include "Activation.h"

namespace SpeedNet {
    class Tanh : public Activation {
    public:
        const double activate(double) override;
        const double derivative(double) override;
        const LayerDefinition::activationType type() override;
    };
}


#endif //SPEEDNET_TANH_H
