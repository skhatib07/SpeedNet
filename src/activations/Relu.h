//
// Created by Poppro on 12/3/2019.
//

#ifndef SPEEDNET_RELU_H
#define SPEEDNET_RELU_H

#include "Activation.h"

namespace SpeedNet {
    class Relu : public Activation {
    public:
        const double activate(double) override;
        const LayerDefinition::activationType type() override;
    };
}


#endif //SPEEDNET_RELU_H