//
// Created by Poppro on 12/3/2019.
//

#ifndef SPEEDNET_STEP_H
#define SPEEDNET_STEP_H

#include "Activation.h"

namespace SpeedNet {
    class Step : public Activation {
        const double activate(double) override;
        const LayerDefinition::activationType type() override;
    };
}


#endif //SPEEDNET_STEP_H
