//
// Created by hunter harloff on 2019-12-01.
//

#ifndef CPPNN_SIGMOID_H
#define CPPNN_SIGMOID_H

#include "Activation.h"

namespace SpeedNet {
    class Sigmoid : public Activation {
    public:
        const double activate(double) override;
        const LayerDefinition::activationType type() override;
    };
}


#endif //CPPNN_SIGMOID_H
