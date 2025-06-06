//
// Created by Poppro on 12/2/2019.
//

#ifndef CPPNN_LAYERDEFINITION_H
#define CPPNN_LAYERDEFINITION_H

namespace SpeedNet {
    class LayerDefinition {
    public:
        enum activationType {
            step,
            linear,
            sigmoid,
            tanh,
            relu,
            leakyRelu,
            elu,
            softplus,
            selu
        };

        // Constructor to be supplied layer size, desired activation function
        LayerDefinition(int, LayerDefinition::activationType);

    private:
        activationType activation;
        int size;
        friend class NeuralNet;
    };
}

#endif //CPPNN_LAYERDEFINITION_H
