[![CI Status](https://github.com/skhatib07/SpeedNet/workflows/CMake/badge.svg?branch=main)](https://github.com/skhatib07/SpeedNet/actions?query=branch%3Amain)
# SpeedNet - A rework of PyreNet
## About
### Intended Use
In short, this is a C++ neural network static library developed as a simple, elegant, multi-purpose solution.

To be a bit more elaborate, this library offers a simple interface for home-cooked reinforcement based deep learning projects. It is optimized for running in a multi-threaded environment, seeking to offer performance and simple, essential, features without the complexity endured from larger-scale libraries. The library supports supervised learning through backpropagation.

View the original repository (PyreNet) [here](https://github.com/Poppro/PyreNet)

### Thread Safety
The library is designed to be fully thread-safe and can be used in multi-threaded environments.

## Quick Start

### Example Usage

```c++
#include "SpeedNet.h"

int main() {
  // Define middle and output layers
  std::vector<SpeedNet::LayerDefinition> layerDefs;
  layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::relu);  // Middle (50 nodes)
  layerDefs.emplace_back(50, SpeedNet::LayerDefinition::activationType::sigmoid);  // Middle (50 nodes)
  layerDefs.emplace_back(5, SpeedNet::LayerDefinition::activationType::relu);  // Output (5 nodes)

  // Initialize the network
  SpeedNet::NeuralNet nn(5, layerDefs);  // Defines the network to have an input size of 5
  nn.mutate_gaussian(0, 1);  // Mutates network weights from a gaussian sample with mean 0, standard deviation 1

  // Run a prediction on an input vector
  std::vector<double> predictions = nn.predict(std::vector<double>{0, 1, 2, 3, 4});
}
```

## Features

### Activations

| Activation Type          |    Identifier    | Description |
| ------------- | ------------- | ------------- |
| ReLU          |   LayerDefinition::relu   | Rectified Linear Unit: f(x) = max(0, x) |
| Linear          |   LayerDefinition::linear   | Linear activation: f(x) = x |
| Hyperbolic Tangent          |   LayerDefinition::tanh   | Hyperbolic tangent: f(x) = tanh(x) |
| Sigmoid          |   LayerDefinition::sigmoid   | Sigmoid function: f(x) = 1/(1+e^(-x)) |
| Step          |   LayerDefinition::step   | Step function: f(x) = 1 if x > 0, else 0 |
| Leaky ReLU    |   LayerDefinition::leakyRelu   | Leaky ReLU: f(x) = x if x > 0, else 0.01x |
| ELU           |   LayerDefinition::elu   | Exponential Linear Unit: f(x) = x if x > 0, else α(e^x-1) |
| Softplus      |   LayerDefinition::softplus   | Softplus: f(x) = ln(1 + e^x) |
| SELU          |   LayerDefinition::selu   | Scaled ELU: f(x) = λx if x > 0, else λα(e^x-1) |

### Training

SpeedNet now supports supervised learning through backpropagation, allowing you to train your neural networks on labeled data.

#### Backpropagation Overview

Backpropagation is a supervised learning algorithm used to train neural networks. It works by:

1. Forward propagating input data through the network to generate predictions
2. Calculating the error between predictions and target values
3. Backward propagating the error through the network to compute gradients
4. Updating weights based on these gradients to minimize the error

The algorithm uses the chain rule of calculus to efficiently compute how each weight in the network contributes to the overall error, allowing for precise adjustments to improve performance.

#### Single Example Training

The `train()` method allows you to train the network using a single input-target pair:

```c++
#include "SpeedNet.h"
#include <vector>

int main() {
  // Define network architecture
  std::vector<SpeedNet::LayerDefinition> layerDefs;
  layerDefs.emplace_back(10, SpeedNet::LayerDefinition::activationType::relu);
  layerDefs.emplace_back(5, SpeedNet::LayerDefinition::activationType::sigmoid);
  
  // Initialize network with 3 input nodes
  SpeedNet::NeuralNet nn(3, layerDefs);
  
  // Create a training example
  std::vector<double> input = {0.1, 0.5, 0.9};
  std::vector<double> target = {0.2, 0.4, 0.6, 0.8, 1.0};
  
  // Train the network with a learning rate of 0.01
  double error = nn.train(input, target, 0.01);
  
  // Print the resulting error
  std::cout << "Training error: " << error << std::endl;
  
  // Make a prediction with the trained network
  std::vector<double> prediction = nn.predict(input);
}
```

The `train()` method returns the mean squared error for the training example, which can be used to monitor the training progress.

#### Batch Training

For more efficient training, the `trainBatch()` method allows you to train on multiple examples at once:

```c++
#include "SpeedNet.h"
#include <vector>

int main() {
  // Define network architecture
  std::vector<SpeedNet::LayerDefinition> layerDefs;
  layerDefs.emplace_back(10, SpeedNet::LayerDefinition::activationType::relu);
  layerDefs.emplace_back(5, SpeedNet::LayerDefinition::activationType::sigmoid);
  
  // Initialize network with 3 input nodes
  SpeedNet::NeuralNet nn(3, layerDefs);
  
  // Create a batch of training examples
  std::vector<std::vector<double>> inputs = {
    {0.1, 0.5, 0.9},
    {0.2, 0.6, 0.3},
    {0.8, 0.1, 0.5},
    {0.4, 0.7, 0.2}
  };
  
  std::vector<std::vector<double>> targets = {
    {0.2, 0.4, 0.6, 0.8, 1.0},
    {0.1, 0.3, 0.5, 0.7, 0.9},
    {0.9, 0.7, 0.5, 0.3, 0.1},
    {0.5, 0.5, 0.5, 0.5, 0.5}
  };
  
  // Train the network with a learning rate of 0.01 and batch size of 4
  double avgError = nn.trainBatch(inputs, targets, 0.01, 4);
  
  // Print the average error across the batch
  std::cout << "Average batch error: " << avgError << std::endl;
}
```

The `trainBatch()` method returns the average mean squared error across all examples in the batch. Batch training is generally more efficient and can lead to more stable convergence compared to training on individual examples.

### Serialization
For convenience, all networks can easily be serialized and deserialized.
```c++
SpeedNet::NeuralNet nn(5, layerDefs)

ofstream ofs("output.txt");
ofs << nn;

ifstream ifs("output.txt");
ifs >> nn;
```

### Mutations
#### Gaussian
```c++
mutate_gaussian(mean, std, OptionalInt(layerIndex));
```
Mutates the weights via a gaussian distribution.

If the layerIndex field is specified, only that layer will be mutated.
Indexing starts from 0 at the first set of weights.

#### Uniform
```c++
mutate_uniform(lower_bound, upper_bound, OptionalInt(layerIndex));
```

Mutates the weights uniformly by a modifier in the range [lower_bound, upper_bound].


## Contributing

Feel free to make a pull request if you have any useful features or bug fixes.
For inquiries, contact samer@samerkhatib.com.

## Authors
* **Samer Khatib** - *Lead Developer of new SpeedNet Library* - [skhatib07](https://github.com/skhatib07)
* **Hunter Harloff** - *Lead Developer of original PyreNet Library* - [Poppro](https://github.com/Poppro)

## License

This project is licensed under the MIT License
