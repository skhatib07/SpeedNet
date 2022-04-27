cmake_minimum_required(VERSION 3.15)
project(SpeedNet)

SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -pthread)
enable_testing()

ADD_EXECUTABLE(SpeedNetTest
        test/Test.cpp
        src/SpeedNet.h
        src/NeuralNet.cpp
        src/NeuralNet.h
        src/Layer.cpp
        src/Layer.h
        src/LayerDefinition.cpp
        src/LayerDefinition.h
        src/Perceptron.cpp
        src/Perceptron.h
        src/activations/Activation.cpp
        src/activations/Activation.h
        src/activations/ActivationFactory.cpp
        src/activations/ActivationFactory.h
        src/activations/Linear.cpp
        src/activations/Linear.h
        src/activations/Relu.cpp
        src/activations/Relu.h
        src/activations/Sigmoid.cpp
        src/activations/Sigmoid.h
        src/activations/Step.cpp
        src/activations/Step.h
        src/activations/Tanh.cpp
        src/activations/Tanh.h
        src/exceptions/InvalidInputSize.cpp
        src/exceptions/InvalidInputSize.h
        src/exceptions/InvalidLayer.cpp
        src/exceptions/InvalidLayer.h
        src/exceptions/InvalidNetworkSize.cpp
        src/exceptions/InvalidNetworkSize.h
        src/random/RandomGenerator.cpp
        src/random/RandomGenerator.h
        src/thread/LayerThreadPool.cpp
        src/thread/LayerThreadPool.h
        )

ADD_LIBRARY(SpeedNet STATIC
        src/SpeedNet.h
        src/NeuralNet.cpp
        src/NeuralNet.h
        src/Layer.cpp
        src/Layer.h
        src/LayerDefinition.cpp
        src/LayerDefinition.h
        src/Perceptron.cpp
        src/Perceptron.h
        src/activations/Activation.cpp
        src/activations/Activation.h
        src/activations/ActivationFactory.cpp
        src/activations/ActivationFactory.h
        src/activations/Linear.cpp
        src/activations/Linear.h
        src/activations/Relu.cpp
        src/activations/Relu.h
        src/activations/Sigmoid.cpp
        src/activations/Sigmoid.h
        src/activations/Step.cpp
        src/activations/Step.h
        src/activations/Tanh.cpp
        src/activations/Tanh.h
        src/exceptions/InvalidInputSize.cpp
        src/exceptions/InvalidInputSize.h
        src/exceptions/InvalidLayer.cpp
        src/exceptions/InvalidLayer.h
        src/exceptions/InvalidNetworkSize.cpp
        src/exceptions/InvalidNetworkSize.h
        src/random/RandomGenerator.cpp
        src/random/RandomGenerator.h
        src/thread/LayerThreadPool.cpp
        src/thread/LayerThreadPool.h)


ADD_TEST(System_Test
        ./SpeedNetTest
        )
