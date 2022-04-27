//
// Created by Poppro on 12/3/2019.
//

#ifndef SPEEDNET_RANDOMGENERATOR_H
#define SPEEDNET_RANDOMGENERATOR_H

#include <random>
#include <chrono>
#include <mutex>

namespace SpeedNet {
    class RandomGenerator {
    public:
        static RandomGenerator *getInstance();

        double generate_uniform(double lower, double upper);

        double generate_gaussian(double mean, double std);

    private:
        RandomGenerator();

        RandomGenerator(RandomGenerator const &);

        RandomGenerator &operator=(RandomGenerator const &);

    private:
        static RandomGenerator *randomGenerator;
        static std::mutex instanceMutex;
        std::default_random_engine *generator;
    };
}


#endif //SPEEDNET_RANDOMGENERATOR_H
