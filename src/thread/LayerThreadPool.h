//
// Created by Poppro on 12/27/2020.
//

#ifndef SPEEDNET_LAYERTHREADPOOL_H
#define SPEEDNET_LAYERTHREADPOOL_H

#include <condition_variable>
#include <vector>
#include <queue>
#include <thread>

#include "../Perceptron.h"
#include "../activations/Activation.h"

namespace SpeedNet {
    class LayerThreadPool {
    public:
        static LayerThreadPool* getInstance();

        struct LayerQueueJob {
            const std::vector<double> &input;
            Perceptron& p;
            Activation* activation;
            int& track;
            LayerQueueJob(const std::vector<double> &input, Perceptron& p, Activation* activation, int& track) :
                input(input), p(p), activation(activation), track(track) {}
        };

        void addJob(const LayerQueueJob &job);

        void waitForTasks(int &track);

    private:
        LayerThreadPool();

        LayerThreadPool(LayerThreadPool const &);

        LayerThreadPool &operator=(LayerThreadPool const &);

    private:
        static LayerThreadPool* instance;
        static std::mutex instanceMutex;
        [[noreturn]] static void threadJob();
        std::vector<std::thread> pool;
        std::condition_variable jobDoneCv;
        std::condition_variable newJobCv;
        std::mutex trackMutex;
        std::mutex queueMutex;
        std::queue<LayerQueueJob> layerQueue;
    };
}


#endif //SPEEDNET_LAYERTHREADPOOL_H
