//
// Created by Poppro on 12/27/2020.
//

#include "LayerThreadPool.h"

namespace PyreNet {
    LayerThreadPool *LayerThreadPool::instance = nullptr;
    std::mutex LayerThreadPool::instanceMutex;

    PyreNet::LayerThreadPool *LayerThreadPool::getInstance() {
        std::unique_lock<std::mutex> lg(instanceMutex);
        if (!instance)
            instance = new LayerThreadPool;
        return instance;
    }

    void LayerThreadPool::addJob(const LayerThreadPool::LayerQueueJob &job) {
        std::unique_lock<std::mutex> lg(queueMutex);
        layerQueue.push(job);
        newJobCv.notify_one();
    }

    void LayerThreadPool::waitForTasks(int &track) {
        std::unique_lock<std::mutex> lg(trackMutex);
        jobDoneCv.wait(lg, [&track]() { return track <= 0; });
    }

    [[noreturn]] void LayerThreadPool::threadJob() {
        LayerThreadPool* tpi = getInstance();
        std::unique_lock<std::mutex> lg(tpi->queueMutex);
        while (true) {
            if (!tpi->layerQueue.empty()) {
                LayerQueueJob job = tpi->layerQueue.front();
                tpi->layerQueue.pop();
                lg.unlock();
                job.p.calculate(job.input, job.activation);
                std::lock_guard<std::mutex> lgt(tpi->trackMutex);
                job.track--;
                if (job.track == 0) {
                    tpi->jobDoneCv.notify_all();
                }
                lg.lock();
            } else {
                tpi->newJobCv.wait(lg);
            }
        }
    }

    PyreNet::LayerThreadPool::LayerThreadPool() {
        instance = this;
        for (int i = 0; i < 16; ++i) {
            pool.emplace_back(threadJob);
        }
    }
}
