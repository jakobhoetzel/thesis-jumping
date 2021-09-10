//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Core>
#include "BasicEigenTypes.hpp"

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    Yaml::Parse(cfg_, cfg);
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
    }

    setSeed(0);
//    setSeed(cfg_["seed"].template As<int>());

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    RSFATAL_IF(environments_[0]->getObDim() == 0 || environments_[0]->getActionDim() == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
  }

  void getRobotState(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->getRobotState(ob.row(i));
  }

  std::vector<std::string> getStepDataTag() {
    return environments_[0]->getStepDataTag();
  }

  int getStepData(int sample_size,
                  Eigen::Ref<EigenDoubleVec> &mean,
                  Eigen::Ref<EigenDoubleVec> &squareSum,
                  Eigen::Ref<EigenDoubleVec> &min,
                  Eigen::Ref<EigenDoubleVec> &max) {
    size_t data_size = getStepDataTag().size();
    if( data_size == 0 ) return sample_size;

    RSFATAL_IF(mean.size() != data_size ||
        squareSum.size() != data_size ||
        min.size() != data_size ||
        max.size() != data_size, "vector size mismatch")

    mean *= sample_size;

    for (int i = 0; i < num_envs_; i++) {
      mean += environments_[i]->getStepData();
      for (int j = 0; j < data_size; j++) {
        min(j) = std::min(min(j), environments_[i]->getStepData()[j]);
        max(j) = std::max(max(j), environments_[i]->getStepData()[j]);
      }
    }

    sample_size += num_envs_;
    mean /= sample_size;
    for (int i = 0; i < num_envs_; i++) {
      for (int j = 0; j < data_size; j++) {
        double temp = environments_[i]->getStepData()[j];
        squareSum[j] += temp * temp;
      }
    }

    return sample_size;
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    environments_[0]->setCommand(command);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return environments_[0]->getObDim(); }
  int getRobotStateDim() { return environments_[0]->getRobotStateDim(); }
  int getActionDim() { return environments_[0]->getActionDim(); }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate(int iter) {
    for (auto *env: environments_)
      env->curriculumUpdate(iter);
  };

  void printTest() { environments_[0]->printTest();}

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;

  int num_envs_ = 1;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
