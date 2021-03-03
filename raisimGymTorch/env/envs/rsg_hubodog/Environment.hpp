// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

// raisimGymTorch include
#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "HubodogController.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    world_ = std::make_unique<raisim::World>();
    auto* robot = world_->addArticulatedSystem(resourceDir + "/hubodog/hubodog.urdf");
    robot->setName("robot");
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    controller_.create(world_.get());

    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::VELOCITY], cfg["reward"]["velocity"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::JOINT_POSITION], cfg["reward"]["joint_position"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::JOINT_VELOCITY], cfg["reward"]["joint_velocity"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::JOINT_ACCELERATION], cfg["reward"]["joint_acceleration"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::SMOOTHNESS1], cfg["reward"]["smoothness1"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::SMOOTHNESS2], cfg["reward"]["smoothness2"])
    READ_YAML(double, rewardCoeff_[HubodogController::RewardType::AIRTIME], cfg["reward"]["airtime"])

    stepData_.resize(controller_.getStepDataTag().size());

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot);
    }
  }

  ~ENVIRONMENT() {
    if(server_) server_->killServer();
  }

  const std::vector<std::string>& getStepDataTag() {
    return controller_.getStepDataTag();
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

  void init() {}

  void reset() {
    controller_.reset(world_.get(), curriculumFactor_);
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    controller_.advance(world_.get(), action);
    float reward = 0;
    stepData_.setZero();

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
      reward += controller_.getReward(world_.get(), rewardCoeff_, simulation_dt_, curriculumFactor_);
      stepData_ += controller_.getStepData();

      if(i % 5 == 0) controller_.updateHistory();
    }

    return reward;
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    ob = controller_.getObservation().cast<float>();
  }

  bool isTerminalState(float &terminalReward) {
    if(controller_.isTerminalState(world_.get())) {
      terminalReward = terminalRewardCoeff_;
      return true;
    }
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {
    curriculumFactor_ *= 0.999;
  };

  float getCurriculumFactor() {
    return float(curriculumFactor_);
  };

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {
    controller_.setSeed(seed);
  };

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return world_.get(); }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  void printTest() {}


 private:
  bool visualizable_ = false;
  std::map<HubodogController::RewardType, float> rewardCoeff_;
  double terminalRewardCoeff_ = -0.;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  double curriculumFactor_ = 1.;
  HubodogController controller_;
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  Eigen::VectorXd stepData_;
};
}

