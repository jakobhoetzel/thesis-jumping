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
#include "AnymalController.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    world_ = std::make_unique<raisim::World>();
    auto* robot = world_->addArticulatedSystem(resourceDir + "/anymal/urdf/anymal.urdf");
    robot->setName("robot");
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    controller_.create(world_.get());
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, forwardVelRewardCoeff_, cfg["reward"]["forwardVelCoeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torqueCoeff"])

    stepDataTag_ = {"vel_rew", "joint_torque_rew"};
    stepData_.resize(stepDataTag_.size());

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

  void init() {}

  void reset() {
    controller_.reset(world_.get());
  }

  const std::vector<std::string>& getStepDataTag() {
    return stepDataTag_;
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    controller_.advance(world_.get(), action);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }

    return controller_.getReward(world_.get(), forwardVelRewardCoeff_, torqueRewardCoeff_);
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

  void curriculumUpdate() {};

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

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
  double terminalRewardCoeff_ = -10.;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  HubodogController controller_;
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};
}

