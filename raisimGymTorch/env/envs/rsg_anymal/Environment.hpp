//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include <stdlib.h>
#include <cstdint>
#include <set>
#include "AnymalController.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    world_ = std::make_unique<raisim::World>();
    robot_ = world_->addArticulatedSystem(resourceDir + "/anymal/urdf/anymal.urdf");
    robot_->setName("robot");
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    controller_.create(world_.get());
    READ_YAML(double, forwardVelRewardCoeff_, cfg["reward"]["forwardVelCoeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torqueCoeff"])

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot_);
    }
  }

  void init() {}

  void reset() {
    controller_.reset(world_.get());
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
    return controller_.isTerminalState(world_.get());
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

 private:
  bool visualizable_ = false;
  raisim::ArticulatedSystem *robot_;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  AnymalController controller_;
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
};
}

