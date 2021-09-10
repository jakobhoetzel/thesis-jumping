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
#include "MinicheetahController.hpp"
#include "RandomHeightMapGenerator.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) : //resourceDir = .../rsc
  visualizable_(visualizable) {
    /// add objects
    world_ = std::make_unique<raisim::World>();
    auto* robot = world_->addArticulatedSystem(resourceDir + "/mini_cheetah/mini-cheetah-vision-v1.5.urdf");
    robot->setName("robot");
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    mu_ = 0.4 + 0.3 * (uniDist_(gen_) + 1);  // [0.4, 1.0]  // should be corrected also in reset method.
//    mu_ = 0.6;
    world_->setDefaultMaterial(mu_, 0, 0);

    controller_.create(world_.get());
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, rewCurriculumFactor_, cfg["rew_curriculum_factor"])
    READ_YAML(double, rewCurriculumRate_, cfg["rew_curriculum_rate"])
    READ_YAML(double, comCurriculumFactor1_, cfg["com_curriculum_factor1"])
    READ_YAML(double, comCurriculumFactor2_, cfg["com_curriculum_factor2"])
    READ_YAML(double, comCurriculumFactor3_, cfg["com_curriculum_factor3"])
    READ_YAML(double, terCurriculumFactor_, cfg["ter_curriculum_factor"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::ANGULARVELOCIY1], cfg["reward"]["bodyAngularVelCoeff1"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::VELOCITY1], cfg["reward"]["forwardVelCoeff1"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::JOINTSPEED], cfg["reward"]["jointSpeedCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::TORQUE], cfg["reward"]["torqueCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::FOOTSLIP], cfg["reward"]["footSlipCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::ORIENTATION], cfg["reward"]["bodyOriCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::SMOOTHNESS1], cfg["reward"]["smoothnessCoeff1"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::SMOOTHNESS2], cfg["reward"]["smoothnessCoeff2"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::AIRTIME], cfg["reward"]["airTimeCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::JOINTPOS], cfg["reward"]["jointPosCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::JOINTACC], cfg["reward"]["jointAccCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::BASEMOTION], cfg["reward"]["baseMotionCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::FOOTCLEARANCE], cfg["reward"]["footClearanceCoeff"])

    isHeightMap_ = cfg["isHeightMap"].template As<bool>();
    controller_.setIsHeightMap(isHeightMap_);
    if (isHeightMap_){
      heightMap_ = terrainGenerator_.generateTerrain(world_.get(), RandomHeightMapGenerator::GroundType(groundType_), 0.0, false, gen_, uniDist_);
    }
    else {
      world_->addGround();
    }

    stepData_.resize(controller_.getStepDataTag().size());

    /// visualize if it is the first environment
    if (visualizable_) {  //RaisimUnity
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot);
    }
  }

  ~ENVIRONMENT() {
    if(server_) server_->killServer();
  }

  void init() {
  }

  void reset() {
    controller_.reset(world_.get(), comCurriculumFactorT_, heightMap_);
    controller_.collisionRandomization(world_.get());

    mu_ = 0.4 + 0.3 * (uniDist_(gen_) + 1);  // [0.4, 1.0]
//    mu_ = 0.6;
    world_->setDefaultMaterial(mu_, 0, 0);
  }

  const std::vector<std::string>& getStepDataTag() {
    return controller_.getStepDataTag();
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) { controller_.setCommand(command); }

  double step(const Eigen::Ref<EigenVec> &action) {
//    controller_.advance(world_.get(), action); ///make sure default pd target is not nan
    stepData_.setZero();
    int loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
//    delayDividedBySimdt = int((0.01 / simulation_dt_)*0.5*(uniDist_(gen_)+1));
    delayDividedBySimdt = int((0.01 / simulation_dt_) - 1e-10);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(i == delayDividedBySimdt){
        controller_.advance(world_.get(), action);
      }
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();  // What does integration do? A. Simulate robot states and motions for the next simulation time.
      if (server_) server_->unlockVisualizationServerMutex();
      controller_.getReward(world_.get(), rewardCoeff_, simulation_dt_, rewCurriculumFactor_, heightMap_);
      stepData_ += controller_.getStepData();
    }

    controller_.updateHistory();  /// update every control_dt
    controller_.updatePreviousActions();  /// update every control_dt

    stepData_ /= loopCount;

    double totalRewardSum = stepData_.tail(3)(2);
    return totalRewardSum;
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    ob = controller_.getObservation().cast<float>();
  }

  void unObsState(Eigen::Ref<EigenVec> ob) {
    ob = controller_.getUnobservableStates(mu_, heightMap_).cast<float>();
  }

  bool isTerminalState(float &terminalReward) {
    if(controller_.isTerminalState(world_.get())) {
      terminalReward = terminalRewardCoeff_;
      return true;
    }
    terminalReward = 0.f;
    return false;
  }
  /////// optional methods ///////
  void curriculumUpdate(int iter) {
    rewCurriculumFactor_ = pow(rewCurriculumFactor_, rewCurriculumRate_);
    comCurriculumFactorT_ = 1 + comCurriculumFactor3_ / (1 + std::exp(-comCurriculumFactor1_ * (iter - comCurriculumFactor2_)));
    comCurriculumFactorT_ = std::fmax(1., comCurriculumFactorT_);

    if(isHeightMap_) {
      groundType_ = (groundType_+1) % 2;
      world_->removeObject(heightMap_);
      double terrain_curriculum = 1 * std::min(1., iter / terCurriculumFactor_);
      heightMap_ = terrainGenerator_.generateTerrain(world_.get(), RandomHeightMapGenerator::GroundType(groundType_), terrain_curriculum, false, gen_, uniDist_);
    }
  };
  float getCurriculumFactor() {return float(rewCurriculumFactor_);};
  void close() { if (server_) server_->killServer(); };
  void setSeed(int seed) {
    controller_.setSeed(seed);
    terrainGenerator_.setSeed(seed);
    groundType_ = seed % 2;
  };
  ////////////////////////////////

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getUnObsDim() { return controller_.getUnObsDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return world_.get(); }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  void printTest() { controller_.printTest();}

 private:
  std::map<MinicheetahController::RewardType, float> rewardCoeff_;
  bool visualizable_ = false;
  bool isHeightMap_;
  double terminalRewardCoeff_ = -10.;
  MinicheetahController controller_;
  std::unique_ptr<raisim::World> world_;
  double rewCurriculumFactor_, rewCurriculumRate_;
  double comCurriculumFactorT_ = 1., comCurriculumFactor1_, comCurriculumFactor2_, comCurriculumFactor3_;
  double terCurriculumFactor_;
  double simulation_dt_;
  double control_dt_;
  double mu_;
  int groundType_ = 0;
  int delayDividedBySimdt;
  std::unique_ptr<raisim::RaisimServer> server_;
  Eigen::VectorXd stepData_;
  RandomHeightMapGenerator terrainGenerator_;
  raisim::HeightMap* heightMap_;
  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}