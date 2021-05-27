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

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) : //resourceDir = .../rsc
      visualizable_(visualizable) {
    /// add objects
    world_ = std::make_unique<raisim::World>();
    auto* robot = world_->addArticulatedSystem(resourceDir + "/mini_cheetah/mini-cheetah-vision-v1.3.urdf");
    robot->setName("robot");
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    controller_.create(world_.get());
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, curriculumFactor_, cfg["curriculum_factor"])
    READ_YAML(double, curriculumRate_, cfg["curriculum_rate"])
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
    delayDevidedBySimdt = 0;// int((control_dt_ / simulation_dt_ + 1e-10)*uniDist_(gen_));
    /// Test code for checking RotorLocation /
//    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world_->getObject("robot"));
//
//    // Debug positions
//    Eigen::Vector3d bodyCOM;
//    Eigen::Vector3d _abadRotorLocation, _hipRotorLocation, _kneeRotorLocation;
//    Eigen::Vector3d _abadLocation, _hipLocation, _kneeLocation;
//    Eigen::Vector3d abadCOM, hipCOM, kneeCOM;
////    raisim::Vec<3> urdf_abdaCOM, urdf_hipCOM, urdf_kneeCOM;
//    Eigen::Vector3d urdf_abad_joint, urdf_hip_joint, urdf_knee_joint;
//    Eigen::Vector3d urdf_abadCOM, urdf_hipCOM, urdf_kneeCOM;
//
//    bodyCOM = {0, 0, 0.4035};
//    urdf_abad_joint = {0.14775, 0.049, 0.0};
//    urdf_hip_joint = {0.055, 0.019, 0.00};
//    urdf_knee_joint = {0.0, 0.049, -0.2085};
//
//    _abadRotorLocation = bodyCOM + Eigen::Vector3d({0.14775, 0.049, 0});
//    _abadLocation = bodyCOM + Eigen::Vector3d({0.20275, 0.049, 0});
//    abadCOM = _abadLocation + Eigen::Vector3d({0, 0.0, 0});
//
//    _hipRotorLocation = _abadLocation + Eigen::Vector3d({0, 0.019, 0});
//    _hipLocation = _abadLocation + Eigen::Vector3d({0, 0.068, 0});
//    hipCOM = _hipLocation + Eigen::Vector3d({0, -0.017, -0.02});
//
//    _kneeRotorLocation = _hipLocation + Eigen::Vector3d({0, 0., 0});
//    _kneeLocation = _hipLocation + Eigen::Vector3d({0, 0., -0.2085});
//    kneeCOM = _kneeLocation + Eigen::Vector3d({0, 0, -0.061});
//
////    cheetah->getFramePosition(cheetah->getFrameIdxByName("torso_to_abduct_fl_j"), urdf_abdaCOM);
//    urdf_abadCOM = bodyCOM + urdf_abad_joint + Eigen::Vector3d({0.055, 0., 0.});
//
////    cheetah->getFramePosition(cheetah->getFrameIdxByName("abduct_fl_to_thigh_fl_j"), urdf_hipCOM);
//    urdf_hipCOM = bodyCOM + urdf_abad_joint + urdf_hip_joint + Eigen::Vector3d({0.0, 0.032, -0.02});
//
////    cheetah->getFramePosition(cheetah->getFrameIdxByName("thigh_fl_to_knee_fl_j"), urdf_kneeCOM);
//    urdf_kneeCOM = bodyCOM + urdf_abad_joint + urdf_hip_joint + urdf_knee_joint + Eigen::Vector3d({0.0, 0.0, -0.061});
//
//    // debug sphere
//    auto debugSphere_abadRotorLocation = server_->addVisualSphere("debug_sphere_abadRotorLocation", 0.1);
//    debugSphere_abadRotorLocation->setColor(1,0,0,1);
//    debugSphere_abadRotorLocation->setPosition(_abadRotorLocation);
//
////    auto debugSphere_abadLocation = server_->addVisualSphere("debug_sphere_abadLocation", 0.1);
////    debugSphere_abadLocation->setColor(0,1,0,1);
////    debugSphere_abadLocation->setPosition(_abadLocation);
//
//    auto debugSphereabadCOM = server_->addVisualSphere("debug_sphereabadCOM", 0.1);
//    debugSphereabadCOM->setColor(0,0,1,1);
//    debugSphereabadCOM->setPosition(abadCOM);
//
//
//    auto debugSphere_hipRotorLocation = server_->addVisualSphere("debug_sphere_hipRotorLocation", 0.1);
//    debugSphere_hipRotorLocation->setColor(1,0,0,1);
//    debugSphere_hipRotorLocation->setPosition(_hipRotorLocation);
//
////    auto debugSphere_hipLocation = server_->addVisualSphere("debug_sphere_hipLocation", 0.1);
////    debugSphere_hipLocation->setColor(0,1,0,1);
////    debugSphere_hipLocation->setPosition(_hipLocation);
//
//    auto debugSpherehipCOM = server_->addVisualSphere("debug_spherehipCOM", 0.1);
//    debugSpherehipCOM->setColor(0,0,1,1);
//    debugSpherehipCOM->setPosition(hipCOM);
//
//
//    auto debugSphere_kneeRotorLocation = server_->addVisualSphere("debug_sphere_kneeRotorLocation", 0.1);
//    debugSphere_kneeRotorLocation->setColor(1,0,0,1);
//    debugSphere_kneeRotorLocation->setPosition(_kneeRotorLocation);
//
////    auto debugSphere_kneeLocation = server_->addVisualSphere("debug_sphere_kneeLocation", 0.1);
////    debugSphere_kneeLocation->setColor(0,1,0,1);
////    debugSphere_kneeLocation->setPosition(_kneeLocation);
//
//    auto debugSpherekneeCOM = server_->addVisualSphere("debug_spherekneeCOM", 0.1);
//    debugSpherekneeCOM->setColor(0,0,1,1);
//    debugSpherekneeCOM->setPosition(kneeCOM);
//
//    auto debugSphere_urdf_abadCOM = server_->addVisualSphere("debug_sphere_urdf_abadCOM", 0.1);
//    debugSphere_urdf_abadCOM->setColor(0,1,0,1);
//    debugSphere_urdf_abadCOM->setPosition(urdf_abadCOM);
//
//    auto debugSphere_urdf_hipCOM = server_->addVisualSphere("debug_sphere_urdf_hipCOM", 0.1);
//    debugSphere_urdf_hipCOM->setColor(0,1,0,1);
//    debugSphere_urdf_hipCOM->setPosition(urdf_hipCOM);
//
//    auto debugSphere_urdf_kneeCOM = server_->addVisualSphere("debug_sphere_urdf_kneeCOM", 0.1);
//    debugSphere_urdf_kneeCOM->setColor(0,1,0,1);
//    debugSphere_urdf_kneeCOM->setPosition(urdf_kneeCOM);
  }

  void reset() {
    controller_.reset(world_.get());
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

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(i == delayDevidedBySimdt){
        controller_.advance(world_.get(), action);
      }
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();  // What does integration do? A. Simulate robot states and motions for the next simulation time.
      if (server_) server_->unlockVisualizationServerMutex();
      controller_.getReward(world_.get(), rewardCoeff_, simulation_dt_, curriculumFactor_);
      stepData_ += controller_.getStepData();
    }

    controller_.updateHistory();  /// update every control_dt
    controller_.updatePreviousActions();  /// update every control_dt

    stepData_ /= loopCount;

    double negativeRewardSum = stepData_.tail(2)(0);
    double positiveRewardSum = stepData_.tail(2)(1);
    double rewardSum = std::exp(0.2 * negativeRewardSum) * positiveRewardSum;
    return rewardSum;
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
  /////// optional methods ///////
  void curriculumUpdate() {curriculumFactor_ = pow(curriculumFactor_, curriculumRate_);};
  float getCurriculumFactor() {return float(curriculumFactor_);};
  void close() { if (server_) server_->killServer(); };
  void setSeed(int seed) { controller_.setSeed(seed); };
  ////////////////////////////////

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

  void printTest() { controller_.printTest();}

 private:
  std::map<MinicheetahController::RewardType, float> rewardCoeff_;
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;
  MinicheetahController controller_;
  std::unique_ptr<raisim::World> world_;
  double curriculumFactor_ = 0.3, curriculumRate_ = 0.998;
  double simulation_dt_ = 0.002;  // 0.002
  double control_dt_ = 0.016;  // 0.016
  int delayDevidedBySimdt;
  std::unique_ptr<raisim::RaisimServer> server_;
  Eigen::VectorXd stepData_;
  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}