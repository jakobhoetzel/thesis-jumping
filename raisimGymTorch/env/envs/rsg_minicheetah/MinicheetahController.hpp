// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include <cmath>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

class MinicheetahController {

 public:
  enum class RewardType : int {
    ANGULARVELOCIY1 = 1,
    VELOCITY1,
    AIRTIME,
    TORQUE,
    JOINTSPEED,
    FOOTSLIP,
    ORIENTATION,
    SMOOTHNESS1,
    SMOOTHNESS2,
    JOINTPOS,
    JOINTACC,
    BASEMOTION
  };

  void setSeed(int seed) { gen_.seed(seed); }

  bool create(raisim::World *world) { // called only once
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// get robot data
    gcDim_ = cheetah->getGeneralizedCoordinateDim();
    gvDim_ = cheetah->getDOF();
    nJoints_ = gvDim_ - 6;  // because each joint has 1 DOF and the main body has 6 DOF.

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_init_noise.setZero(gcDim_);  //gc_ and gv_ are expressed in the joint frame and with respect to the parent body
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_init_noise.setZero(gvDim_);
    gc_stationay_target.setZero(gcDim_);
    footPos_.resize(4); footVel_.resize(4);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of minicheetah
    gc_init_ << 0, 0, 0.25, //0.07,  // gc_init_.segment(0, 3): x, y, z position  //0.28
        1.0, 0.0, 0.0, 0.0,  // gc_init_.segment(3, 4): quaternion
//        0, -0.8, 1.8, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6;  // stand up
        0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326;  // stand up
//        -0.6, -1, 2.7, 0.6, -1, 2.7, -0.6, -1, 2.7, 0.6, -1, 2.7;
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
//        -0.62057, -1.039, 2.7, 0.6206, -1.039, 2.7, -0.6245, -1.034, 2.7, 0.62436, -1.034, 2.7;  // intended initial pose
//        0., 0., 0.5, 0., 0., 0.5, 0., 0., 0.5, 0., 0., 0.5;
//        -0.8659, -0.5307, 1.9672, 0.8661, -0.5302, 1.9666, -1.1019, -0.7014, 1.8063, 1.1017, -0.701, 1.8066;
//        -0.726685, -0.947298, 2.7, 0.726636, -0.947339, 2.7, -0.727, -0.94654, 2.65542, 0.727415, -0.946541, 2.65542;  // unintended initial pose
    gc_stationay_target << gc_init_.head(7),
        0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    cheetah->setPdGains(jointPgain, jointDgain);
    cheetah->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 130;  //34 //130 //106
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);  // action dimension is the same as the number of joints(by applying torque)
    obDouble_.setZero(obDim_);
    preJointTorque_.setZero(gv_.size()); preJointVel_.setZero(nJoints_);
    previousAction_.setZero(actionDim_); prepreviousAction_.setZero(actionDim_);
    airTime_.setZero(4);
    jointPosHist_.setZero(nJoints_ * historyLength_); jointVelHist_.setZero(nJoints_ * historyLength_);
    historyTempMem_.setZero(nJoints_ * historyLength_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// indices of links that are only possible to make contact with ground
    footIndices_.push_back(cheetah->getBodyIdx("shank_fr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_fl"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hl"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_fr_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_fl_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_hr_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_hl_joint"));

    stepDataTag_ = {"rewBodyAngularVel", "rewLinearVel", "rewAirTime", "rewTorque", "rewJointSpeed", "rewFootSlip",
                    "rewBodyOri", "rewSmoothness1", "rewSmoothness2", "rewJointPosition", "rewJointAcc", "rewBaseMotion",
                    "negativeRewardSum", "positiveRewardSum"};
    stepData_.resize(stepDataTag_.size());

    updateObservation(world);
    return true;
  }

  bool init(raisim::World *world) {
    return true;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {  // action is a position target. Eigen::Ref is for interchanging between c++ and python data types.
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    preJointTorque_ = cheetah->getGeneralizedForce();
    previousAction_ = pTarget12_;
    prepreviousAction_ = previousAction_;
    preJointVel_ = gv_.tail(nJoints_);

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

//    pTarget_ = gc_stationay_target;

    cheetah->setPdTarget(pTarget_, vTarget_);  // Set vTarget as 0 because we don't know vTarget. It works quite well.

    updateObservation(world);  // update obDouble, and so on.
    return true;
  }

  bool reset(raisim::World *world) {
    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));

    bool init_noise = true;

    if (init_noise) {
      /// Generalized Coordinates randomization.
      for (int i = 0; i < gcDim_; i++) {
        if(i<3) {
          continue;  /// XYZ position: no noise.
        } else if(i<7) {
          gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.1;  /// quaternion: +- 0.1
        } else {
          gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.2;  /// joint angles: +- 0.2rad
        }
      }
      double quat_sum = gc_init_noise.segment(3, 4).norm();
      gc_init_noise.segment(3, 4) /= quat_sum;

      /// Generalized Velocities randomization.
      for (int i = 0; i < gcDim_; i++) {
        if(i<3) {
          gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 0.5;  /// XYZ velocity: +- 0.5m/s
        } else if(i<6) {
          gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 0.5;  /// rpy: +- 0.5m/s
        } else {
          gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 0.9;  /// joint speed: +- 0.9rad/s
        }
      }

      /// Set the lowest foot on the ground.
      cheetah->setGeneralizedCoordinate(gc_init_noise);
      raisim::Vec<3> footPosition;
      double maxNecessaryShift = -1e20; // some arbitrary high negative value
      for(auto& foot: footFrameIndices_) {
        cheetah->getFramePosition(foot, footPosition);
//      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
        double terrainHeightMinusFootPosition = 0.0 - footPosition(2);
        maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
      }
      gc_init_noise(2) += maxNecessaryShift;
      cheetah->setState(gc_init_noise, gv_init_noise);
    } else {
      cheetah->setState(gc_init_, gv_init_);  // initialize the environment
    }


    preJointTorque_.setZero(); preJointVel_.setZero(); previousAction_ << actionMean_;
    jointVelHist_.setZero();
    jointPosHist_.setZero();
    historyTempMem_.setZero();
    jointVelHist_.segment(nJoints_ * (historyLength_-1), nJoints_) = gv_init_.tail(nJoints_);
    pTarget_ = gc_init_; vTarget_ = gv_init_; pTarget12_ = pTarget_.tail(nJoints_);

    cheetah->setPdTarget(pTarget_, vTarget_);
    world->integrate();

    updateObservation(world);  // initialize the robot

    return true;
  }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointPosHist_;
    jointPosHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointPosHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);
  }

  void getReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff, double simulation_dt, double curriculumFactor) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    Eigen::Vector3d linVelTarget(1, 0, 0);
    Eigen::Vector3d angVelTarget(0, 0, 0);
    Eigen::Vector3d oriTarget(0, 0, 1);

    /// A variable for foot slip reward
    double footTangentialForSlip = 0;
    for(int i = 0; i < 4; i++)
      if(footContactState_[i]) {
        footTangentialForSlip += footVel_[i].e().head(2).squaredNorm();
      }

    /// A variable for airtime reward calculation
    double airtimeTotal = 0;
    for(size_t i=0; i<4; i++) {
      if (footContactState_[i])
        airTime_[i] = std::min(0., airTime_[i]) - simulation_dt;
      else
        airTime_[i] = std::max(0., airTime_[i]) + simulation_dt;

      if (airTime_[i] < 0.5 && airTime_[i] > 0.)
        airtimeTotal += std::min(airTime_[i], 0.35);
      else if (airTime_[i] > -0.5 && airTime_[i] < 0.)
        airtimeTotal += std::min(-airTime_[i], 0.35);
    }

    /// Reward functions
    // no curriculum factor is applied at the moment
    double rewBodyAngularVel = std::exp(-1.5 * pow((angVelTarget(2) - bodyAngularVel_(2)), 2)) * rewardCoeff.at(RewardType::ANGULARVELOCIY1);
    double rewLinearVel = std::exp(-1.0 * (linVelTarget.head(2) - bodyLinearVel_.head(2)).squaredNorm()) * rewardCoeff.at(RewardType::VELOCITY1);
    double rewAirTime = airtimeTotal * rewardCoeff.at(RewardType::AIRTIME);
    double rewTorque = cheetah->getGeneralizedForce().squaredNorm() * rewardCoeff.at(RewardType::TORQUE);
    double rewJointSpeed = (gv_.tail(12)).squaredNorm() * rewardCoeff.at(RewardType::JOINTSPEED);
    double rewFootSlip = footTangentialForSlip * rewardCoeff.at(RewardType::FOOTSLIP);
    double rewBodyOri = std::acos(rot_(8)) * std::acos(rot_(8)) * rewardCoeff.at(RewardType::ORIENTATION);
    double rewSmoothness1 = rewardCoeff.at(RewardType::SMOOTHNESS1) * (pTarget12_ - previousAction_).squaredNorm();
    double rewSmoothness2 = rewardCoeff.at(RewardType::SMOOTHNESS2) * (pTarget12_ - 2 * previousAction_ + prepreviousAction_).squaredNorm();
    double rewJointPosition = (gc_.tail(nJoints_) - gc_init_.tail(nJoints_)).squaredNorm() * rewardCoeff.at(RewardType::JOINTPOS);
    double rewJointAcc = (gv_.tail(12) - preJointVel_.e()).squaredNorm() * rewardCoeff.at(RewardType::JOINTACC);
    double rewBaseMotion = (0.8 * bodyLinearVel_[2] * bodyLinearVel_[2] + 0.2 * fabs(bodyAngularVel_[0]) + 0.2 * fabs(bodyAngularVel_[1])) * rewardCoeff.at(RewardType::BASEMOTION);


    stepData_[0] = rewBodyAngularVel;  /// positive reward
    stepData_[1] = rewLinearVel;  /// positive reward
    stepData_[2] = rewAirTime;  /// positive reward
    stepData_[3] = rewTorque;
    stepData_[4] = rewJointSpeed;
    stepData_[5] = rewFootSlip;
    stepData_[6] = rewBodyOri;
    stepData_[7] = rewSmoothness1;
    stepData_[8] = rewSmoothness2;
    stepData_[9] = rewJointPosition;
    stepData_[10] = rewJointAcc;
    stepData_[11] = rewBaseMotion;

    double negativeRewardSum = stepData_.segment(3, 9).sum();
    double positiveRewardSum = stepData_.head(3).sum();

    stepData_[12] = negativeRewardSum;
    stepData_[13] = positiveRewardSum;
  }

  double rewKernel(double x) {
    return 1 / (exp(x) + 2 + exp(-x));  // which is opposite in sign to the const function version.
  }

  const std::vector<std::string>& getStepDataTag() {
    return stepDataTag_;
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }


  void updateObservation(raisim::World *world) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    cheetah->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);  // How to convert quaternion to orientation?
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    for(int i = 0; i < 4; i++) {
      cheetah->getFramePosition(footFrameIndices_[i], footPos_[i]);
      cheetah->getFrameVelocity(footFrameIndices_[i], footVel_[i]);
    }

    for(size_t i=0; i<4; i++)
      footContactState_[i] = false;

    for(auto& contact: cheetah->getContacts())
      for(size_t i=0; i<4; i++)
        if(contact.getlocalBodyIndex() == footIndices_[i])
          footContactState_[i] = true;

    obDouble_ << gc_[2], /// body height. 1
        rot_.e().row(2).transpose(), /// body orientation(z-axis in world frame expressed in body frame). 3
        gc_.tail(12), /// joint angles 12
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity. 3, 3
        gv_.tail(12), /// joint velocity 12
        previousAction_, /// previous action 12
        prepreviousAction_, /// preprevious action 12
        jointPosHist_.segment((historyLength_ - 9) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), /// joint History 24
        jointPosHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 4) * nJoints_, nJoints_), /// joint History 24
        jointPosHist_.segment((historyLength_ - 3) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 2) * nJoints_, nJoints_); /// joint History 24

    /// Observation noise
    bool addObsNoise = true;

    if(addObsNoise) {
      for(int i=0; i<obDim_; i++) {
        if(i<16) {
          obDouble_(i) = obDouble_(i) * (1 + normDist_(gen_) * 0.01) + normDist_(gen_) * 0.01;  /// height, orientation, joint angle
        } else if(i<34){
          obDouble_(i) = obDouble_(i) * (1 + normDist_(gen_) * 0.05) + normDist_(gen_) * 0.05;  /// velocities, joint velocity
        }
      }
    }

  }

  float generateRandomFloat(float min_f, float max_f) {
    return (max_f - min_f) * (uniDist_(gen_) + 1) / 2 + min_f;
  }

  const Eigen::VectorXd& getObservation() {
    return obDouble_;
  }

  /// If the contact body is not feet
  bool isTerminalState(raisim::World *world) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    for(auto& contact: cheetah->getContacts()) {  //getContacts() returns a vector of Contact instances
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }
    return false;
  }

  int getObDim() {
    return obDim_;
  }

  int getActionDim() {
    return actionDim_;
  }

  void printTest() {
    std::cout << "Test1: Observation before normalization." << std::endl;
    std::cout << obDouble_ << std::endl;
    std::cout << "joint target after scaling: " << std::endl;
    std::cout << pTarget12_ << std::endl;
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  raisim::Mat<3,3> rot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd gc_init_noise, gv_init_noise, gc_stationay_target;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, previousAction_, prepreviousAction_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::vector<size_t> footIndices_;
  std::array<bool, 4> footContactState_;
  std::vector<raisim::Vec<3>> footPos_, footVel_;
  std::vector<size_t> footFrameIndices_;
  int obDim_=0, actionDim_=0;
  int historyLength_ = 9;
  Eigen::VectorXd stepData_;
  Eigen::VectorXd airTime_;
  std::vector<std::string> stepDataTag_;
  Eigen::VectorXd jointPosHist_, jointVelHist_, historyTempMem_;
  VecDyn preJointTorque_, preJointVel_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937 raisim::MinicheetahController::gen_;
thread_local std::normal_distribution<double> raisim::MinicheetahController::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::MinicheetahController::uniDist_(-1., 1.);
}