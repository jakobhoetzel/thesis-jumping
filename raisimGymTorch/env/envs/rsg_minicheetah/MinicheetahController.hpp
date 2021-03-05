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
    ANGULARVELOCIY2,
    VELOCITY1,
    VELOCITY2,
    TORQUE,
    JOINTSPEED,
    FOOTCLEARANCE,
    FOOTSLIP,
    ORIENTATION,
    SMOOTHNESS,
    AIRTIME,
    HEIGHT
  };

  bool create(raisim::World *world) { // called only once
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
/*    std::vector<double> mass = cheetah->getMass();
    for (std::vector<double>::const_iterator i = mass.begin(); i != mass.end(); ++i)
      std::cout << *i << ' ';*/

    /// get robot data
    gcDim_ = cheetah->getGeneralizedCoordinateDim();
    gvDim_ = cheetah->getDOF();
    nJoints_ = gvDim_ - 6;  // because each joint has 1 DOF and the main body has 6 DOF.

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_init_noise.setZero(gcDim_);  //gc_ and gv_ are expressed in the joint frame and with respect to the parent body
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_init_noise.setZero(gvDim_);
    gc_stationay_target.setZero(gcDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_); // p and v mean position and velocity.
    footVelocityFR_.setZero(); footVelocityFL_.setZero(); footVelocityHR_.setZero(); footVelocityHL_.setZero();
//    desiredFootPositionFR_.setZero(), desiredFootPositionFL_.setZero(), desiredFootPositionHR_.setZero(), desiredFootPositionHL_.setZero();
    desiredFootZPosition_ = 0.04;
    currentFootPositionFR_.setZero(), currentFootPositionFL_.setZero(), currentFootPositionHR_.setZero(), currentFootPositionHL_.setZero();
    footPositionInBodyFrame_ = {0.0, 0.0, -0.18};
    // pTarget12_, which is for last 12 values of pTarget_, would be incorporated into pTarget later.

    /// this is nominal configuration of minicheetah
    gc_init_ << 0, 0, 0.28, //0.07,  // gc_init_.segment(0, 3): x, y, z position  //0.28
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

//    gv_init_ << 0.016, 0, 0.09, 0.028, -0.016, 0,
//        0.36, 0.068, -0.394, -0.847, 0.03, -1.08, 0.325, 0.0556, -0.364, -0.324, 0.073, -0.406;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    cheetah->setPdGains(jointPgain, jointDgain);
    cheetah->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);  // action dimension is the same as the number of joints(by applying torque)
    obDouble_.setZero(obDim_);
    preJointTorque_.setZero(gv_.size());
    airTime_.setZero(4);
    jointVelHist_.setZero(nJoints_ * historyLength_);
    jointErrorHist_.setZero(nJoints_ * historyLength_);
    historyTempMem_.setZero(nJoints_ * historyLength_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// indices of links that are only possible to make contact with ground
    footIndices_.push_back(cheetah->getBodyIdx("shank_fr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_fl"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hl"));

    stepDataTag_ = {"rewBodyAngularVel", "rewLinearVel", "rewTorque", "rewJointSpeed", "rewFootClearance",
                    "rewFootSlip", "rewBodyOri", "rewSmoothness", "rewAirTime", "rewHeight"};
    stepData_.resize(stepDataTag_.size());  // reward & gv_

    srand(static_cast <unsigned> (time(0)));  // for observation noise

    updateObservation(world);
    return true;
  }

  bool init(raisim::World *world) {
    return true;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {  // action is a position target. Eigen::Ref is for interchanging between c++ and python data types.
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    preJointTorque_ = cheetah->getGeneralizedForce();

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
      for (int i = 0; i < gcDim_; i++) {
        if(i==0) {
          gc_init_noise(i) = gc_init_(i) + generateRandomFloat(-0.1, 0.1);
        } else if(i<4) {
          gc_init_noise(i) = gc_init_(i) + generateRandomFloat(-0.01, 0.01);
        } else {
          gc_init_noise(i) = gc_init_(i) + generateRandomFloat(-0.4, 0.4);
        }
      }
      double quat_sum = gc_init_noise.segment(1, 3).norm();
      gc_init_noise.segment(1, 3) /= quat_sum;
      for (int i = 0; i < gcDim_; i++) {
        if(i<3) {
          gv_init_noise(i) = gv_init_(i) + generateRandomFloat(-0.5, 0.5);
        } else if(i<6) {
          gv_init_noise(i) = gv_init_(i) + generateRandomFloat(-0.5, 0.5);
        } else {
          gv_init_noise(i) = gv_init_(i) + generateRandomFloat(-0.9, 0.9);
        }
      }
      cheetah->setState(gc_init_noise, gv_init_noise);
    } else {
      cheetah->setState(gc_init_, gv_init_);  // initialize the environment
    }


    preJointTorque_.setZero();
    jointVelHist_.setZero();
    jointErrorHist_.setZero();
    historyTempMem_.setZero();
    pTarget_ = gc_init_; vTarget_ = gv_init_; pTarget12_ = pTarget_.tail(nJoints_);
    footVelocityFR_.setZero(); footVelocityFL_.setZero(); footVelocityHR_.setZero(); footVelocityHL_.setZero();

    cheetah->setPdTarget(pTarget_, vTarget_);
    world->integrate();

    updateObservation(world);  // initialize the robot

    return true;
  }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointErrorHist_;
    jointErrorHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointErrorHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);
  }

  double getReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff, double simulation_dt, double curriculumFactor) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d linVelTarget(1, 0, 0);
    Eigen::Vector3d angVelTarget(0, 0, 0);
    Eigen::Vector3d oriTarget(0, 0, 1);

    VecDyn torqueDifference = cheetah->getGeneralizedForce();
    torqueDifference -= preJointTorque_;

    for(size_t i=0; i<4; i++)
      footContactState_[i] = false;

    for(auto& contact: cheetah->getContacts())
      for(size_t i=0; i<4; i++)
        if(contact.getlocalBodyIndex() == footIndices_[i])
          footContactState_[i] = true;

    cheetah->getPosition(cheetah->getBodyIdx("shank_fr"), footPositionInBodyFrame_, currentFootPositionFR_);
    cheetah->getPosition(cheetah->getBodyIdx("shank_fl"), footPositionInBodyFrame_, currentFootPositionFL_);
    cheetah->getPosition(cheetah->getBodyIdx("shank_hr"), footPositionInBodyFrame_, currentFootPositionHR_);
    cheetah->getPosition(cheetah->getBodyIdx("shank_hl"), footPositionInBodyFrame_, currentFootPositionHL_);

    cheetah->getVelocity(cheetah->getBodyIdx("shank_fr"), footPositionInBodyFrame_, footVelocityFR_);
    cheetah->getVelocity(cheetah->getBodyIdx("shank_fl"), footPositionInBodyFrame_, footVelocityFL_);
    cheetah->getVelocity(cheetah->getBodyIdx("shank_hr"), footPositionInBodyFrame_, footVelocityHR_);
    cheetah->getVelocity(cheetah->getBodyIdx("shank_hl"), footPositionInBodyFrame_, footVelocityHL_);

    double footTangentialFR = sqrt(pow(footVelocityFR_[0], 2) + pow(footVelocityFR_[1], 2));
    double footTangentialFL = sqrt(pow(footVelocityFL_[0], 2) + pow(footVelocityFL_[1], 2));
    double footTangentialHR = sqrt(pow(footVelocityHR_[0], 2) + pow(footVelocityHR_[1], 2));
    double footTangentialHL = sqrt(pow(footVelocityHL_[0], 2) + pow(footVelocityHL_[1], 2));

    double footTangentialForSlip = footContactState_[0] * footTangentialFR + footContactState_[1] * footTangentialFL +
        footContactState_[2] * footTangentialHR + footContactState_[3] * footTangentialHL;

    double airtimeRew = 0;

    for(size_t i=0; i<4; i++) {
      if (footContactState_[i])
        airTime_[i] = std::min(0., airTime_[i]) - simulation_dt;
      else
        airTime_[i] = std::max(0., airTime_[i]) + simulation_dt;

      if (airTime_[i] < 0.4 && airTime_[i] > 0.)
        airtimeRew += std::min(airTime_[i], 0.2);
      else if (airTime_[i] > -0.4 && airTime_[i] < 0.)
        airtimeRew += std::min(-airTime_[i], 0.2);
    }

    double footClearanceFR = pow((desiredFootZPosition_ - currentFootPositionFR_[2]), 2) * footTangentialFR * (1 - footContactState_[0]);
    double footClearanceFL = pow((desiredFootZPosition_ - currentFootPositionFL_[2]), 2) * footTangentialFL * (1 - footContactState_[1]);
    double footClearanceHR = pow((desiredFootZPosition_ - currentFootPositionHR_[2]), 2) * footTangentialHR * (1 - footContactState_[2]);
    double footClearanceHL = pow((desiredFootZPosition_ - currentFootPositionHL_[2]), 2) * footTangentialHL * (1 - footContactState_[3]);

    //double rewVel = std::min(2.0, bodyLinearVel_[0])* rewardCoeff.at(RewardType::VELOCITY1);
    double rewBodyAngularVel = rewKernel((angVelTarget - bodyAngularVel_).squaredNorm() * rewardCoeff.at(RewardType::ANGULARVELOCIY2)) * rewardCoeff.at(RewardType::ANGULARVELOCIY1);
    double rewLinearVel = rewKernel((linVelTarget - bodyLinearVel_).norm() * rewardCoeff.at(RewardType::VELOCITY2)) * rewardCoeff.at(RewardType::VELOCITY1);
    double rewTorque = curriculumFactor * cheetah->getGeneralizedForce().squaredNorm() * rewardCoeff.at(RewardType::TORQUE);
    double rewFootClearance = curriculumFactor * (footClearanceFR + footClearanceFL + footClearanceHR + footClearanceHL) * rewardCoeff.at(RewardType::FOOTCLEARANCE);
    double rewJointSpeed = curriculumFactor * (gv_.tail(12)).squaredNorm() * rewardCoeff.at(RewardType::JOINTSPEED);
    double rewFootSlip = curriculumFactor * footTangentialForSlip * rewardCoeff.at(RewardType::FOOTSLIP);
    double rewBodyOri = curriculumFactor * (oriTarget - rot.e().row(2).transpose()).norm() * rewardCoeff.at(RewardType::ORIENTATION);
    double rewSmoothness = curriculumFactor * (torqueDifference).squaredNorm() * rewardCoeff.at(RewardType::SMOOTHNESS);
    double rewAirTime = curriculumFactor * airtimeRew * rewardCoeff.at(RewardType::AIRTIME);
    double rewHeight = curriculumFactor * std::abs(gc_[0]-0.25) * rewardCoeff.at(RewardType::HEIGHT);
//     generalized force is joint torques, and doesn't include contact forces.
//     positive rot.e().row(2)[0]: pitch up
//     positive rot.e().row(2)[1]: roll to the right
//     rot.e().row(2): projection of body frame's unit vectors onto the gravity vector.

//    std::cout << (desiredFootZPosition_ - currentFootPositionFR_[2]) << std::endl;

    stepData_[0] = rewBodyAngularVel;
    stepData_[1] = rewLinearVel;
    stepData_[2] = rewTorque;
    stepData_[3] = rewJointSpeed;
    stepData_[4] = 0;//rewFootClearance;
    stepData_[5] = 0;//rewFootSlip;
    stepData_[6] = rewBodyOri;
    stepData_[7] = rewSmoothness;
    stepData_[8] = rewAirTime;
    stepData_[9] = rewHeight;

    return stepData_.sum();
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


  void updateObservation(raisim::World *world) {  // Update gc_, gv_, bodyLinearVel_, bodyAngularVel_ => obDouble_
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    cheetah->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);  // How to convert quaternion to orientation?
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height. 1
        rot.e().row(2).transpose(), /// body orientation(z-axis in world frame expressed in body frame). 3
        gc_.tail(12), /// joint angles 12
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity. 3, 3
        gv_.tail(12); /// joint velocity 12
//        jointVelHist_, /// history 48
//        jointErrorHist_; /// pos error 48

    bool addObsNoise = true;

    if(addObsNoise) {
      for(int i=0; i<obDim_; i++) {
        if(i<16) {
          obDouble_(i) = obDouble_(i) * (1 + generateRandomFloat(-0.03, 0.03)) + generateRandomFloat(-0.03, 0.03);
        } else {
          obDouble_(i) = obDouble_(i) * (1 + generateRandomFloat(-0.1, 0.1)) + generateRandomFloat(-0.1, 0.1);
        }
      }
    }

  }

  float generateRandomFloat(float min_f, float max_f) {
    return (max_f - min_f) * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min_f;
  }

  const Eigen::VectorXd& getObservation() {
    return obDouble_;
  }

  /// If the contact body is not feet
  bool isTerminalState(raisim::World *world) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    for(auto& contact: cheetah->getContacts()){  //getContacts() returns a vector of Contact instances
      if(std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
//      std::cout<<"is ObjectA: "<<contact.isObjectA()<<std::endl;
//      std::cout<<"Contact position in the world frame: "<<contact.getPosition().e().transpose()<<std::endl;
//      std::cout<<"It collides with: "<<int(contact.getPairObjectBodyType())<<std::endl;
    }
//    std::cout<<"One step is finished" << std::endl;
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
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd gc_init_noise, gv_init_noise, gc_stationay_target;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::vector<size_t> footIndices_;
  std::array<bool, 4> footContactState_;
//  raisim::Vec<3> desiredFootPositionFR_, desiredFootPositionFL_, desiredFootPositionHR_, desiredFootPositionHL_;
  double desiredFootZPosition_;
  raisim::Vec<3> currentFootPositionFR_, currentFootPositionFL_, currentFootPositionHR_, currentFootPositionHL_;
  raisim::Vec<3> footVelocityFR_, footVelocityFL_, footVelocityHR_, footVelocityHL_;
  raisim::Vec<3> footPositionInBodyFrame_;
  int obDim_=0, actionDim_=0;
  int historyLength_ = 4;
  Eigen::VectorXd stepData_;
  Eigen::VectorXd airTime_;
  std::vector<std::string> stepDataTag_;
  Eigen::VectorXd jointVelHist_, jointErrorHist_, historyTempMem_;
  VecDyn preJointTorque_;
};

}