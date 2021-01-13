// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

class HubodogController {

 public:
  bool create(raisim::World *world) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// get robot data
    gcDim_ = anymal->getGeneralizedCoordinateDim();
    gvDim_ = anymal->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal->setPdGains(jointPgain, jointDgain);
    anymal->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal->getBodyIdx("RH_SHANK"));

    updateObservation(world);
    return true;
  }

  bool init(raisim::World *world) {
    return true;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal->setPdTarget(pTarget_, vTarget_);
    updateObservation(world);
    return true;
  }

  bool reset(raisim::World *world) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    anymal->setState(gc_init_, gv_init_);
    updateObservation(world);
    return true;
  }

  double getReward(raisim::World *world, double forwardVelCoeff, double torqueCoeff) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    return std::min(4.0, bodyLinearVel_[0]) * forwardVelCoeff + anymal->getGeneralizedForce().squaredNorm() * torqueCoeff;
  }

  void updateObservation(raisim::World *world) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    anymal->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12); /// joint velocity
  }

  const Eigen::VectorXd& getObservation() {
    return obDouble_;
  }

  bool isTerminalState(raisim::World *world) {
    auto* anymal = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    for(auto& contact: anymal->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  int getObDim() {
    return obDim_;
  }

  int getActionDim() {
    return actionDim_;
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_=0, actionDim_=0;
};

}