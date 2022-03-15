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

Eigen::IOFormat CleanFmt(4, 0, "\t", "\n", "[", "]");

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
    BASEMOTION,
    FOOTCLEARANCE,
    HURDLES,
    SYMMETRY,
    FOOTCONTACT,
  };

  void setSeed(int seed) { gen_.seed(seed); }

  bool create(raisim::World *world) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// get robot data
    gcDim_ = cheetah->getGeneralizedCoordinateDim();
    gvDim_ = cheetah->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_init_noise.setZero(gcDim_);  //gc_ and gv_ are expressed in the joint frame and with respect to the parent body
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_init_noise.setZero(gvDim_);
    gc_stationay_target.setZero(gcDim_);
    footPos_.resize(4); footVel_.resize(4);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of minicheetah
    gc_init_ << 0, 0, 0.25,  // x, y, z position
        1.0, 0.0, 0.0, 0.0,  // quaternion
        0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8;  // joint

    gc_stationay_target << gc_init_.head(7),
        0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8;

    command_.setZero();

    /// set pd gains
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(17.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(0.4);
    cheetah->setPdGains(jointPgain_, jointDgain_);
    cheetah->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 141 + 2;  //34 //106 //130 //133 //198 //2 for sensor
    robotStateDim_ = 11;  //4
    historyLength_ = 6;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);  // action dimension is the same as the number of joints(by applying torque)
    obDouble_.setZero(obDim_); robotState_.setZero(robotStateDim_);
    preJointVel_.setZero(nJoints_);
    previousAction_.setZero(actionDim_); prepreviousAction_.setZero(actionDim_);
    airTime_.setZero(4); stanceTime_.setZero(4);
    jointPosErrorHist_.setZero(nJoints_ * historyLength_); jointVelHist_.setZero(nJoints_ * historyLength_);
    historyTempMem_.setZero(nJoints_ * historyLength_);
    jointFrictions_.setZero(nJoints_);
    maxBodyHeight_ = 0.0;
    maxXPos_ = 0.0;
    step=0;

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);  // 0.3

    /// indices of links that are only possible to make contact with ground
    footIndices_.push_back(cheetah->getBodyIdx("shank_fr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_fl"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hr"));
    footIndices_.push_back(cheetah->getBodyIdx("shank_hl"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_fr_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_fl_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_hr_joint"));
    footFrameIndices_.push_back(cheetah->getFrameIdxByName("toe_hl_joint"));

    stepDataTag_ = {"rewBodyAngularVel", "rewLinearVel", "rewAirTime", "rewHurdles", "rewTorque", "rewJointSpeed",
                    "rewFootSlip", "rewBodyOri", "rewSmoothness1", "rewSmoothness2", "rewJointPosition", "rewJointAcc",
                    "rewBaseMotion", "rewFootClearance", "rewSymmetry", "rewFootContact",
                    "negativeRewardSum", "positiveRewardSum", "totalRewardSum"};

    stepData_.resize(stepDataTag_.size());

    updateObservation(world);
    return true;
  }

  bool init(raisim::World *world) {
    return true;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {  // action is a position target. Eigen::Ref is for interchanging between c++ and python data types.
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

//    pTarget_ = gc_stationay_target;

    cheetah->setPdTarget(pTarget_, vTarget_);

    // joint friction
    Eigen::VectorXd tau; tau.setZero(gvDim_);
    for (int i = 0; i < nJoints_; i++){
      double jTorque = jointPgain_.tail(nJoints_)(i) * (pTarget12_(i) - gc_.tail(nJoints_)(i));
      if (jTorque > 0) tau.tail(nJoints_)(i) = std::min(jointFrictions_(i), jTorque);
      else tau.tail(nJoints_)(i) = std::max(-jointFrictions_(i), jTorque);
    }
    cheetah->setGeneralizedForce(-tau);

    if(false) { //output if max joint speed or torque is exceeded
//      std::cout << gv_.tail(nJoints_)  << std::endl;
//      std::cout << cheetah->getGeneralizedForce() << std::endl;
      if (std::abs(gv_.tail(nJoints_)(0)) > 40 or std::abs(gv_.tail(nJoints_)(3)) > 40
          or std::abs(gv_.tail(nJoints_)(6)) > 40 or std::abs(gv_.tail(nJoints_)(9)) > 40) {
        std::cout << "Exceeds maximum joint speed at hip actuator" << std::endl;
      }
      if (std::abs(gv_.tail(nJoints_)(1)) > 40 or std::abs(gv_.tail(nJoints_)(4)) > 40
          or std::abs(gv_.tail(nJoints_)(7)) > 40 or std::abs(gv_.tail(nJoints_)(10)) > 40) {
        std::cout << "Exceeds maximum joint speed at ab/ad actuator" << std::endl;
      }
      if (std::abs(gv_.tail(nJoints_)(2)) > 25.8 or std::abs(gv_.tail(nJoints_)(5)) > 25.8
          or std::abs(gv_.tail(nJoints_)(8)) > 25.8 or std::abs(gv_.tail(nJoints_)(11)) > 25.8) {
        std::cout << "Exceeds maximum joint speed at knee actuator" << std::endl;
      }
      if (std::abs(cheetah->getGeneralizedForce()[6]) > 17 or std::abs(cheetah->getGeneralizedForce()[9]) > 17
          or std::abs(cheetah->getGeneralizedForce()[12]) > 17 or std::abs(cheetah->getGeneralizedForce()[15]) > 17) {
        std::cout << "Exceeds maximum joint torque at hip actuator" << std::endl;
      }
      if (std::abs(cheetah->getGeneralizedForce()[7]) > 17 or std::abs(cheetah->getGeneralizedForce()[10]) > 17
          or std::abs(cheetah->getGeneralizedForce()[13]) > 17 or std::abs(cheetah->getGeneralizedForce()[16]) > 17) {
        std::cout << "Exceeds maximum joint torque at ab/ad actuator" << std::endl;
      }
      if (std::abs(cheetah->getGeneralizedForce()[8]) > 26.3 or std::abs(cheetah->getGeneralizedForce()[11]) > 26.3
          or std::abs(cheetah->getGeneralizedForce()[14]) > 26.3 or std::abs(cheetah->getGeneralizedForce()[17]) > 26.3) {
        std::cout << "Exceeds maximum joint torque at knee actuator" << std::endl;
      }
    }

    return true;
  }

  void go_straight_controller(){
//    command_(1) = -0.2 * gc_(1);
    command_(2) = -2 * gv_(1);
  }

  bool reset(raisim::World *world, double comCurriculumFactor, raisim::HeightMap* heightMap_, bool hurdleTraining) {
    hurdleTraining_ = hurdleTraining;
    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));

    /// pd gain randomization
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(17.0 + 2 * uniDist_(gen_));
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(0.4 + 0.1 * uniDist_(gen_));
    cheetah->setPdGains(jointPgain_, jointDgain_);

    /// command generation
    double p = uniDist_(gen_);
    if(hurdleTraining_){
      command_ << 3.5, 0.0, 0.0; // 4.0, 0, 0
    }
    else{
      if(fabs(p) < 0.2) {  // 10%
        command_.setZero();
        standingMode_ = true;
      }
      else {
        do {
          command_ << comCurriculumFactor * uniDist_(gen_), 0.5 * uniDist_(gen_), 0.5 * uniDist_(gen_); // comCurriculumFactor, 1.0, 2.0
          if (command_(0) < 0) {
            command_(0) *= 0.5;
          }
        } while (command_.norm() < 0.3);
        standingMode_ = false;
      }
    }


    bool keep_state = fabs(uniDist_(gen_)) < 0.25; /// keep state and only change command. 25%
    if (keep_state){
      gc_init_noise.tail(17) = gc_.tail(17);  // body z, quaterniton, joint position. X and Y positions are set to zero.
      gv_init_noise = gv_;
    }
    else {
      bool init_noise = true;
      if (init_noise) { //TODO: set noise
        /// Generalized Coordinates randomization.
        for (int i = 0; i < gcDim_; i++) {
          if (i < 3) {
            continue;  /// XYZ position: no noise.
          } else if (i < 7) {
            gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.2;  /// quaternion: +- 0.2
          } else {
            if (i % 3 == 1)
              gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.2;  /// HAA joint angles: +- 0.2rad (hip abduction/adduction)
            if (i % 3 == 2)
              gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.2;  /// HFE joint angles: +- 0.2rad (hip flexion/extension)
            else
              gc_init_noise(i) = gc_init_(i) + uniDist_(gen_) * 0.2;  /// knee joint angles: +- 0.2rad
          }
        }
        double quat_sum = gc_init_noise.segment(3, 4).norm();
        gc_init_noise.segment(3, 4) /= quat_sum;

        /// Generalized Velocities randomization.
        for (int i = 0; i < gvDim_; i++) {
          if (i < 3) {
            if (i == 0) {
              gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 1.0;  /// X velocity: +- 1.0m/s
            }
            else {
              gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 0.5;  /// YZ velocity: +- 0.5m/s
            }
          } else if (i < 6) {
            gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 0.7;  /// rpy: +- 0.7rad/s
          } else {
            gv_init_noise(i) = gv_init_(i) + uniDist_(gen_) * 2.5;  /// joint speed: +- 2.5rad/s
          }

          if (standingMode_) gv_init_noise(i) *= 2;
        }
      } else {
        gc_init_noise = gc_init_;
        gv_init_noise = gv_init_;
      }
    }

    double jFrictionHAAHFE = 0.15 * (uniDist_(gen_) + 1);  // [0, 0.3]
    double jFrictionKFE = 0.3 * (uniDist_(gen_) + 1) + 0.1;  // [0.1, 0.7]
    jointFrictions_ << jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE,
        jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE;


    /// Set the lowest foot on the ground.
    cheetah->setGeneralizedCoordinate(gc_init_noise);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; // some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      cheetah->getFramePosition(foot, footPosition);

      double terrainHeightMinusFootPosition = 0.;
      if(isHeightMap_){
        terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      }
      else {
        terrainHeightMinusFootPosition = 0.0 - footPosition(2);
      }

      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_noise(2) += maxNecessaryShift;
    cheetah->setState(gc_init_noise, gv_init_noise);  // initialize the environment
    updateObservation(world);  // initialize the robot

    if(keep_state) return true;  // if keep_state, keep history remained.

    preJointVel_.setZero();
    historyTempMem_.setZero();
    for(int i = 0; i < historyLength_; i++) {
      jointPosErrorHist_.segment(nJoints_ * i, nJoints_).setZero();
      jointVelHist_.segment(nJoints_ * i, nJoints_) = gv_init_noise.tail(nJoints_);
    }
    pTarget_.tail(nJoints_) = gc_init_noise.tail(nJoints_); pTarget12_ = pTarget_.tail(nJoints_);
    previousAction_ << pTarget12_; prepreviousAction_ << pTarget12_;

    for(int i=0; i<4; i++) airTime_[i] = 0;
    for(int i=0; i<4; i++) stanceTime_[i] = 0;
    maxBodyHeight_ = 0.0;
    maxXPos_ = 0.0;
    step=0;

    return true;
  }

  void collisionRandomization(raisim::World *world) {  // randomizing foot size and positions
    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));

    size_t foot_fr_idx, foot_fl_idx, foot_hr_idx, foot_hl_idx;
    foot_fr_idx = 4; foot_fl_idx = 8; foot_hr_idx = 12; foot_hl_idx = 16;

    Vec<3> pos_offset;
    pos_offset = {uniDist_(gen_)*0.01, uniDist_(gen_)*0.005, uniDist_(gen_)*0.02 - 0.195};
    cheetah->setCollisionObjectPositionOffset(foot_fr_idx, pos_offset);
    pos_offset = {uniDist_(gen_)*0.01, uniDist_(gen_)*0.005, uniDist_(gen_)*0.02 - 0.195};
    cheetah->setCollisionObjectPositionOffset(foot_fl_idx, pos_offset);
    pos_offset = {uniDist_(gen_)*0.01, uniDist_(gen_)*0.005, uniDist_(gen_)*0.02 - 0.195};
    cheetah->setCollisionObjectPositionOffset(foot_hr_idx, pos_offset);
    pos_offset = {uniDist_(gen_)*0.01, uniDist_(gen_)*0.005, uniDist_(gen_)*0.02 - 0.195};
    cheetah->setCollisionObjectPositionOffset(foot_hl_idx, pos_offset);

    std::vector<double> rand_radius;
    rand_radius.push_back((uniDist_(gen_)+1) * 0.002 + 0.006);  // [6, 10] mm
    cheetah->setCollisionObjectShapeParameters(foot_fr_idx, rand_radius);
    cheetah->setCollisionObjectShapeParameters(foot_fl_idx, rand_radius);
    cheetah->setCollisionObjectShapeParameters(foot_hr_idx, rand_radius);
    cheetah->setCollisionObjectShapeParameters(foot_hl_idx, rand_radius);
  }

  void getReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff, double simulation_dt, double rewCurriculumFactor, raisim::HeightMap* heightMap_, double xPosHurdles, int iteration) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    double desiredFootZPosition = 0.09;
    preJointVel_ = gv_.tail(nJoints_);
    updateObservation(world);

    /// A variable for foot slip reward and foot clearance reward
    double footTangentialForSlip = 0, footClearanceTangential = 0;
    for(int i = 0; i < 4; i++) {
      if (footContactState_[i]) {
        footTangentialForSlip += footVel_[i].e().head(2).squaredNorm();
      } else {
        if (!standingMode_) {
          if(isHeightMap_) {
            footClearanceTangential +=
                std::pow((footPos_[i].e()(2) - heightMap_->getHeight(footPos_[i].e()(0), footPos_[i].e()(1)) - desiredFootZPosition), 2) * sqrt(footVel_[i].e().head(2).norm());
          }
          else {
            footClearanceTangential +=
                std::pow((footPos_[i].e()(2) - desiredFootZPosition), 2) * sqrt(footVel_[i].e().head(2).norm());
          }
        }
      }
    }

    /// A variable for airtime reward calculation
    double airtimeTotal = 0;
    for(size_t i=0; i<4; i++) {
      if (footContactState_[i]) {
        stanceTime_[i] += simulation_dt;
        airTime_[i] = 0;
      }
      else {
        airTime_[i] += simulation_dt;
        stanceTime_[i] = 0;
      }

      if (standingMode_) {
        airtimeTotal += std::min(std::max(stanceTime_[i] - airTime_[i], -0.3), 0.3);
      }
      else {
        if (airTime_[i] < 0.25)
          airtimeTotal += std::min(airTime_[i], 0.2);
        if (stanceTime_[i] < 0.25)
          airtimeTotal += std::min(stanceTime_[i], 0.2);
      }
    }

    /// A variable for hurdles reward calculation
    double hurdlesVar = 0;
    double footContactNumber = std::accumulate(footContactState_.begin(), footContactState_.end(),0); //number of feet touching the ground
    if (gc_[0] >= xPosHurdles && maxXPos_ < xPosHurdles && footContactNumber == 0){ // get reward once when above hurdle
      hurdlesVar = 1.0 + gc_[2]/2;
    } else if ( gc_[0] >= xPosHurdles + 0.14 && maxXPos_ < xPosHurdles + 0.14 ){  // get reward once when behind
      hurdlesVar = 1.0;
    }
    maxXPos_ = std::max(maxXPos_, gc_[0]);

    /// A variable for hurdles reward calculation
    maxBodyHeight_ = std::max(maxBodyHeight_, gc_[2]);
    maxBodyHeight_ = std::max(maxBodyHeight_, 0.2);

    /// A variable to calculate the symmetry of the motion ("cheetah instead of horse")
    Eigen::Vector3d anglesRF = gc_.segment(7, 3); /// order: RF - LF - RH - LH
    Eigen::Vector3d anglesLF = gc_.segment(10, 3);
    anglesLF[0] = -anglesLF[0]; // HAA joint angles must have different sign to be symmetric
    Eigen::Vector3d anglesRH = gc_.segment(13, 3);
    Eigen::Vector3d anglesLH = gc_.segment(16, 3);
    anglesLH[0] = -anglesLH[0];
    double symmetryCoeff = (anglesRF - anglesLF).norm() + (anglesRH - anglesLH).norm();

    /// A variable to reward the correct touching of the ground
    double footContactVar;
    if (footContactNumber == 0 || footContactNumber == 1){ //zero or one foot on the ground
      footContactVar = -1; // good due to multiplication with negative coefficient
    }
    else if (footContactNumber == 3 || footContactNumber == 4){ //three or four feet on the ground
      footContactVar = 1; // bad
    }
    else if (footContactState_[0] == footContactState_[1]){ //only foreleg or hind leg on the ground
      footContactVar = -1;
    }
    else{
      footContactVar = 1;
    }

    double exceedFactor = 1;
    if(iteration>5000){
      exceedFactor = 10 * (iteration-5000)/2500;
    }


    /// Reward functions
    // curriculum factor in negative reward
    double rewBodyAngularVel = std::exp(-1.5 * pow((command_(2) - bodyAngularVel_(2)), 2)) * rewardCoeff.at(RewardType::ANGULARVELOCIY1);
    double rewLinearVel = std::exp(-1.0 * (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm()) * rewardCoeff.at(RewardType::VELOCITY1);
//    double rewLinearVel = std::exp(0.4 * std::min(bodyLinearVel_[0],3.5) - 0.4*std::abs(bodyLinearVel_[1])) * rewardCoeff.at(RewardType::VELOCITY1); //max reward limited
    double rewAirTime = airtimeTotal * rewardCoeff.at(RewardType::AIRTIME);
    double rewHurdles = hurdlesVar * rewardCoeff.at(RewardType::HURDLES);
    double rewTorque = rewardCoeff.at(RewardType::TORQUE) * cheetah->getGeneralizedForce().squaredNorm() * exceedFactor; //max torque: 17, 17, 26.3(?) Nm (last is knee)
    double rewJointSpeed = (gv_.tail(12)).squaredNorm() * rewardCoeff.at(RewardType::JOINTSPEED) * exceedFactor; // max joint speed: 40, 40, 25.8(?) rad/s
    double rewFootSlip = footTangentialForSlip * rewardCoeff.at(RewardType::FOOTSLIP);
    double rewBodyOri = std::acos(rot_(8)) * std::acos(rot_(8)) * rewardCoeff.at(RewardType::ORIENTATION);
    double rewSmoothness1 = rewardCoeff.at(RewardType::SMOOTHNESS1) * (pTarget12_ - previousAction_).squaredNorm();
    double rewSmoothness2 = rewardCoeff.at(RewardType::SMOOTHNESS2) * (pTarget12_ - 2 * previousAction_ + prepreviousAction_).squaredNorm();
    double rewJointPosition = (gc_.tail(nJoints_) - gc_init_.tail(nJoints_)).squaredNorm() * rewardCoeff.at(RewardType::JOINTPOS);
    double rewJointAcc = (gv_.tail(12) - preJointVel_).squaredNorm() * rewardCoeff.at(RewardType::JOINTACC);
    double rewBaseMotion = (0.3 * bodyLinearVel_[2] * bodyLinearVel_[2] + 0.2 * fabs(bodyAngularVel_[0]) + 0.2 * fabs(bodyAngularVel_[1])) * rewardCoeff.at(RewardType::BASEMOTION);
    double rewFootClearance = footClearanceTangential * rewardCoeff.at(RewardType::FOOTCLEARANCE);
    double rewSymmetry = (1 - rewCurriculumFactor) * symmetryCoeff * rewardCoeff.at(RewardType::SYMMETRY); /// curriculum 1->0
    double rewFootContact = footContactVar * rewardCoeff.at(RewardType::FOOTCONTACT);

    stepData_[0] = rewBodyAngularVel;  /// positive reward; maximization
    stepData_[1] = rewLinearVel;  /// positive reward
    stepData_[2] = rewAirTime;  /// positive reward
    stepData_[3] = rewHurdles;  /// positive reward
    stepData_[4] = rewTorque;
    stepData_[5] = rewJointSpeed;
    stepData_[6] = rewFootSlip;
    stepData_[7] = rewBodyOri;
    stepData_[8] = rewSmoothness1;
    stepData_[9] = rewSmoothness2;
    stepData_[10] = rewJointPosition;
    stepData_[11] = rewJointAcc;
    stepData_[12] = rewBaseMotion;
    stepData_[13] = rewFootClearance;
    stepData_[14] = rewSymmetry / (rewCurriculumFactor + 1e-3); /// not affected of curriculum
    stepData_[15] = rewFootContact / (rewCurriculumFactor + 1e-3);

    double negativeRewardSum = stepData_.segment(4, stepDataTag_.size()-7).sum()* rewCurriculumFactor; /// curriculum 0->1
    double positiveRewardSum = stepData_.head(4).sum();

    stepData_[16] = negativeRewardSum;
    stepData_[17] = positiveRewardSum;
    stepData_[18] = std::exp(0.2 * negativeRewardSum) * positiveRewardSum;  // totalReward
  }

  const std::vector<std::string>& getStepDataTag() {
    return stepDataTag_;
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
    std::cout << "command: " << command_ << std::endl; // print command
  }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointPosErrorHist_;
    jointPosErrorHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointPosErrorHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);
  }

  void updatePreviousActions() {
    prepreviousAction_ = previousAction_;
    previousAction_ = pTarget12_;
  }

  void updateObservation(raisim::World *world) { // only for reset
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    cheetah->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);  // rot_: R_wb
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
  }

  const Eigen::VectorXd& getObservation() {
    obDouble_ << rot_.e().row(2).transpose(), /// body orientation(z-axis in world frame expressed in body frame). 3
        gc_.tail(12), /// joint angles 12
//        bodyLinearVel_, /// body linear velocity. 3
        bodyAngularVel_, /// body angular velocity. 3
        gv_.tail(12), /// joint velocity 12
        previousAction_, /// previous action 12
        prepreviousAction_, /// preprevious action 12
//        jointPosErrorHist_.segment((historyLength_ - 12) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 12) * nJoints_, nJoints_), /// joint History 24
//        jointPosErrorHist_.segment((historyLength_ - 10) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 10) * nJoints_, nJoints_), /// joint History 24
//        jointPosErrorHist_.segment((historyLength_ - 8) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 8) * nJoints_, nJoints_), /// joint History 24
        jointPosErrorHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), /// joint History 24
        jointPosErrorHist_.segment((historyLength_ - 4) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 4) * nJoints_, nJoints_), /// joint History 24
        jointPosErrorHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), /// joint History 24
        rot_.e().transpose() * (footPos_[0].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[1].e() - gc_.head(3)),
        rot_.e().transpose() * (footPos_[2].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[3].e() - gc_.head(3)),  /// relative foot position with respect to the body COM, expressed in the body frame 12
        command_,  /// command 3
        0.0, gc_(0); //x_pos; sensor observation in environment

    /// Observation noise
    bool addObsNoise = true;
    if(addObsNoise) {
      for(int i=0; i<obDim_; i++) {
        if(i<3) {  // orientation
          obDouble_(i) = obDouble_(i)  + uniDist_(gen_) * 0.03;
        } else if(i<15) {  // joint angle
          obDouble_(i) = obDouble_(i)  + uniDist_(gen_) * 0.05;
        } else if(i<18) {  // body angular velocity
          obDouble_(i) = obDouble_(i)  + uniDist_(gen_) * 0.1;
        } else if(i<30) {  // joint velocity
          obDouble_(i) = obDouble_(i)  + uniDist_(gen_) * 0.5;
        } else if(i>=126 && i < 138) {  // foot position
          obDouble_(i) = obDouble_(i)  + uniDist_(gen_) * 0.03; //no noise on command
        } //noise on sensor data in environment
      }
    }
    step++;
    return obDouble_;
  }

  const Eigen::VectorXd& getRobotState(raisim::HeightMap* heightMap_) {
    if(isHeightMap_) {
      robotState_ <<
      bodyLinearVel_,  /// body linear velocity. 3
      (footPos_[0].e()(2) - heightMap_->getHeight(footPos_[0].e()(0), footPos_[0].e()(1))), (footPos_[1].e()(2) - heightMap_->getHeight(footPos_[1].e()(0), footPos_[1].e()(1))),
      (footPos_[2].e()(2) - heightMap_->getHeight(footPos_[2].e()(0), footPos_[2].e()(1))), (footPos_[3].e()(2) - heightMap_->getHeight(footPos_[3].e()(0), footPos_[3].e()(1))),
      /// foot z position 4
              footContactState_[0], footContactState_[1], footContactState_[2], footContactState_[3];   /// foot contact state/probability 4
    }
    else {
      robotState_ <<
      bodyLinearVel_,  /// body linear velocity. 3
      footPos_[0].e()(2), footPos_[1].e()(2), footPos_[2].e()(2), footPos_[3].e()(2),  /// foot z position 4
              footContactState_[0], footContactState_[1], footContactState_[2], footContactState_[3];  /// foot contact state/probability 4
    }
    return robotState_;
  }

  /// If the contact body is not feet
  bool isTerminalState(raisim::World *world, int iteration, int testNumber) {
    auto* cheetah = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    for(auto& contact: cheetah->getContacts()) {  //getContacts() returns a vector of Contact instances
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }
    double exceedFactor = 100000; // how much can max joint torque and speed be exceeded (curriculum)
//    if (iteration > 2500 and testNumber==0){ //only during training
////      exceedFactor = std::max(1, 2 - (iteration-2500) / 5000);
//      exceedFactor = std::max(1, 5 - 4 * (iteration-2500) / 5000);
//    }

    if (std::abs(gv_.tail(nJoints_)(0)) > 40*exceedFactor or std::abs(gv_.tail(nJoints_)(3)) > 40*exceedFactor //hip
        or std::abs(gv_.tail(nJoints_)(6)) > 40*exceedFactor or std::abs(gv_.tail(nJoints_)(9)) > 40*exceedFactor) {
//      std::cout << "Terminate 1" << std::endl;
      return true;
    }
    else if (std::abs(gv_.tail(nJoints_)(1)) > 40*exceedFactor or std::abs(gv_.tail(nJoints_)(4)) > 40*exceedFactor //ab/ad
        or std::abs(gv_.tail(nJoints_)(7)) > 40*exceedFactor or std::abs(gv_.tail(nJoints_)(10)) > 40*exceedFactor) {
//      std::cout << "Terminate 2" << std::endl;
      return true;
    }
    else if (std::abs(gv_.tail(nJoints_)(2)) > 25.8*exceedFactor or std::abs(gv_.tail(nJoints_)(5)) > 25.8*exceedFactor // knee
        or std::abs(gv_.tail(nJoints_)(8)) > 25.8*exceedFactor or std::abs(gv_.tail(nJoints_)(11)) > 25.8*exceedFactor) {
//      std::cout << "Terminate 3" << std::endl;
      return true;
    }
    else if (std::abs(cheetah->getGeneralizedForce()[6]) > 17*exceedFactor or std::abs(cheetah->getGeneralizedForce()[9]) > 17*exceedFactor //hip
        or std::abs(cheetah->getGeneralizedForce()[12]) > 17*exceedFactor or std::abs(cheetah->getGeneralizedForce()[15]) > 17*exceedFactor) {
//      std::cout << "Terminate 4" << std::endl;
      return true;
    }
    else if (std::abs(cheetah->getGeneralizedForce()[7]) > 17*exceedFactor or std::abs(cheetah->getGeneralizedForce()[10]) > 17*exceedFactor //ab/ad
        or std::abs(cheetah->getGeneralizedForce()[13]) > 17*exceedFactor or std::abs(cheetah->getGeneralizedForce()[16]) > 17*exceedFactor) {
//      std::cout << "Terminate 5" << std::endl;
      return true;
    }
    else if (std::abs(cheetah->getGeneralizedForce()[8]) > 26.3*exceedFactor or std::abs(cheetah->getGeneralizedForce()[11]) > 26.3*exceedFactor //knee
        or std::abs(cheetah->getGeneralizedForce()[14]) > 26.3*exceedFactor or std::abs(cheetah->getGeneralizedForce()[17]) > 26.3*exceedFactor) {
//      std::cout << "Terminate 6" << std::endl;
      return true;
    }
    else if (hurdleTraining_ and gc_[0]>2.0 and gv_[0]<0.25){ //to prevent robot from stopping in front of hurdle
//      std::cout << "Terminate 7" << std::endl;
      return true;
    }
    else if (hurdleTraining_ and gv_[0]<0.1 and step>30){ //to prevent robot from standing stil
//      std::cout << "Terminate 8" << std::endl;
      return true;
    }
    return false;
  }

  int getObDim() {
    return obDim_;
  }

  int getRobotStateDim() {
    return robotStateDim_;
  }

  int getActionDim() {
    return actionDim_;
  }

  void setIsHeightMap(bool isHeightMap) { isHeightMap_ = isHeightMap;}

  void printTest() {
//    std::cout << "Test1: Observation before normalization." << std::endl;
//    std::cout << obDouble_ << std::endl;
//    std::cout << "joint target after scaling: " << std::endl;
//    std::cout << pTarget12_ << std::endl;

//    std::cout << "Observation Test for debugging!" << std::endl;
//    std::cout << "Observation: " << std::endl;
//    std::cout << getObservation() << std::endl;
//    std::cout << "control com: "<< command_ << std::endl;
//    std::cout << "RF: "<< gc_.segment(7, 3) << std::endl;
//    std::cout << "LF: "<< gc_.segment(10, 3) << std::endl;
//    std::cout << "RH: "<< gc_.segment(13, 3) << std::endl;
//    std::cout << "FF: "<< gc_.segment(16, 3) << std::endl;
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
  int obDim_=0, actionDim_=0, robotStateDim_;
  int historyLength_;
  Eigen::VectorXd stepData_;
  Eigen::VectorXd airTime_, stanceTime_;
  std::vector<std::string> stepDataTag_;
  Eigen::VectorXd jointPosErrorHist_, jointVelHist_, historyTempMem_, preJointVel_;
  Eigen::VectorXd robotState_;
  Eigen::VectorXd jointFrictions_;
  Eigen::VectorXd jointPgain_, jointDgain_;

  Eigen::Vector3d command_;
  bool standingMode_;
  bool isHeightMap_;
  double maxBodyHeight_,  maxXPos_;
  bool hurdleTraining_;
  int step;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937 raisim::MinicheetahController::gen_;
thread_local std::normal_distribution<double> raisim::MinicheetahController::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::MinicheetahController::uniDist_(-1., 1.);
}