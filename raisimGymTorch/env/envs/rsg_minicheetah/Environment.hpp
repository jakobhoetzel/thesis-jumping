// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once
#include <algorithm>

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
    READ_YAML(double, rewCurriculumFactor2_, cfg["rew_curriculum_factor"])
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
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::HURDLES], cfg["reward"]["hurdlesCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::SYMMETRY], cfg["reward"]["symmetryCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::FOOTCONTACT], cfg["reward"]["footContactCoeff"])
    READ_YAML(double, rewardCoeff_[MinicheetahController::RewardType::FEETFORWARDJUMP], cfg["reward"]["feetForwardJumpCoeff"])

    terrain_curriculum_ = terCurriculumFactor_*0.25;
    testNumber = 0;
    isHeightMap_ = cfg["isHeightMap"].template As<bool>();
    controller_.setIsHeightMap(isHeightMap_);
    if (isHeightMap_){
      heightMap_ = terrainGenerator_.generateTerrain(world_.get(), RandomHeightMapGenerator::GroundType(0), 1.0, false, gen_, uniDist_);
    }
    else {
      world_->addGround();
    }
    xPos_Hurdles_ = uniDist_(gen_)*0.5 + 5.0;
    double p = std::abs(uniDist_(gen_));
    hurdleHeight_ = terrain_curriculum_;
    if(p<0.15 and testNumber==0){ //train with lower hurdles
      double val = uniDist_(gen_);
      hurdleHeight_ = std::max(0.1, terrain_curriculum_*(1.0/2.0 + 1.0/2.0*val));
    }
    auto hurdle1_ = world_->addBox(0.1, 500, hurdleHeight_, 100000); //x, y, z length, mass; change also in init
    hurdle1_->setPosition(xPos_Hurdles_, 0, hurdleHeight_/2.0+0.15); //pos of cog
    hurdle1_->setOrientation(1., 0, 0, 0); //quaternion
    hurdle1_->setName("hurdle1");


    stepData_.resize(controller_.getStepDataTag().size());

    /// visualize if it is the first environment
    if (visualizable_) {  //RaisimUnity
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot);
    }
    iteration = 0;

    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world_->getObject("robot"));
    previousError = Eigen::VectorXd::Zero(cheetah->getDOF()-6); //when self coded pd controller is used for torque limit
    pTarget_ = Eigen::VectorXd::Zero(cheetah->getDOF()-6); //when self coded pd controller is used for torque limit
    friction = Eigen::VectorXd::Zero(cheetah->getDOF()-6); //when self coded pd controller is used for torque limit
  }

  ~ENVIRONMENT() {
    if(server_) server_->killServer();
  }

  void init() {
  }

  void reset() {
    double p = uniDist_(gen_); //p between -1 and 1
    hurdleTraining = true; // testNumber = 1: test of jumping -> always hurdles
    if (p < -1.0 and testNumber==0){ // training -> hurdles according to probability (set probability here)
      hurdleTraining = false;
    } else if(testNumber==2){
      hurdleTraining = false; // test of running without hurdles
    }
    controller_.reset(world_.get(), comCurriculumFactorT_, heightMap_, hurdleTraining);
    controller_.collisionRandomization(world_.get());
    std::tie(jointPgain_, jointDgain_) = controller_.getPDGain_self(); //when self coded pd controller is used for torque limit

    mu_ = 0.4 + 0.3 * (uniDist_(gen_) + 1);  // [0.4, 1.0]
    world_->setDefaultMaterial(mu_, 0, 0);

    auto hurdle1_ = world_->getObject("hurdle1");
    xPos_Hurdles_ = uniDist_(gen_)*0.5 + 5.0;
    world_->removeObject(hurdle1_);
    if (hurdleTraining){
      p = std::abs(uniDist_(gen_));
      hurdleHeight_ = terrain_curriculum_;
      if(p<0.15 and testNumber==0){ //train with lower hurdles
        double val = uniDist_(gen_);
        hurdleHeight_ = std::max(0.1, terrain_curriculum_*(1.0/2.0 + 1.0/2.0*val));
      }
      auto hurdle2_ = world_->addBox(0.1, 500, hurdleHeight_, 100000); //x, y, z length, mass; change also in init
      hurdle2_->setPosition(xPos_Hurdles_, 0, hurdleHeight_/2.0+0.15); //pos of cog
      hurdle2_->setOrientation(1., 0, 0, 0); //quaternion
      hurdle2_->setName("hurdle1");
    }else{
      auto hurdle2_ = world_->addBox(0, 0, 0, 0); //no hurdle
      hurdle2_->setPosition(xPos_Hurdles_, 0, 0); //pos of cog
      hurdle2_->setOrientation(1., 0, 0, 0); //quaternion
      hurdle2_->setName("hurdle1");
    }

  }

  const std::vector<std::string>& getStepDataTag() {
    return controller_.getStepDataTag();
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

  void setCommand(const Eigen::Ref<EigenVec>& command, int testNumber_=0) {
    controller_.setCommand(command);
    if (testNumber_==0 or testNumber_==1 or testNumber_==2){
      testNumber = testNumber_; // 0=training(hurdles according to probability), 1=test jumping(with hurdle), 0=train run (without hurdle)
    } else{
      std::cout << "Unknown test number: " << testNumber_ << std:: endl;
    }
  }

  void go_straight_controller() { controller_.go_straight_controller(); }

  double step(const Eigen::Ref<EigenVec> &action) {
    stepData_.setZero();
    int loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
//    delayDividedBySimdt = int((0.01 / simulation_dt_)*0.5*(uniDist_(gen_)+1));
    delayDividedBySimdt = int((0.01 / simulation_dt_) - 1e-10);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(i == delayDividedBySimdt){
        controller_.advance(world_.get(), action);
        pTarget_ = controller_.getPDTarget_self(); //when self coded pd controller is used for torque limit
        friction = controller_.getJointFriction_self(); //must be set negative!
      }
//      setGeneralizedForce_self(); //when self coded pd controller is used for torque limit, remove when using built-in pd controller
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();  // What does integration do? A. Simulate robot states and motions for the next simulation time.
      if (server_) server_->unlockVisualizationServerMutex();
      controller_.getReward(world_.get(), rewardCoeff_, simulation_dt_, rewCurriculumFactor_, heightMap_, xPos_Hurdles_, hurdleHeight_, iteration);
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
    double dist_obs_next = 0;
    if ((xPos_Hurdles_-ob.tail(1)(0)-0.15) >= 0) { // head before hurdle
      dist_obs_next = std::min( std::max(xPos_Hurdles_-ob.tail(1)(0)-0.15, 0.0), 5.0); //distance between 0 and 5
    } else if ((xPos_Hurdles_-ob.tail(1)(0)-0.15) >= -0.3){ // before landing
      dist_obs_next = xPos_Hurdles_-ob.tail(1)(0)-0.15;
    } else { // after hurdle
      dist_obs_next = 5; //distance between 0 and 8
    }
    if(hurdleTraining){
      ob.tail(2) << hurdleHeight_+uniDist_(gen_) * 0.05, dist_obs_next+uniDist_(gen_) * 0.15;
//      ob.tail(2) << hurdleHeight_, dist_obs_next;
    } else{
      ob.tail(2) << uniDist_(gen_) * 0.05, 5 + uniDist_(gen_) * 0.05; //no hurdle
    }
  }

  void getRobotState(Eigen::Ref<EigenVec> ob) {  // related to the estimator network learning
    ob = controller_.getRobotState(heightMap_).cast<float>();
  }

  void setGeneralizedForce_self() {  //when self coded pd controller is used for torque limit
    auto *cheetah = reinterpret_cast<raisim::ArticulatedSystem *>(world_->getObject("robot"));
    jointState = controller_.getJointState_self();

    Eigen::VectorXd error = pTarget_ - jointState;

    Eigen::VectorXd genForce = Eigen::VectorXd::Zero(cheetah->getDOF());

    genForce.tail(cheetah->getDOF() - 6) = jointPgain_.cwiseProduct(error) + jointDgain_.cwiseProduct((error-previousError)/simulation_dt_);
    genForce = genForce-friction;

    double exceedFactor = std::max(1.0, 2.0 - iteration/5000.0);

    for(int i=6; i<cheetah->getDOF(); i++){
      if((i-6)%3==0 or (i-6)%3==1){ //both hip joints
        genForce(i) = std::max(-17.0, std::min(genForce(i), 17.0));
      }else{ //knee joints
        genForce(i) = std::max(-26.3*exceedFactor, std::min(genForce(i), 26.3*exceedFactor));
      }
    }

    if (previousError.isApprox(Eigen::VectorXd::Zero(cheetah->getDOF()-6), 1.e-50)){
      cheetah->setGeneralizedForce(Eigen::VectorXd::Zero(cheetah->getDOF())); //for first timesteps after initialization
    } else{
      cheetah->setGeneralizedForce(genForce);
    }
    previousError = error;
  }

  bool isTerminalState(float &terminalReward) {
    if(controller_.isTerminalState(world_.get(), iteration, testNumber)){
      terminalReward = terminalRewardCoeff_;
      return true;
    }
    terminalReward = 0.f;
    return false;
  }
  /////// optional methods ///////
  void curriculumUpdate(int iter) {
//    rewCurriculumFactor_ = pow(rewCurriculumFactor_+1, rewCurriculumRate_);
    rewCurriculumFactor2_ = rewCurriculumFactor2_*rewCurriculumRate_;
    rewCurriculumFactor_ = 1 - rewCurriculumFactor2_;
    comCurriculumFactorT_ = 1 + comCurriculumFactor3_ / (1 + std::exp(-comCurriculumFactor1_ * (iter - comCurriculumFactor2_)));
    comCurriculumFactorT_ = std::fmax(1., comCurriculumFactorT_);
    terrain_curriculum_ = std::min(iter * (terCurriculumFactor_*0.75) / 5000.0 + terCurriculumFactor_*0.25, terCurriculumFactor_);
    iteration = iter;
    if(isHeightMap_) {
      double terrain_curriculum = 1;
      world_->removeObject(heightMap_);
      heightMap_ = terrainGenerator_.generateTerrain(world_.get(), RandomHeightMapGenerator::GroundType(0), terrain_curriculum, false, gen_, uniDist_);
    }
  };
  float getCurriculumFactor() {return float(rewCurriculumFactor_);};
  void close() { if (server_) server_->killServer(); };
  void setSeed(int seed) {
    controller_.setSeed(seed);
    terrainGenerator_.setSeed(seed);
    gen_.seed(seed);
    //groundType_ = seed % 2;
  };
  ////////////////////////////////

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getRobotStateDim() { return controller_.getRobotStateDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return world_.get(); }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  void printTest() { controller_.printTest();
  }

 private:
  std::map<MinicheetahController::RewardType, float> rewardCoeff_;
  bool visualizable_ = false;
  bool isHeightMap_;
  double terminalRewardCoeff_ = -10.;
  MinicheetahController controller_;
  std::unique_ptr<raisim::World> world_;
  double rewCurriculumFactor_=0, rewCurriculumFactor2_, rewCurriculumRate_;
  double comCurriculumFactorT_ = 1., comCurriculumFactor1_, comCurriculumFactor2_, comCurriculumFactor3_;
  double terCurriculumFactor_;
  double terrain_curriculum_; // height of hurdles
  double terrain_curriculum_heightMap_ = 3000; // height of hurdles
  double xPos_Hurdles_;
  double simulation_dt_;
  double control_dt_;
  double mu_;
  double hurdleHeight_;
  int groundType_ = 0; //0  Set ground Type
  int delayDividedBySimdt;
  bool hurdleTraining;
  int testNumber, iteration;
  Eigen::VectorXd jointPgain_, jointDgain_, pTarget_, friction, jointState, previousError; //when self coded pd controller is used for torque limit
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