//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_
#define _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_

#include "raisim/World.hpp"

namespace raisim {

class RandomHeightMapGenerator {
 public:

  enum class GroundType : int {
    HEIGHT_MAP = 0,
    HEIGHT_MAP_DISCRETE = 4,
    STEPS = 2,
    STAIRS = 3,
    Slope = 1
  };

  RandomHeightMapGenerator() = default;

  void setSeed(int seed) {
    terrain_seed_ = seed;
  }

  raisim::HeightMap* generateTerrain(raisim::World* world,
                                     GroundType groundType,
                                     double curriculumFactor,
                                     bool createHoles,
                                     std::mt19937& gen,
                                     std::uniform_real_distribution<double>& uniDist) {
    std::vector<double> heightVec, heightVec1, heightVec2;
    heightVec.resize(heightMapSampleSize_*heightMapSampleSize_);
    std::unique_ptr<raisim::TerrainGenerator> genPtr;
    double targetRoughness = 0.15;

    switch (groundType) {
      case GroundType::HEIGHT_MAP: {
        terrainProperties_.frequency = 0.8;
        terrainProperties_.zScale = targetRoughness * curriculumFactor * 1.4;
        terrainProperties_.xSize = 20.0;
        terrainProperties_.ySize = 20.0;
        terrainProperties_.xSamples = 50;
        terrainProperties_.ySamples = 50;
        terrainProperties_.fractalOctaves = 5;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_++;
        terrainProperties_.stepSize = 0.;
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();

        return world->addHeightMap(terrainProperties_.xSamples, terrainProperties_.ySamples, terrainProperties_.xSize, terrainProperties_.ySize, 0., 0., heightVec);
        break;
      }

      case GroundType::HEIGHT_MAP_DISCRETE:
      {
        terrainProperties_.frequency = 0.3;
        terrainProperties_.zScale = targetRoughness * curriculumFactor * 1.2;
        terrainProperties_.xSize = 8.0;
        terrainProperties_.ySize = 8.0;
        terrainProperties_.xSamples = 80;
        terrainProperties_.ySamples = 80;
        terrainProperties_.fractalOctaves = 3;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_++;
        terrainProperties_.stepSize = 0.04 * curriculumFactor;
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();

        return world->addHeightMap(80, 80, 8.0, 8.0, 0., 0., heightVec);
        break;
      }
      case GroundType::STEPS:
      {
        heightVec.resize(120*120);
        for(int xBlock = 0; xBlock < 15; xBlock++) {
          for(int yBlock = 0; yBlock < 15; yBlock++) {
            double height = 0.04 * uniDist(gen) * curriculumFactor;
            for(int i=0; i<8; i++) {
              for(int j=0; j<8; j++) {
                heightVec[120 * (8*xBlock+i) + (8*yBlock+j)] = height + xBlock * 0.05 * curriculumFactor;
              }
            }
          }
        }

        return world->addHeightMap(120, 120, 8.0, 8.0, 0., 0., heightVec);
        break;
      }

      case GroundType::STAIRS:
      {
        heightVec.resize(2*3000);
        double curriculumFactor2 = uniDist(gen);  // 50% highest step height, 50% random height
        bool is_train = true;

        if (is_train) {
          if (curriculumFactor2 > 0) {
            for (int xBlock = 0; xBlock < 25; xBlock++) {
              for (int i = 0; i < 2 * 3000 / 25; i++) {
                heightVec[xBlock * 2 * 3000 / 25 + i] = xBlock * 0.08 * curriculumFactor;
              }
            }
          } else {
            for (int xBlock = 0; xBlock < 25; xBlock++) {
              for (int i = 0; i < 2 * 3000 / 25; i++) {
                heightVec[xBlock * 2 * 3000 / 25 + i] = xBlock * 0.08 * (-curriculumFactor2) * curriculumFactor;
              }
            }
          }
        }
        else {
          for (int xBlock = 0; xBlock < 25; xBlock++) {
            for (int i = 0; i < 2 * 3000 / 25; i++) {
              heightVec[xBlock * 2 * 3000 / 25 + i] = xBlock * 0.18 * curriculumFactor;
            }
          }
        }
        double y_size = uniDist(gen) * 2.5 + 9.5;  // [7m, 12m] / 25 = [28cm, 48cm]
        //        std::cout << "curriculum factor: " << curriculumFactor2 << std::endl;
        //        std::cout << "y_size: " << y_size << std::endl;

        return world->addHeightMap(2, 3000, 7.25, y_size, 0., 0., heightVec);
        break;

      }

      case GroundType::Slope:
      {
        double curriculumFactor2 = uniDist(gen);
        double min_height=0;
        terrainProperties_.frequency = 0.8;
        terrainProperties_.zScale = targetRoughness * curriculumFactor * 1.4;
        terrainProperties_.xSize = 20.0;
        terrainProperties_.ySize = 20.0;
        terrainProperties_.xSamples = 50;
        terrainProperties_.ySamples = 50;
        terrainProperties_.fractalOctaves = 5;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_++;
        terrainProperties_.stepSize = 0.;
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec1 = genPtr->generatePerlinFractalTerrain();

        if (curriculumFactor2 > 0){
          for (int xBlock = 0; xBlock < terrainProperties_.xSamples; xBlock++) {
            for (int i = 0; i < terrainProperties_.ySamples; i++) {
              heightVec1[xBlock * terrainProperties_.ySamples + i] += xBlock * tan(10/360.*2*M_PI)*(terrainProperties_.xSize/terrainProperties_.xSamples) * curriculumFactor* curriculumFactor2;
              min_height=fmin(min_height,heightVec1[xBlock * terrainProperties_.ySamples + i]);
            }
          }
        }
        else {
          for (int xBlock = 0; xBlock < terrainProperties_.xSamples; xBlock++) {
            for (int i = 0; i < terrainProperties_.ySamples; i++) {
              heightVec1[xBlock * terrainProperties_.ySamples + i] += xBlock * tan(10/360.*2*M_PI)*(terrainProperties_.xSize/terrainProperties_.xSamples) * curriculumFactor* curriculumFactor2;
              min_height=fmin(min_height,heightVec1[xBlock * terrainProperties_.ySamples + i]);
            }
          }
        }

        for (int xBlock = 0; xBlock < terrainProperties_.xSamples; xBlock++) {
          for (int i = 0; i < terrainProperties_.ySamples; i++) {
            heightVec1[xBlock * terrainProperties_.ySamples + i] -= min_height;
          }
        }

        return world->addHeightMap(terrainProperties_.xSamples, terrainProperties_.ySamples, terrainProperties_.xSize, terrainProperties_.ySize, 0., 0., heightVec1);
        break;
      }
    }
    return nullptr;
  }

 private:
  raisim::TerrainProperties terrainProperties_;
  int heightMapSampleSize_ = 120;
  int terrain_seed_;
};

}

#endif //_RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_