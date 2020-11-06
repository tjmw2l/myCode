#pragma once
#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class MyCostFactor : public ceres::SizedCostFunction<1,1>
{
  public:
    MyCostFactor(const double depthFromStereo):stereo_depth_i(depthFromStereo){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
      const double estimatedDepth=parameters[0][0];
      residuals[0]=10*(stereo_depth_i-estimatedDepth);
      //residuals[0]=10*(stereo_depth_i-estimatedDepth)/(stereo_depth_i+estimatedDepth);
      //double cost_function=exp(2*(stereo_depth_i-estimatedDepth)/(stereo_depth_i+estimatedDepth));
      //residuals[0]=cost_function;

      if (jacobians != NULL && jacobians[0] != NULL)
      {
        //jacobians[0][0] = -10*-2*(stereo_depth_i/pow(stereo_depth_i+estimatedDepth,2));
        jacobians[0][0] = -10;
        //jacobians[0][0] = -4*(stereo_depth_i/pow(stereo_depth_i+estimatedDepth,2))*cost_function;
      }

      return true;
    }

    double stereo_depth_i;
};
