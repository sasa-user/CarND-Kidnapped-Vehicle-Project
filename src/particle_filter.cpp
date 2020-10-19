/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

constexpr double eps = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p;

    // Sample from these normal distributions
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.id = i;
    p.weight = 1;

    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  double theta_new, x_new, y_new;

  for (auto& particle : particles) {
    // Old values
    double x_old	 = particle.x;
    double y_old	 = particle.y;
    double theta_old = particle.theta;

    // New values depending on the yaw_rate
    if (fabs(yaw_rate) < eps) {
      theta_new = theta_old;
      x_new	= x_old + velocity * delta_t * cos(theta_old);
      y_new = y_old + velocity * delta_t * sin(theta_old);
    } else {
      theta_new = theta_old + yaw_rate * delta_t;
      x_new = x_old + velocity / yaw_rate * (sin(theta_new) - sin(theta_old));
      y_new = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_new));
    }

    // Calculating Gaussian for new values
    normal_distribution<double> dist_x(x_new, std_pos[0]);
    normal_distribution<double> dist_y(y_new, std_pos[1]);
    normal_distribution<double> dist_theta(theta_new, std_pos[2]);

    // Adding values with random Gaussian noise
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double min_dist;
  for (auto& observation : observations) {
    min_dist = std::numeric_limits<double>::max();

    for (const auto& prediction : predicted) {
        double curr_dist = dist(observation.x, observation.y, prediction.x, prediction.y);
        if (curr_dist < min_dist) {
          observation.id = prediction.id;
          min_dist = curr_dist;
        }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  const double gauss_norm = 2 * M_PI * sig_x * sig_y;
  double weight_sum = 0.0;

  for (auto& p : particles) {
    // reseting weight
    p.weight = 1.0;

    // gathering all landmarks in range of the particle
    std::vector<LandmarkObs> landmarks_in_range;
    for (const auto& map_landmark : map_landmarks.landmark_list) {
      LandmarkObs tmpLandmark;
      tmpLandmark.id = map_landmark.id_i;
      tmpLandmark.x = (double) map_landmark.x_f;
      tmpLandmark.y = (double) map_landmark.y_f;

      double range = dist(p.x, p.y, tmpLandmark.x, tmpLandmark.y);
      if (range < sensor_range) {
        landmarks_in_range.push_back(tmpLandmark);
      }
    }

    // transform to map coordinates
    std::vector<LandmarkObs> map_coor_observations;
    for (const auto& observation : observations) {
      map_coor_observations.push_back(LandmarkObs{
        observation.id,
        p.x + cos(p.theta) * observation.x - sin(p.theta) * observation.y,
        p.y + sin(p.theta) * observation.x + cos(p.theta) * observation.y,
      });
    }

    dataAssociation(landmarks_in_range, map_coor_observations);

    // Updating weights
    for(const auto& observation : map_coor_observations) {
      double land_x = eps, land_y = eps;
      for (const auto& landmark: landmarks_in_range) {
        if (landmark.id == observation.id) {
          land_x = landmark.x;
          land_y = landmark.y;
          break;
        }
      }

      double x_obs = observation.x - land_x;
      double y_obs = observation.y - land_y;

      // calculate exponent
      double exponent;
      exponent = (pow(x_obs, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs, 2) / (2 * pow(sig_y, 2)));

      // calculate weight using normalization terms and exponent
      p.weight *= (exp(-exponent) / gauss_norm);
    }
    weight_sum += p.weight;
  }

  // Normalize weights
  for (auto& particle : particles) {
      particle.weight /= (weight_sum + eps);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Get all weights
  vector<double> weights;
  for (const auto& particle : particles) {
    weights.push_back(particle.weight);
  }

  std::discrete_distribution<int> distributed_weights(weights.begin(), weights.end());

  // Create new particles
  vector<Particle> new_particles;
  int index;
  for (int i = 0; i < num_particles; ++i) {
    index = distributed_weights(gen);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
