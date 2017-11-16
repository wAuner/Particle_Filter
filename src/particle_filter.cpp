#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

// required for Udacity code
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  debug_mode = false;

  if (debug_mode) num_particles = 10;
  else num_particles = 1000;

  std::default_random_engine random_engine;

  // define distributions
  std::normal_distribution<double> norm_x(x, std[0]);
  std::normal_distribution<double> norm_y(y, std[1]);
  std::normal_distribution<double> norm_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle temp;
    temp.id = i;
    temp.x = norm_x(random_engine);
    temp.y = norm_y(random_engine);
    temp.theta = norm_theta(random_engine);
    temp.theta = normalizeAngle(temp.theta);
    temp.weight = 1.;
    particles.push_back(temp);
  }

  if (debug_mode) {
    for (auto &particle :particles) {
      cout << "\nInitialized particle #" << particle.id << ":\n";
      cout << "x: " << particle.x << endl;
      cout << "y: " << particle.y << endl;
    }
  }

  is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine random_engine;

  if (debug_mode) {
   cout << "Prediction...\n";
   cout << "===============================================\n";
  }

  for (auto &part : particles) {
    if (debug_mode) {
      cout << "before pred: Particle " << part.id << " x: " << part.x << " y: " << part.y
           << " theta: " << part.theta << endl;
    }
    // use motion model for prediction
    if (fabs(yaw_rate) > 1e-3) {
      part.x += velocity / yaw_rate * (sin(part.theta + yaw_rate * delta_t) - sin(part.theta));
      part.y += velocity / yaw_rate * (cos(part.theta) - cos(part.theta + yaw_rate * delta_t));
      part.theta += yaw_rate * delta_t;
    } else {
      part.x += velocity * delta_t * cos(part.theta);
      part.y += velocity * delta_t * sin(part.theta);
    }
    // create distributions around predicted mean
    std::normal_distribution<double> norm_x(part.x, std_pos[0]);
    std::normal_distribution<double> norm_y(part.y, std_pos[1]);
    std::normal_distribution<double> norm_theta(part.theta, std_pos[2]);

    // update values with noisy predictions
    part.x = norm_x(random_engine);
    part.y = norm_y(random_engine);
    part.theta = norm_theta(random_engine);
    part.theta = normalizeAngle(part.theta);

    if (debug_mode) {
      cout << "Predicted: Particle " << part.id << " x: " << part.x << " y: " << part.y
           << " theta: " << part.theta << endl << endl;
    }
  }
}


void ParticleFilter::dataAssociation(const Map& mapLandmarks, std::vector<LandmarkObs>& measurements) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  if (debug_mode) {
    cout << "\nData association...\n" ;
    cout << "===============================================\n";
  }

  for (auto &meas : measurements) {
    double closest_dist = 1e4;
    for (const Map::single_landmark_s &landmark : mapLandmarks.landmark_list) {
      double dist = sqrt(pow(meas.x - landmark.x_f, 2) + pow(meas.y - landmark.y_f, 2));
      // assign closest landmark to measurement
      if (dist < closest_dist) {
        closest_dist = dist;
        meas.id = landmark.id_i;
      }
    }
    if (debug_mode) {
      cout << "Associated observation (" << meas.x << " / " << meas.y << ")" << " -> Landmark #"
           << meas.id << " (" << mapLandmarks.landmark_list[meas.id - 1].x_f << " / "
           << mapLandmarks.landmark_list[meas.id - 1].y_f << ")\n";
    }
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
     /*
	   TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	   NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	   according to the MAP'S coordinate system. You will need to transform between the two systems.
	   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	   The following is a good resource for the theory:
	   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	   and the following is a good resource for the actual equation to implement (look at equation
	   3.33
	   http://planning.cs.uiuc.edu/node99.html
	   */

  if (debug_mode) {
    cout << "Updating weights...\n";
    cout << "===============================================\n";
  }

  for (auto &particle : particles) {
    // transform measurements in map coordinates, relative to each particle
    std::vector<LandmarkObs> obs_map_coord = car2MapCoordinates(observations, particle);
    // associate measurements with nearest landmark in map
    dataAssociation(map_landmarks, obs_map_coord);
    // update weights
    particle.weight = 1.0;
    for (auto &measurement : obs_map_coord) {
      // get the coordinates of the associated landmark
      Map::single_landmark_s nearest_landmark = map_landmarks.landmark_list[measurement.id - 1];
      double &x = measurement.x;
      double &y = measurement.y;
      auto x_landmark = static_cast<double>(nearest_landmark.x_f);
      auto y_landmark = static_cast<double>(nearest_landmark.y_f);
      // factorial over the probabilities for all measurement points
      particle.weight *= probMultiVariate(x, y, x_landmark, y_landmark, std_landmark[0], std_landmark[1]);
    }
    if (debug_mode) {
      cout << "\n-> Weight after update of particle #" << particle.id << ": " << particle.weight <<"\n";
    }
  }
}


// maps relative measurements to particle and then to absolute map coordinates
std::vector<LandmarkObs> ParticleFilter::car2MapCoordinates(const std::vector<LandmarkObs> &observations,
                                               const Particle particle) {
  std::vector<LandmarkObs> map_coord = observations;
  if (debug_mode) {
    cout << "\nTransforming measurement coordinates for particle #" << particle.id << ":\n";
    cout << "Particle x,y, theta: (" << particle.x << " / " << particle.y << "), theta: "
         << particle.theta << endl;
    cout << "===============================================\n";
  }

  for (auto &obs : map_coord) {
    if (debug_mode) cout << "(" << obs.x << " / " << obs.y << ") -> ";

    obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
    obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
    obs.id = -1;

    if (debug_mode) cout << "(" << obs.x << " / " << obs.y << ") \n";
  }
  return map_coord;
}


// returns the probability density of an x,y point according to the multivariate normal
// at x_mean, y_mean with std_dev std_x, std_y
double ParticleFilter::probMultiVariate(double x_coord, double y_coord, double x_mean, double y_mean, double std_x, double std_y){

  double term_x = pow(x_coord - x_mean, 2) / (2 * pow(std_x, 2));
  double term_y = pow(y_coord - y_mean, 2) / (2 * pow(std_y, 2));

  return 1. / (2. * M_PI * std_x * std_y) * std::exp(-(term_x + term_y));
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  if (debug_mode) {
    cout << "\nResampling particles...\n";
    cout << "===============================================\n";
  }
  std::default_random_engine gen;

  // create vector with particle weights
  std::vector<double> particle_probs;

  int counter = 0;
  for (auto &particle : particles) {
    particle_probs.push_back(particle.weight);

    if (debug_mode) {
      cout << "Index: " << counter << " Particle id: " << particle.id << " weight: "
           << particle.weight << endl;
    }
    counter++;
  }

  // create distribution to draw indices proportional to particle weights
  std::discrete_distribution<int> weight_dist(particle_probs.begin(), particle_probs.end());
  // sample particles by drawing from the above distribution w/ replacement
  std::vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; ++i) {
    resampled_particles.push_back(particles[weight_dist(gen)]);
  }
  // exchange old particle list with new resampled list
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

double ParticleFilter::normalizeAngle(double angle)
{
  if (angle > 2 * M_PI)
    return fmod(angle, 2 * M_PI);
  else if (angle < 0)
    return fmod(angle, 2 * M_PI) + 2 * M_PI;

  return angle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
