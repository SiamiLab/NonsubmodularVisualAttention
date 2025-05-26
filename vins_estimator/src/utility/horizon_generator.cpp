#include "horizon_generator.h"

HorizonGenerator::HorizonGenerator(ros::NodeHandle nh)
: nh_(nh)
{
  pub_horizon_ = nh_.advertise<nav_msgs::Path>("horizon", 10);

  // load ground truth data if available
  std::string data_csv;
  if (nh_.getParam("gt_data_csv", data_csv)) {
    loadGroundTruth(data_csv);
  }

  // load user commands and velocity for bicycle model
  std::string user_cmd_csv;
  if (nh_.getParam("bicycle_model_data_csv", user_cmd_csv)) {
    loadBicycleData(user_cmd_csv);
  }

  vel_x = 0;
  vel_y = 0;
  steering_angle = 0;
}

// ----------------------------------------------------------------------------

void HorizonGenerator::setParameters(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
  q_IC_ = R;
  t_IC_ = t;
}

// ----------------------------------------------------------------------------

state_horizon_t HorizonGenerator::imu(
                      const state_t& state_0, const state_t& state_1,
                      const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                      int nrImuMeasurements, double deltaImu)
{
  state_horizon_t state_kkH;

  // initialize with the the last frame's pose (tail of optimizer window)
  state_kkH[0] = state_0;

  // Additionally, since we already have an IMU propagated estimate
  // (yet-to-be-corrected) of the current frame, we will start there.
  state_kkH[1] = state_1;

  // let's just assume constant bias over the horizon
  auto Ba = state_kkH[0].first.segment<3>(xB_A);

  // we also assume a constant angular velocity during the horizon
  auto Qimu = Utility::deltaQ(w * deltaImu);

  for (int h=2; h<=HORIZON; ++h) { // NOTE: we already have k+1.

    // use the prev frame state to initialize the k+h frame state
    state_kkH[h] = state_kkH[h-1];

    // constant acceleration IMU propagation
    for (int i=0; i<nrImuMeasurements; ++i) {

      // propagate attitude with incremental IMU update
      state_kkH[h].second = state_kkH[h].second * Qimu;

      // Convenience: quat from world to current IMU-rate body pose
      const auto& q_hi = state_kkH[h].second;

      // vi, eq (11)
      state_kkH[h].first.segment<3>(xVEL) += (gravity + q_hi*(a - Ba))*deltaImu;
      
      // ti, second equality in eq (12)
      state_kkH[h].first.segment<3>(xPOS) += state_kkH[h].first.segment<3>(xVEL)*deltaImu + 0.5*gravity*deltaImu*deltaImu + 0.5*(q_hi*(a - Ba))*deltaImu*deltaImu;
    }

  }

  return state_kkH;
}


// Kian: implementation of the bicycle model prediction
state_horizon_t HorizonGenerator::bicycle_model(const state_t& state_0, const state_t& state_1, double deltaFrame)
{


  // reading from file - if you want from bag comment these lines
  // get the timestamp of the previous frame
  double timestamp = state_0.first.coeff(xTIMESTAMP);

  // if this condition is true, then it is likely the first state_0 (which may have random values)
  if (timestamp > bicycle_data_.back().timestamp) timestamp = bicycle_data_.front().timestamp;

  // naive time synchronization with the previous image frame and ground truth
  int seeker = 0;
  while (seeker < static_cast<int>(bicycle_data_.size()) && 
                          bicycle_data_[seeker++].timestamp <= timestamp);
  int idx = seeker-1;

  double vel_x_ = bicycle_data_[idx].v.x();
  double vel_y_ = bicycle_data_[idx].v.y();
  double velocity_ = std::sqrt(vel_x_ * vel_x_ + vel_y_ * vel_y_) + 0.5;
  double steering_angle_ = bicycle_data_[idx].steering_angle;

  // if using from bag
  // steering angle is stored in 'steering_angle' var and linear velocity in 'vel_x', and 'vel_y' variables.
  // so change steering_angle_ to steering_angle and velocity to velocity_ and vice versa
  // double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y) + 0.5;



  state_horizon_t state_kkH;
  constexpr double wheelbase = 0.26; // meters

  // extract initial pose (x, y, yaw)
  Eigen::Vector3d pos = state_0.first.segment<3>(xPOS);
  Eigen::Quaterniond orientation = state_0.second;
  double yaw = atan2(2.0 * (orientation.w() * orientation.z() + orientation.x() * orientation.y()),
                     1.0 - 2.0 * (orientation.y() * orientation.y() + orientation.z() * orientation.z()));


  // set initial state
  state_kkH[0].first = state_0.first;
  state_kkH[0].second = state_0.second;


  for (int h = 1; h <= HORIZON; ++h)
  {
    // Bicycle model update
    double dt = deltaFrame;
    pos.x() += velocity_ * std::cos(yaw) * dt;
    pos.y() += velocity_ * std::sin(yaw) * dt;
    yaw += (velocity_ / wheelbase) * std::tan(steering_angle_) * dt;

    // Update quaternion from yaw
    Eigen::Quaterniond q_next(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    // Fill state horizon
    state_kkH[h].first = state_kkH[h - 1].first;  // Copy previous state
    state_kkH[h].first.segment<3>(xPOS) = pos;
    state_kkH[h].second = q_next;
  }

  
  return state_kkH;
}

void HorizonGenerator::loadBicycleData(std::string data_csv)
{
  // open the CSV file and create an iterator
  std::ifstream file(data_csv);
  CSVIterator it(file);

  // throw away the headers
  ++it;

  bicycle_data_.clear();

  for (; it != CSVIterator(); ++it) {
    bicycle_data_t data;

    data.timestamp = std::stod((*it)[0])*1e-9; // convert ns to s
    data.throttle = std::stod((*it)[1]);
    data.steering_angle = std::stod((*it)[2]);
    data.v << std::stod((*it)[3]), std::stod((*it)[4]), std::stod((*it)[5]);

    bicycle_data_.push_back(data);
  }
  // for(const auto& data : bicycle_data_)
  //   std::cout << "Kian: " << data.timestamp << ", " << data.steering_angle << ", " << data.v.z() << std::endl;
}

// ----------------------------------------------------------------------------

state_horizon_t HorizonGenerator::groundTruth(const state_t& state_0,
                                    const state_t& state_1, double deltaFrame)
{
  state_horizon_t state_kkH;

  // get the timestamp of the previous frame
  double timestamp = state_0.first.coeff(xTIMESTAMP);

  // if this condition is true, then it is likely the first state_0 (which may have random values)
  if (timestamp > truth_.back().timestamp) timestamp = truth_.front().timestamp;

  // naive time synchronization with the previous image frame and ground truth
  seek_idx_ = 0;
  while (seek_idx_ < static_cast<int>(truth_.size()) && 
                          truth_[seek_idx_++].timestamp <= timestamp);
  int idx = seek_idx_-1;


  // initialize state horizon structure with [xk]. The rest are future states.
  state_kkH[0].first = state_0.first;
  state_kkH[0].second = state_0.second;

  // ground-truth pose of previous frame
  Eigen::Vector3d prevP = truth_[idx].p;
  Eigen::Quaterniond prevQ = truth_[idx].q;

  // predict pose of camera for frames k to k+H
  for (int h=1; h<=HORIZON; ++h) {

    // while the inertial frame of ground truth and vins will be different,
    // it is not a problem because we only care about _relative_ transformations.
    auto gtPose = getNextFrameTruth(idx, deltaFrame);

    // Ground truth orientation of frame k+h w.r.t. orientation of frame k+h-1
    auto relQ = prevQ.inverse() * gtPose.q;

    // Ground truth position of frame k+h w.r.t. position of frame k+h-1
    auto relP = gtPose.q.inverse() * (gtPose.p - prevP);


    // "predict" where the current frame in the horizon (k+h)
    // will be by applying this relative rotation to
    // the previous frame (k+h-1)
    state_kkH[h].first.segment<3>(xPOS) = state_kkH[h-1].first.segment<3>(xPOS) + state_kkH[h-1].second * relP;
    state_kkH[h].second = state_kkH[h-1].second * relQ;

    // for next iteration
    prevP = gtPose.p;
    prevQ = gtPose.q;
  }
  
  return state_kkH;
}

// ----------------------------------------------------------------------------

void HorizonGenerator::visualize(const std_msgs::Header& header,
                                 const state_horizon_t& state_kkH)
{
  // Don't waste cycles unnecessarily
  if (pub_horizon_.getNumSubscribers() == 0) return;

  nav_msgs::Path path;
  path.header = header;
  path.header.frame_id = "world";

  // include the current state, xk (i.e., h=0)
  for (int h=0; h<=HORIZON; ++h) {

    // for convenience
    const auto& x_h = state_kkH[h].first;
    const auto& q_h = state_kkH[h].second;

    // Compose world-to-imu estimate with imu-to-cam extrinsic transform
    Eigen::Vector3d P = x_h.segment<3>(xPOS) + q_h * t_IC_;
    Eigen::Quaterniond R = q_h * q_IC_;

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = P.x();
    pose.pose.position.y = P.y();
    pose.pose.position.z = P.z();
    pose.pose.orientation.w = R.w();
    pose.pose.orientation.x = R.x();
    pose.pose.orientation.y = R.y();
    pose.pose.orientation.z = R.z();

    path.poses.push_back(pose);
  }


  pub_horizon_.publish(path);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void HorizonGenerator::loadGroundTruth(std::string data_csv)
{
  // open the CSV file and create an iterator
  std::ifstream file(data_csv);
  CSVIterator it(file);

  // throw away the headers
  ++it;

  truth_.clear();

  for (; it != CSVIterator(); ++it) {
    truth_t data;

    data.timestamp = std::stod((*it)[0])*1e-9; // convert ns to s
    data.p << std::stod((*it)[1]), std::stod((*it)[2]), std::stod((*it)[3]);
    data.q = Eigen::Quaterniond(std::stod((*it)[4]), std::stod((*it)[5]), std::stod((*it)[6]), std::stod((*it)[7]));
    data.v << std::stod((*it)[8]), std::stod((*it)[9]), std::stod((*it)[10]);
    data.w << std::stod((*it)[11]), std::stod((*it)[12]), std::stod((*it)[13]);
    data.a << std::stod((*it)[14]), std::stod((*it)[15]), std::stod((*it)[16]);

    truth_.push_back(data);
  }

  // reset seek
  seek_idx_ = 0;

}

// ----------------------------------------------------------------------------

HorizonGenerator::truth_t HorizonGenerator::getNextFrameTruth(int& idx,
                                                            double deltaFrame)
{
  double nextTimestep = truth_[idx].timestamp + deltaFrame;

  // naive time synchronization with the previous image frame and ground truth
  while (idx < static_cast<int>(truth_.size()) && 
                          truth_[idx++].timestamp <= nextTimestep);

  return truth_[idx];
}

void HorizonGenerator::user_command_callback(const geometry_msgs::Vector3Stamped &msg)
{
  // std::cout << "Kian (in)" << msg.vector.x << ", " << msg.vector.y << std::endl;
  steering_angle = msg.vector.y;
}

void HorizonGenerator::velocity_encoder_callback(const geometry_msgs::Vector3Stamped &msg)
{
  // std::cout << "Kian (in)" << msg.vector.x << ", " << msg.vector.y << ", " << msg.vector.z << std::endl;
  vel_x = msg.vector.x;
  vel_y = msg.vector.y;
}
