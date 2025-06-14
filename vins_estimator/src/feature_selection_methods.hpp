#include "feature_selector.h"



std::map<int, omega_horizon_t> FeatureSelector::calcInfoFromFeatures_Full(
    const image_t& image,
    const state_horizon_t& state_kkH,
    std::map<int, Eigen::MatrixXd>& Fs, std::map<int, Eigen::MatrixXd>& Ps)
{
  std::map<int, omega_horizon_t> Delta_ells;

  const auto& t_WC_k1 = state_k1_.first.segment<3>(xPOS) + state_k1_.second * t_IC_;
  const auto& q_WC_k1 = state_k1_.second * q_IC_;

  auto depthsByIdx = initKDTree();

  for (const auto& fpair : image) {
    constexpr int c = 0;
    int feature_id = fpair.first;
    Eigen::Vector3d feature = fpair.second[c].second.head<3>();

    double d = findNNDepth(depthsByIdx, feature.coeff(0), feature.coeff(1));
    feature = feature.normalized() * d;


    auto pell = t_WC_k1 + q_WC_k1 * feature;

    std::vector<Eigen::Matrix<double, 3, 9 * (HORIZON+1)>> F_blocks;
    std::vector<Eigen::Matrix<double, 3, 3>> E_blocks;
    int numVisible = 0;

    //
    // ---- Handle k+1 frame separately (assumed detection frame)
    //
    {
      Eigen::Vector3d uell = (q_WC_k1.inverse() * (pell - t_WC_k1)).normalized();
      Eigen::Vector2d pixels;
      m_camera_->spaceToPlane(uell, pixels);
      if (inFOV(pixels)) {
        Eigen::Matrix<double, 3, 9 * (HORIZON+1)> F_row = Eigen::Matrix<double, 3, 9 * (HORIZON+1)>::Zero();
        Eigen::Matrix3d Bh = Utility::skewSymmetric(uell) * ((q_WC_k1 * q_IC_).inverse()).toRotationMatrix();
        F_row.block<3, 3>(0, 9) = Bh;

        F_blocks.push_back(F_row);
        E_blocks.push_back(-Bh);
        ++numVisible;
      }
    }

    //
    // ---- Forward simulate from k+2 to k+H (h = 2 onward)
    //
    for (int h = 2; h <= HORIZON; ++h) {
      const auto& t_WC_h = state_kkH[h].first.segment<3>(xPOS) + state_kkH[h].second * t_IC_;
      const auto& q_WC_h = state_kkH[h].second * q_IC_;

      Eigen::Vector3d uell = (q_WC_h.inverse() * (pell - t_WC_h)).normalized();
      Eigen::Vector2d pixels;
      m_camera_->spaceToPlane(uell, pixels);
      if (!inFOV(pixels)) continue;
      

      Eigen::Matrix<double, 3, 9 * (HORIZON+1)> F_row = Eigen::Matrix<double, 3, 9 * (HORIZON+1)>::Zero();
      Eigen::Matrix3d Bh = Utility::skewSymmetric(uell) * ((q_WC_h * q_IC_).inverse()).toRotationMatrix();
      F_row.block<3, 3>(0, 9 * h) = Bh;


      F_blocks.push_back(F_row);
      E_blocks.push_back(-Bh);
      ++numVisible;
    }

    // Require at least 2 views for triangulation
    if (numVisible < 2) continue;

    // Stack F and E matrices
    const int m = numVisible;
    Eigen::MatrixXd F(m * 3, 9 * (HORIZON+1));
    Eigen::MatrixXd E(m * 3, 3);
    for (int i = 0; i < m; ++i) {
      F.block<3, 9 * (HORIZON+1)>(3 * i, 0) = F_blocks[i];
      E.block<3, 3>(3 * i, 0) = E_blocks[i];
    }

    // Compute Schur complement form
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m * 3, m * 3);
    Eigen::Matrix3d EtE = E.transpose() * E;
    if (EtE.determinant() < 1e-10) continue;

    Eigen::MatrixXd P = I - E * EtE.inverse() * E.transpose();
    Eigen::MatrixXd Delta = F.transpose() * P * F;


    omega_horizon_t fixedDelta = Delta;  // need to convert Eigen::MatrixXd to omega_horizon_t -> implicit conversion (safe when sizes match)
    Delta_ells[feature_id] = fixedDelta;
    Fs[feature_id] = F;
    Ps[feature_id] = P;
  }

  return Delta_ells;
}


std::vector<int> FeatureSelector::select_low_rank_update(image_t& subset,
          const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
          const std::map<int, omega_horizon_t>& Delta_ells,
          const std::map<int, omega_horizon_t>& Delta_used_ells,
          const std::map<int, Eigen::MatrixXd>& Fs,
          const std::map<int, Eigen::MatrixXd>& Ps)
  {
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
        int feature_id = Delta.first;
        double p = subset.at(feature_id)[0].second.coeff(fPROB);
        Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    // select the indices of the best features
    for (int i=0; i<kappa; ++i)
    {
        double f_min = 10000000;
        int feature_id_min = -1;

        omega_horizon_t A = Omega + OmegaS; // Omega base for low rank update
        omega_horizon_t A_inv = A.llt().solve(omega_horizon_t::Identity());
        for (const auto& Delta : Delta_ells)
        {
          int feature_id = Delta.first;
          // check if this feature chosen before; if yes: skip
          bool in_blacklist = std::find(blacklist.begin(), blacklist.end(), feature_id) != blacklist.end();
          if (in_blacklist) continue;

          double p = image.at(feature_id)[0].second.coeff(fPROB);

          // low rank update
          const Eigen::MatrixXd& F = Fs.at(feature_id);
          const Eigen::MatrixXd& P = Ps.at(feature_id);
          
          // Eigen::MatrixXd S = Eigen::MatrixXd::Identity(P.rows(), P.rows())/p + F * A_inv * F.transpose() * P;
          // omega_horizon_t information_matrix_tmp_inv = A_inv - A_inv * F.transpose() * P * S.inverse() * F * A_inv;
          const auto F_txP = F.transpose() * P;
          const auto FxA_inv = F * A_inv;
          Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.rows());
          Eigen::MatrixXd S = I/p + FxA_inv * F_txP;
          omega_horizon_t information_matrix_tmp_inv = A_inv - A_inv * F_txP * S.inverse() * FxA_inv;
          
          double f_tmp = information_matrix_tmp_inv.trace();


          // // old (direct) method
          // TicToc tt2{};
          // tt2.tic();
          omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta.second;
          double f_tmp_old = information_matrix_tmp.llt().solve(omega_horizon_t::Identity()).trace();
          // double time_ms_old = tt2.toc();

          // ROS_INFO_STREAM("Kian: new: " << f_tmp << " (" << time_ms_new << " ms)" << " - old: " << f_tmp_old << " (" << time_ms_old << " ms)");
          // ROS_INFO_STREAM("Kian: new: " << f_tmp  << " - old: " << f_tmp_old);



          if(f_tmp < f_min)
          {
              f_min = f_tmp;
              feature_id_min = feature_id;
          }
        }
        // if feature_id_max == -1 there was likely a nan (probably because roundoff error
        // caused det(M) < 0). I guess there just won't be a feature this iter.
        if (feature_id_min > -1) {
          double p = image.at(feature_id_min)[0].second.coeff(fPROB);
          OmegaS += p*Delta_ells.at(feature_id_min);

          // add feature that returns the most information to the subset
          subset[feature_id_min] = image.at(feature_id_min);

          // mark as used
          blacklist.push_back(feature_id_min);
        }

    }

    return blacklist;
  }



std::vector<int> FeatureSelector::select_traceofinv_simple(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
        int feature_id = Delta.first;
        double p = subset.at(feature_id)[0].second.coeff(fPROB);
        Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    // select the indices of the best features
    for (int i=0; i<kappa; ++i)
    {
        double f_min = 10000000;
        int feature_id_min = -1;
        for (const auto& Delta : Delta_ells)
        {
        int feature_id = Delta.first;
        // check if this feature chosen before; if yes: skip
        bool in_blacklist = std::find(blacklist.begin(), blacklist.end(), feature_id) != blacklist.end();
        if (in_blacklist) continue;

        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta.second;

        //// trace of inverse
        // first method - using sum of inverse of eigenvalues
        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 126, 126>> solver(information_matrix_tmp);
        // if (solver.info() != Eigen::Success) {
        //     ROS_INFO_STREAM("KIAN [trace of inverse] " << "there is a problem in solver");
        //     continue;
        // }
        // double f_tmp = solver.eigenvalues().cwiseInverse().sum();
        // second method - using the actual trace of inverse
        // double f_tmp = information_matrix_tmp.inverse().trace();
        // third method - cholesky decomposition
        double f_tmp = information_matrix_tmp.llt().solve(omega_horizon_t::Identity()).trace();

        

        if(f_tmp < f_min)
        {
            f_min = f_tmp;
            feature_id_min = feature_id;
        }
        }
        // if feature_id_max == -1 there was likely a nan (probably because roundoff error
        // caused det(M) < 0). I guess there just won't be a feature this iter.
        if (feature_id_min > -1) {
          double p = image.at(feature_id_min)[0].second.coeff(fPROB);
          OmegaS += p*Delta_ells.at(feature_id_min);

          // add feature that returns the most information to the subset
          subset[feature_id_min] = image.at(feature_id_min);

          // mark as used
          blacklist.push_back(feature_id_min);
        }

    }


    // omega_horizon_t information_matrix = Omega + OmegaS;
    // double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    // return fValue;
    return blacklist;
  }






std::vector<int> FeatureSelector::select_traceofinv_lazy(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
    int feature_id = Delta.first;
    double p = subset.at(feature_id)[0].second.coeff(fPROB);
    Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    // select the indices of the best features
    for (int i=0; i<kappa; ++i) {

      // compute lower bounds in form of <LB, featureId> descending by LB
      std::map<double, int, std::greater<double>> LBs;
      const omega_horizon_t M = Omega + OmegaS;

      // min eigenvalue and min eigen vector
      Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(M);
      double minEigenvalue = es.eigenvalues()(0);
      Eigen::VectorXd minEigenvector = es.eigenvectors().col(0);


      for (const auto& fpair : Delta_ells) {
          int feature_id = fpair.first;

          // if a feature was already selected, do not calc LB. Not including it
          // in the LBs prevents it from being selected again.
          bool in_blacklist = std::find(blacklist.begin(), blacklist.end(),
                                      feature_id) != blacklist.end();
          if (in_blacklist) continue;

          // find probability of this feature being tracked
          double p = image.at(feature_id)[0].second.coeff(fPROB);

          // construct the argument to the logdetUB function
          omega_horizon_t delta = p*Delta_ells.at(feature_id);

          int n = M.rows();
          double lb = n/(minEigenvalue + (delta * minEigenvector).norm());
          LBs[lb] = feature_id;

      }

      // initialize the best cost function value and feature ID to worst case
      double fMin = 100000000.0;
      int lMin = -1;

      // iterating through upperBounds in descending order, check each feature
      for (const auto& fpair : LBs) {
          int feature_id = fpair.second;
          double lb = fpair.first;

          // lazy evaluation: break if lB is greater than the current best cost
          if (lb > fMin) break;

          // convenience: the information matrix corresponding to this feature
          const auto& Delta_ell = Delta_ells.at(feature_id);

          // find probability of this feature being tracked
          double p = image.at(feature_id)[0].second.coeff(fPROB);

          // calculate logdet efficiently
          auto information_matrix_tmp = Omega + OmegaS + p*Delta_ell;
          double fValue = information_matrix_tmp.llt().solve(omega_horizon_t::Identity()).trace();

          // nan check
          if (std::isnan(fValue)) ROS_ERROR_STREAM("trinv returned nan!");

          // store this feature/reward if better than before
          if (fValue < fMin) {
          fMin = fValue;
          lMin = feature_id;
          }
      }

      // if lMin == -1 there was likely a nan (probably because roundoff error
      // caused det(M) < 0). I guess there just won't be a feature this iter.
      if (lMin > -1) {
          // Accumulate combined feature information in subset
          double p = image.at(lMin)[0].second.coeff(fPROB);
          OmegaS += p*Delta_ells.at(lMin);

          // add feature that returns the most information to the subset
          subset[lMin] = image.at(lMin);

          // mark as used
          blacklist.push_back(lMin);
      }
    }


    // omega_horizon_t information_matrix = Omega + OmegaS;
    // double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    // return fValue;
    return blacklist;

}





std::vector<int> FeatureSelector::select_traceofinv_randomized(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
        int feature_id = Delta.first;
        double p = subset.at(feature_id)[0].second.coeff(fPROB);
        Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();


    int N = Delta_ells.size();
    float eps = (exp(-kappa)+exp(-kappa/float(N)))/2.0; // choosing center of the interval!
    // eps = 0.1;
    int r = (float(N)/kappa) * log(1.0/eps);

    if(r == 0)
      ROS_WARN_STREAM("Kian: traceofinv randomized greedy with r = 0! debug: N: " << N << ", kappa: " << kappa << ", eps_low: " << exp(-kappa) << ", eps_high: " << exp(-kappa/float(N)) << ", eps_chosen: " << eps);


    // vector of all feature ids
    std::vector<int> ids;
    for (const auto& element : Delta_ells)
        ids.push_back(element.first);
    // std::stringstream logMessage; logMessage << "ids to choose from: "; for(const auto& id : ids) {logMessage << id << " - ";} ROS_INFO_STREAM(logMessage.str());

    std::vector<int> S; // to be selected feature ids
    for (int i=0; i<kappa; ++i)
    {
        // ROS_INFO_STREAM("round: " << i);
        std::vector<int> U_selected; // U/S
        std::copy_if(ids.begin(), ids.end(), std::back_inserter(U_selected), [&S](int elem) { return std::find(S.begin(), S.end(), elem) == S.end(); });
        // std::stringstream logmessage1; logmessage1 << "    U minus selected (U_selected): "; for(const auto& id : U_selected) {logmessage1 << id << " - ";} ROS_INFO_STREAM(logmessage1.str());
        // selecting r random elements from U_selected
        r = std::min((size_t)r, U_selected.size()); // Ensure r is not greater than the size of U_selected
        // ROS_INFO_STREAM("chunk sizes (r): " << r);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(U_selected.begin(), U_selected.end(), std::default_random_engine(seed)); // Shuffle U_selected
        // std::stringstream logmessage2; logmessage2 << "    shuffled U_selected: "; for(const auto& id : U_selected) {logmessage2 << id << " - ";} ROS_INFO_STREAM(logmessage2.str());
        std::vector<int> V(U_selected.begin(), U_selected.begin() + r);

        // std::stringstream logmessage3; logmessage3 << "    selected chunk: "; for(const auto& id : V) {logmessage3 << id << " - ";} ROS_INFO_STREAM(logmessage3.str());
        
        double f_min = 10000000;
        int feature_id_min = -1;
        for (const auto& feature_id : V)
        {
        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta_ells.at(feature_id);
        double f_tmp = information_matrix_tmp.llt().solve(omega_horizon_t::Identity()).trace();

        if(f_tmp < f_min)
        {
            f_min = f_tmp;
            feature_id_min = feature_id;
        }
        }

        // ROS_INFO_STREAM("    selected feature: " << feature_id_min);
        // if feature_id_max == -1 there was likely a nan (probably because roundoff error
        // caused det(M) < 0). I guess there just won't be a feature this iter.
        if (feature_id_min > -1) {
        double p = image.at(feature_id_min)[0].second.coeff(fPROB);
        OmegaS += p*Delta_ells.at(feature_id_min);
        S.push_back(feature_id_min);
        // add feature that returns the most information to the subset
        subset[feature_id_min] = image.at(feature_id_min);
        }
    }



    // omega_horizon_t information_matrix = Omega + OmegaS;
    // double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    // return fValue;
    return S;
  }





std::vector<int> FeatureSelector::select_logdet_simple(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    // select the indices of the best features
    for (int i=0; i<kappa; ++i)
    {
      double f_max = -10000000;
      int feature_id_min = -1;
      for (const auto& Delta : Delta_ells)
      {
        int feature_id = Delta.first;
        // check if this feature chosen before; if yes: skip
        bool in_blacklist = std::find(blacklist.begin(), blacklist.end(), feature_id) != blacklist.end();
        if (in_blacklist) continue;

        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta.second;

        double f_tmp = Utility::logdet(information_matrix_tmp, true);

        

        if(f_tmp > f_max)
        {
          f_max = f_tmp;
          feature_id_min = feature_id;
        }
      }
      // if feature_id_max == -1 there was likely a nan (probably because roundoff error
      // caused det(M) < 0). I guess there just won't be a feature this iter.
      if (feature_id_min > -1) {
        double p = image.at(feature_id_min)[0].second.coeff(fPROB);
        OmegaS += p*Delta_ells.at(feature_id_min);

        // add feature that returns the most information to the subset
        subset[feature_id_min] = image.at(feature_id_min);

        // mark as used
        blacklist.push_back(feature_id_min);
      }

    }


    // omega_horizon_t information_matrix = Omega + OmegaS;
    // double fValue =  Utility::logdet(information_matrix, true);
    // return fValue;
    return blacklist;
}



std::vector<int> FeatureSelector::select_logdet_lazy(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
  // Combine motion information with information from features that are already
  // being used in the VINS-Mono optimization backend
  omega_horizon_t Omega = Omega_kkH;
  for (const auto& Delta : Delta_used_ells) {
    Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
  }

  // blacklist of already selected features (by id)
  std::vector<int> blacklist;
  blacklist.reserve(kappa);

  // combined information of subset
  omega_horizon_t OmegaS = omega_horizon_t::Zero();

  // select the indices of the best features
  for (int i=0; i<kappa; ++i) {

    // compute upper bounds in form of <UB, featureId> descending by UB
    auto upperBounds = sortedlogDetUB(Omega, OmegaS, Delta_ells, blacklist, image);

    // initialize the best cost function value and feature ID to worst case
    double fMax = -1.0;
    int lMax = -1;

    // iterating through upperBounds in descending order, check each feature
    for (const auto& fpair : upperBounds) {
      int feature_id = fpair.second;
      double ub = fpair.first;

      // lazy evaluation: break if UB is less than the current best cost
      if (ub < fMax) break;
 
      // convenience: the information matrix corresponding to this feature
      const auto& Delta_ell = Delta_ells.at(feature_id);

      // find probability of this feature being tracked
      double p = image.at(feature_id)[0].second.coeff(fPROB);

      // calculate logdet efficiently
      double fValue = Utility::logdet(Omega + OmegaS + p*Delta_ell, true);

      // nan check
      if (std::isnan(fValue)) ROS_ERROR_STREAM("logdet returned nan!");

      // store this feature/reward if better than before
      if (fValue > fMax) {
        fMax = fValue;
        lMax = feature_id;
      }
    }

    // if lMax == -1 there was likely a nan (probably because roundoff error
    // caused det(M) < 0). I guess there just won't be a feature this iter.
    if (lMax > -1) {
      // Accumulate combined feature information in subset
      double p = image.at(lMax)[0].second.coeff(fPROB);
      OmegaS += p*Delta_ells.at(lMax);

      // add feature that returns the most information to the subset
      subset[lMax] = image.at(lMax);

      // mark as used
      blacklist.push_back(lMax);
    }
  }

  // omega_horizon_t information_matrix = Omega + OmegaS;
  // double fValue = Utility::logdet(Omega + OmegaS, true);
  // return fValue;
  return blacklist;
}



std::vector<int> FeatureSelector::select_logdet_randomized(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    int N = Delta_ells.size();
    float eps = (exp(-kappa)+exp(-kappa/float(N)))/2.0; // choosing center of the interval!
    // eps = 0.1;
    int r = (float(N)/kappa) * log(1.0/eps);

    if(r == 0)
      ROS_WARN_STREAM("Kian: logdet randomized greedy with r = 0! debug: N: " << N << ", kappa: " << kappa << ", eps_low: " << exp(-kappa) << ", eps_high: " << exp(-kappa/float(N)) << ", eps_chosen: " << eps);


    // vector of all feature ids
    std::vector<int> ids;
    for (const auto& element : Delta_ells)
      ids.push_back(element.first);
    // std::stringstream logMessage; logMessage << "ids to choose from: "; for(const auto& id : ids) {logMessage << id << " - ";} ROS_INFO_STREAM(logMessage.str());

    std::vector<int> S; // to be selected feature ids
    for (int i=0; i<kappa; ++i)
    {
      // ROS_INFO_STREAM("round: " << i);
      std::vector<int> U_selected; // U/S
      std::copy_if(ids.begin(), ids.end(), std::back_inserter(U_selected), [&S](int elem) { return std::find(S.begin(), S.end(), elem) == S.end(); });
      // std::stringstream logmessage1; logmessage1 << "    U minus selected (U_selected): "; for(const auto& id : U_selected) {logmessage1 << id << " - ";} ROS_INFO_STREAM(logmessage1.str());
      // selecting r random elements from U_selected
      r = std::min((size_t)r, U_selected.size()); // Ensure r is not greater than the size of U_selected
      // ROS_INFO_STREAM("chunk sizes (r): " << r);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(U_selected.begin(), U_selected.end(), std::default_random_engine(seed)); // Shuffle U_selected
      // std::stringstream logmessage2; logmessage2 << "    shuffled U_selected: "; for(const auto& id : U_selected) {logmessage2 << id << " - ";} ROS_INFO_STREAM(logmessage2.str());
      std::vector<int> V(U_selected.begin(), U_selected.begin() + r);

      // std::stringstream logmessage3; logmessage3 << "    selected chunk: "; for(const auto& id : V) {logmessage3 << id << " - ";} ROS_INFO_STREAM(logmessage3.str());
      
      double f_max = -10000000;
      int feature_id_min = -1;
      for (const auto& feature_id : V)
      {
        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta_ells.at(feature_id);
        double f_tmp = Utility::logdet(information_matrix_tmp, true);

        if(f_tmp > f_max)
        {
          f_max = f_tmp;
          feature_id_min = feature_id;
        }
      }

      // ROS_INFO_STREAM("    selected feature: " << feature_id_min);
      // if feature_id_max == -1 there was likely a nan (probably because roundoff error
      // caused det(M) < 0). I guess there just won't be a feature this iter.
      if (feature_id_min > -1) {
        double p = image.at(feature_id_min)[0].second.coeff(fPROB);
        OmegaS += p*Delta_ells.at(feature_id_min);
        S.push_back(feature_id_min);
        // add feature that returns the most information to the subset
        subset[feature_id_min] = image.at(feature_id_min);
      }
    }
    // omega_horizon_t information_matrix = Omega + OmegaS;
    // double fValue = Utility::logdet(information_matrix, true);
    // return fValue;
    return S;
}








std::vector<int> FeatureSelector::select_lambdamin_simple(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    // select the indices of the best features
    for (int i=0; i<kappa; ++i)
    {
      double f_max = -10000000;
      int feature_id_min = -1;
      for (const auto& Delta : Delta_ells)
      {
        int feature_id = Delta.first;
        // check if this feature chosen before; if yes: skip
        bool in_blacklist = std::find(blacklist.begin(), blacklist.end(), feature_id) != blacklist.end();
        if (in_blacklist) continue;

        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta.second;

        Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix_tmp);
        double minEigenvalue = es.eigenvalues()(0);
        double f_tmp = minEigenvalue;

        

        if(f_tmp > f_max)
        {
          f_max = f_tmp;
          feature_id_min = feature_id;
        }
      }
      // if feature_id_max == -1 there was likely a nan (probably because roundoff error
      // caused det(M) < 0). I guess there just won't be a feature this iter.
      if (feature_id_min > -1) {
        double p = image.at(feature_id_min)[0].second.coeff(fPROB);
        OmegaS += p*Delta_ells.at(feature_id_min);

        // add feature that returns the most information to the subset
        subset[feature_id_min] = image.at(feature_id_min);

        // mark as used
        blacklist.push_back(feature_id_min);
      }

    }


    // omega_horizon_t information_matrix = Omega + OmegaS;
    // Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
    // double minEigenvalue = es.eigenvalues()(0);
    // double fValue =  minEigenvalue;
    // return fValue;
    return blacklist;
}



std::vector<int> FeatureSelector::select_lambdamin_lazy(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{  
  // Combine motion information with information from features that are already
  // being used in the VINS-Mono optimization backend
  omega_horizon_t Omega = Omega_kkH;
  for (const auto& Delta : Delta_used_ells) {
    Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
  }

  // blacklist of already selected features (by id)
  std::vector<int> blacklist;
  blacklist.reserve(kappa);

  // combined information of subset
  omega_horizon_t OmegaS = omega_horizon_t::Zero();

  // select the indices of the best features
  for (int i=0; i<kappa; ++i) {

    // compute upper bounds in form of <UB, featureId> descending by UB
    auto upperBounds = sortedlambdaminUB(Omega, OmegaS, Delta_ells, blacklist, image);

    // initialize the best cost function value and feature ID to worst case
    double fMax = -1.0;
    int lMax = -1;


    omega_horizon_t omega_omega_s = Omega + OmegaS;
    // iterating through upperBounds in descending order, check each feature
    for (const auto& fpair : upperBounds) {
      int feature_id = fpair.second;
      double ub = fpair.first;

      // lazy evaluation: break if UB is less than the current best cost
      if (ub < fMax) break;
 
      // convenience: the information matrix corresponding to this feature
      const auto& Delta_ell = Delta_ells.at(feature_id);

      // find probability of this feature being tracked
      double p = image.at(feature_id)[0].second.coeff(fPROB);

      // calculate lambdamin efficiently
      Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(omega_omega_s + p*Delta_ell);
      double minEigenvalue = es.eigenvalues()(0);
      double fValue = minEigenvalue;

      // nan check
      if (std::isnan(fValue)) ROS_ERROR_STREAM("lambdamin returned nan!");

      // store this feature/reward if better than before
      if (fValue > fMax) {
        fMax = fValue;
        lMax = feature_id;
      }
    }

    // if lMax == -1 there was likely a nan (probably because roundoff error
    // caused det(M) < 0). I guess there just won't be a feature this iter.
    if (lMax > -1) {
      // Accumulate combined feature information in subset
      double p = image.at(lMax)[0].second.coeff(fPROB);
      OmegaS += p*Delta_ells.at(lMax);

      // add feature that returns the most information to the subset
      subset[lMax] = image.at(lMax);

      // mark as used
      blacklist.push_back(lMax);
    }
  }

  // omega_horizon_t information_matrix = Omega + OmegaS;
  // Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
  // double minEigenvalue = es.eigenvalues()(0);
  // return minEigenvalue;

  return blacklist;
}




std::vector<int> FeatureSelector::select_lambdamin_randomized(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // combined information of subset
    omega_horizon_t OmegaS = omega_horizon_t::Zero();

    int N = Delta_ells.size();
    float eps = (exp(-kappa)+exp(-kappa/float(N)))/2.0; // choosing center of the interval!
    // eps = 0.1;
    int r = (float(N)/kappa) * log(1.0/eps);

    if(r == 0)
      ROS_WARN_STREAM("Kian: lambdamin randomized greedy with r = 0! debug: N: " << N << ", kappa: " << kappa << ", eps_low: " << exp(-kappa) << ", eps_high: " << exp(-kappa/float(N)) << ", eps_chosen: " << eps);



    // vector of all feature ids
    std::vector<int> ids;
    for (const auto& element : Delta_ells)
      ids.push_back(element.first);
    // std::stringstream logMessage; logMessage << "ids to choose from: "; for(const auto& id : ids) {logMessage << id << " - ";} ROS_INFO_STREAM(logMessage.str());

    std::vector<int> S; // to be selected feature ids
    for (int i=0; i<kappa; ++i)
    {
      // ROS_INFO_STREAM("round: " << i);
      std::vector<int> U_selected; // U/S
      std::copy_if(ids.begin(), ids.end(), std::back_inserter(U_selected), [&S](int elem) { return std::find(S.begin(), S.end(), elem) == S.end(); });
      // std::stringstream logmessage1; logmessage1 << "    U minus selected (U_selected): "; for(const auto& id : U_selected) {logmessage1 << id << " - ";} ROS_INFO_STREAM(logmessage1.str());
      // selecting r random elements from U_selected
      r = std::min((size_t)r, U_selected.size()); // Ensure r is not greater than the size of U_selected
      // ROS_INFO_STREAM("chunk sizes (r): " << r);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(U_selected.begin(), U_selected.end(), std::default_random_engine(seed)); // Shuffle U_selected
      // std::stringstream logmessage2; logmessage2 << "    shuffled U_selected: "; for(const auto& id : U_selected) {logmessage2 << id << " - ";} ROS_INFO_STREAM(logmessage2.str());
      std::vector<int> V(U_selected.begin(), U_selected.begin() + r);

      // std::stringstream logmessage3; logmessage3 << "    selected chunk: "; for(const auto& id : V) {logmessage3 << id << " - ";} ROS_INFO_STREAM(logmessage3.str());
      
      double f_max = -10000000;
      int feature_id_min = -1;
      for (const auto& feature_id : V)
      {
        double p = image.at(feature_id)[0].second.coeff(fPROB);

        omega_horizon_t information_matrix_tmp = Omega + OmegaS + p*Delta_ells.at(feature_id);
        Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix_tmp);
        double minEigenvalue = es.eigenvalues()(0);
        double f_tmp = minEigenvalue;

        if(f_tmp > f_max)
        {
          f_max = f_tmp;
          feature_id_min = feature_id;
        }
      }

      // ROS_INFO_STREAM("    selected feature: " << feature_id_min);
      // if feature_id_max == -1 there was likely a nan (probably because roundoff error
      // caused det(M) < 0). I guess there just won't be a feature this iter.
      if (feature_id_min > -1) {
        double p = image.at(feature_id_min)[0].second.coeff(fPROB);
        OmegaS += p*Delta_ells.at(feature_id_min);
        S.push_back(feature_id_min);
        // add feature that returns the most information to the subset
        subset[feature_id_min] = image.at(feature_id_min);
      }
    }

    // omega_horizon_t information_matrix = Omega + OmegaS;
    // Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
    // double minEigenvalue = es.eigenvalues()(0);
    // double fValue = minEigenvalue;
    // return fValue;
    return S;
}









std::map<double, int, std::greater<double>> FeatureSelector::sortedlambdaminUB(
  const omega_horizon_t& Omega, const omega_horizon_t& OmegaS,
  const std::map<int, omega_horizon_t>& Delta_ells,
  const std::vector<int>& blacklist, const image_t& image)
{
  // returns a descending sorted map with upper bound as the first key,
  // and feature id as the value for all features in image
  std::map<double, int, std::greater<double>> UBs;


  const omega_horizon_t M = Omega + OmegaS;
  // min eigenvalue and min eigen vector
  Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(M);
  double minEigenvalue = es.eigenvalues()(0);
  Eigen::VectorXd minEigenvector = es.eigenvectors().col(0);

  // Find the upper bound of adding each Delta_ell to M independently
  for (const auto& fpair : Delta_ells) {
    int feature_id = fpair.first;

    // if a feature was already selected, do not calc UB. Not including it
    // in the UBs prevents it from being selected again.
    bool in_blacklist = std::find(blacklist.begin(), blacklist.end(),
                                  feature_id) != blacklist.end();
    if (in_blacklist) continue;

    // find probability of this feature being tracked
    double p = image.at(feature_id)[0].second.coeff(fPROB);

    // construct the argument to the logdetUB function
    omega_horizon_t Delta = p*Delta_ells.at(feature_id);

    // calculate upper bound (eq 29)
    double ub = minEigenvalue + (Delta * minEigenvector).norm();

    // store in map for automatic sorting (desc) and easy lookup
    UBs[ub] = feature_id;
  }

  return UBs;
}







std::vector<int> FeatureSelector::select_linearized(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }
    auto OmegaInverse = Omega; // Omega.llt().solve(omega_horizon_t::Identity());
    // auto gradient_rho = OmegaInverse * OmegaInverse; // this is slow
    // faster way using cholseky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(OmegaInverse);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::MatrixXd gradient_rho = L * (L.transpose() * L) * L.transpose();

    // compute linearized score in form of <score, featureId> descending by score
    std::map<double, int, std::greater<double>> scores;
    for (const auto& fpair : Delta_ells) {
      int feature_id = fpair.first;

      // find probability of this feature being tracked
      double p = image.at(feature_id)[0].second.coeff(fPROB);

      // construct the argument to the logdetUB function
      omega_horizon_t delta = p*Delta_ells.at(feature_id);

      double score = (gradient_rho * delta).trace();
      scores[score] = feature_id;
    }


    // Iterate through the map and collect the first kappa IDs
    std::vector<int> ids;
    int kappa_ = kappa;
    for (auto it = scores.begin(); kappa_ > 0 && it != scores.end(); ++it, --kappa_) {
        ids.push_back(it->second);
        subset[it->second] = image.at(it->second);
    }


    // for(auto id : ids)
    // {
    //   const auto& Delta_ell = Delta_ells.at(id);
    //   double p = image.at(id)[0].second.coeff(fPROB);
    //   Omega += p*Delta_ell;
    // }
    // return Omega.llt().solve(omega_horizon_t::Identity()).trace();
    return ids;
  }


  




std::vector<int> FeatureSelector::select_quality(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    // Combine motion information with information from features that are already
    // being used in the VINS-Mono optimization backend
    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    // blacklist of already selected features (by id)
    std::vector<int> blacklist;
    blacklist.reserve(kappa);

    std::vector<std::pair<int, double>> id_score_pairs;
    id_score_pairs.clear();
    for (const auto& Delta : Delta_ells)
    {
      int feature_id = Delta.first;
      double p = image.at(feature_id)[0].second.coeff(fPROB);

      id_score_pairs.push_back({feature_id, p});
    }

    std::sort(id_score_pairs.begin(), id_score_pairs.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b){return a.second > b.second;});
    
    std::vector<int> top_ids;
    top_ids.clear();
    for(int i{}; i < kappa && i < (int)id_score_pairs.size(); i++)
      top_ids.push_back(id_score_pairs[i].first);
    
    for(const auto& id : top_ids)
      subset[id] = image.at(id);

    return top_ids;
  }


std::vector<int> FeatureSelector::select_grid(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    // Grid configuration
    const int w_num = 15; // Number of horizontal divisions
    const int h_num = 12; // Number of vertical divisions

    const int image_width = m_camera_->imageWidth();
    const int image_height = m_camera_->imageHeight();

    const double cell_width = static_cast<double>(image_width) / w_num;
    const double cell_height = static_cast<double>(image_height) / h_num;

    // Define a hashable GridKey struct for unordered_map
    struct GridKey {
        int row;
        int col;

        bool operator==(const GridKey& other) const {
            return row == other.row && col == other.col;
        }
    };

    // Hash function for GridKey (C++14 style)
    struct GridKeyHash {
        std::size_t operator()(const GridKey& k) const {
            return std::hash<int>()(k.row) ^ (std::hash<int>()(k.col) << 1);
        }
    };

    // Map from grid cell to feature IDs
    std::unordered_map<GridKey, std::vector<int>, GridKeyHash> grid_cells;
    // Assign features to grid cells
    for (std::map<int, omega_horizon_t>::const_iterator it = Delta_ells.begin(); it != Delta_ells.end(); ++it)
    {
        int feature_id = it->first;
        if (image.count(feature_id) == 0 || image.at(feature_id).empty())
            continue;

        const Eigen::Matrix<double, 8, 1>& feat = image.at(feature_id)[0].second;
        double u = feat[3];
        double v = feat[4];

        int col = std::min(static_cast<int>(u / cell_width), w_num - 1);
        int row = std::min(static_cast<int>(v / cell_height), h_num - 1);

        GridKey key = {row, col};
        grid_cells[key].push_back(feature_id);
    }

    std::vector<int> selected_features;
    std::mt19937 rng(std::random_device{}());

    // Select one random feature per cell
    for (std::unordered_map<GridKey, std::vector<int>, GridKeyHash>::iterator it = grid_cells.begin(); it != grid_cells.end(); ++it)
    {
        std::vector<int>& features = it->second;
        if (!features.empty()) {
            std::shuffle(features.begin(), features.end(), rng);
            selected_features.push_back(features[0]);
        }
    }

    // If not enough features, add more randomly
    if (selected_features.size() < static_cast<size_t>(kappa)) {
        std::set<int> selected_set(selected_features.begin(), selected_features.end());
        std::vector<int> remaining;

        for (std::map<int, omega_horizon_t>::const_iterator it = Delta_ells.begin(); it != Delta_ells.end(); ++it) {
            int fid = it->first;
            if (selected_set.find(fid) == selected_set.end()) {
                remaining.push_back(fid);
            }
        }

        std::shuffle(remaining.begin(), remaining.end(), rng);
        for (size_t i = 0; i < remaining.size() && selected_features.size() < static_cast<size_t>(kappa); ++i) {
            selected_features.push_back(remaining[i]);
        }
    }

    // If too many, truncate
    if (selected_features.size() > static_cast<size_t>(kappa)) {
        std::shuffle(selected_features.begin(), selected_features.end(), rng);
        selected_features.resize(kappa);
    }

    for(const auto& id : selected_features)
      subset[id] = image.at(id);


    return selected_features;
  }




std::vector<int> FeatureSelector::select_actualrandom(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    std::vector<std::pair<int, omega_horizon_t>> pairs;
    // Copy the pairs from the map to the vector
    for (const auto& pair : Delta_ells)
        pairs.push_back(pair);
    // Shuffle the vector
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pairs.begin(), pairs.end(), g);
    // Resize the vector to contain only k elements if it's larger
    if (pairs.size() > kappa) {
        pairs.resize(kappa);
    }

    std::vector<int> blacklist;
    blacklist.reserve(kappa);
    for(const auto& pair : pairs)
    {
    int id = pair.first;
    double p = image.at(id)[0].second.coeff(fPROB);

    subset[id] = image.at(id);
    blacklist.push_back(id);
    }

    return blacklist;
}