#include "feature_selector.h"



std::pair<double, double> calc_mean_std(std::vector<double> vec)
{
  // Calculating mean
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  double mean = sum / vec.size();

  // Calculating standard deviation
  double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0,
                                      std::plus<double>(), [&](double a, double b) {
                                          return (a - mean) * (a - mean);
                                      });
  double stddev = std::sqrt(sq_sum / vec.size());

  std::pair<double, double> res(mean, stddev);
  return res;
}


void FeatureSelector::time_and_metric_analysis(image_t& subset,
          const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
          const std::map<int, omega_horizon_t>& Delta_ells,
          const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    if(Delta_ells.size() < 135) return;
    ROS_INFO_STREAM(">>************** [feature_selector] KIAN");

    int runs_for_randoms = 50;
    std::vector<int> kappas{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 139};
    ROS_INFO_STREAM(" - new feature number " << Delta_ells.size());
    ROS_INFO_STREAM(" - used feature number " << Delta_used_ells.size());

    // trace of inverse metric - random
    ROS_INFO_STREAM(" ---- trace of inverse metric - random");
    for(const auto& kappa_ : kappas)
    {
      std::vector<double> fs;
      std::vector<double> elapsed_times;
      for(int i{}; i < runs_for_randoms; i++)
      {
        std::pair<float, float> res = select_actualrandom_analysis(subset, image, kappa_, Omega_kkH, Delta_ells, Delta_used_ells);
        fs.push_back(res.first);
        elapsed_times.push_back(res.second);
      }
      auto fs_mean_std = calc_mean_std(fs);
      auto ts_mean_std = calc_mean_std(elapsed_times);
      ROS_INFO_STREAM("kappa: " << kappa_ << " - f_logdet: " << fs_mean_std.first << " (std: " << fs_mean_std.second << ")" << " - elapsed(ms): " << ts_mean_std.first << " (std: " << ts_mean_std.second << ")" << " (AVG FOR " << runs_for_randoms << " runs)");
    }

    // trace of inverse metric - simple greedy
    ROS_INFO_STREAM(" ---- trace of inverse metric - simple greedy");
    for(const auto& kappa_ : kappas)
    {
      std::vector<double> fs;
      std::vector<double> elapsed_times;
      for(int i{}; i < runs_for_randoms; i++)
      {
        std::pair<float, float> res = select_traceofinv_simple_analysis(subset, image, kappa_, Omega_kkH, Delta_ells, Delta_used_ells);
        fs.push_back(res.first);
        elapsed_times.push_back(res.second);
      }
      auto fs_mean_std = calc_mean_std(fs);
      auto ts_mean_std = calc_mean_std(elapsed_times);
      ROS_INFO_STREAM("kappa: " << kappa_ << " - f_logdet: " << fs_mean_std.first << " (std: " << fs_mean_std.second << ")" << " - elapsed(s): " << ts_mean_std.first << " (std: " << ts_mean_std.second << ")" << " (AVG FOR " << runs_for_randoms << " runs)");
    }

    // trace of inverse metric - lazy greedy
    ROS_INFO_STREAM(" ---- trace of inverse metric - lazy greedy");
    for(const auto& kappa_ : kappas)
    {
      std::vector<double> fs;
      std::vector<double> elapsed_times;
      for(int i{}; i < runs_for_randoms; i++)
      {
        std::pair<float, float> res = select_traceofinv_lazy_analysis(subset, image, kappa_, Omega_kkH, Delta_ells, Delta_used_ells);
        fs.push_back(res.first);
        elapsed_times.push_back(res.second);
      }
      auto fs_mean_std = calc_mean_std(fs);
      auto ts_mean_std = calc_mean_std(elapsed_times);
      ROS_INFO_STREAM("kappa: " << kappa_ << " - f_logdet: " << fs_mean_std.first << " (std: " << fs_mean_std.second << ")" << " - elapsed(s): " << ts_mean_std.first << " (std: " << ts_mean_std.second << ")" << " (AVG FOR " << runs_for_randoms << " runs)");
    }

    // trace of inverse metric - randomized greedy
    ROS_INFO_STREAM(" ---- trace of inverse metric - randomized greedy");
    for(const auto& kappa_ : kappas)
    {
      std::vector<double> fs;
      std::vector<double> elapsed_times;
      for(int i{}; i < runs_for_randoms; i++)
      {
        std::pair<float, float> res = select_traceofinv_randomized_analysis(subset, image, kappa_, Omega_kkH, Delta_ells, Delta_used_ells);
        fs.push_back(res.first);
        elapsed_times.push_back(res.second);
      }
      auto fs_mean_std = calc_mean_std(fs);
      auto ts_mean_std = calc_mean_std(elapsed_times);
      ROS_INFO_STREAM("kappa: " << kappa_ << " - f_logdet: " << fs_mean_std.first << " (std: " << fs_mean_std.second << ")" << " - elapsed(s): " << ts_mean_std.first << " (std: " << ts_mean_std.second << ")" << " (AVG FOR " << runs_for_randoms << " runs)");
    }

    // trace of inverse metric - linearized greedy
    ROS_INFO_STREAM(" ---- trace of inverse metric - linearized");
    for(const auto& kappa_ : kappas)
    {
      std::vector<double> fs;
      std::vector<double> elapsed_times;
      for(int i{}; i < runs_for_randoms; i++)
      {
        std::pair<float, float> res = select_linearized_analysis(subset, image, kappa_, Omega_kkH, Delta_ells, Delta_used_ells);
        fs.push_back(res.first);
        elapsed_times.push_back(res.second);
      }
      auto fs_mean_std = calc_mean_std(fs);
      auto ts_mean_std = calc_mean_std(elapsed_times);
      ROS_INFO_STREAM("kappa: " << kappa_ << " - f_logdet: " << fs_mean_std.first << " (std: " << fs_mean_std.second << ")" << " - elapsed(s): " << ts_mean_std.first << " (std: " << ts_mean_std.second << ")" << " (AVG FOR " << runs_for_randoms << " runs)");
    }


    ROS_INFO_STREAM("<<************** [feature_selector] KIAN");
  }





std::pair<float, float> FeatureSelector::select_traceofinv_simple_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    TicToc timer_ms{};
    timer_ms.tic();
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


    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    return std::make_pair(fValue, time_ms);
    // return blacklist;
  }






std::pair<float, float> FeatureSelector::select_traceofinv_lazy_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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
      double maxEigenvalue = es.eigenvalues()(es.eigenvalues().size()-1);
      Eigen::VectorXd maxEigenvector = es.eigenvectors().col(es.eigenvalues().size()-1);


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
          double lb = 1/(maxEigenvalue + (delta * maxEigenvector).norm());
          LBs[lb] = feature_id;

          // double actual = (M+delta).llt().solve(omega_horizon_t::Identity()).trace();
          // ROS_INFO_STREAM("id: " << feature_id
          //             << ", actual traceofinv: " << actual
          //             << ", lowerbound: " << lb
          //             << ", norm of minEigenvector: " << minEigenvector.norm()
          //             << ", p: " << p
          //             << ", denominator: " << (minEigenvalue + (delta * minEigenvector).norm())
          //             << ", n: " << n
          //               );

      }

      // initialize the best cost function value and feature ID to worst case
      double fMin = 100000000.0;
      int lMin = -1;

      // iterating through upperBounds in descending order, check each feature
      for (const auto& fpair : LBs) {
          int feature_id = fpair.second;
          double lb = fpair.first;

          // lazy evaluation: break if lB is greater than the current best cost
          if (lb >= fMin) break;

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


    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    return std::make_pair(fValue, time_ms);
    // return blacklist;

}





std::pair<float, float> FeatureSelector::select_traceofinv_randomized_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    TicToc timer_ms{};
    timer_ms.tic();
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



    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    return std::make_pair(fValue, time_ms);
    // return S;
  }





std::pair<float, float> FeatureSelector::select_logdet_simple_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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


    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    double fValue =  Utility::logdet(information_matrix, true);
    return std::make_pair(fValue, time_ms);
    // return blacklist;
}



std::pair<float, float> FeatureSelector::select_logdet_lazy_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
  TicToc timer_ms{};
  timer_ms.tic();
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

  float time_ms = timer_ms.toc();
  omega_horizon_t information_matrix = Omega + OmegaS;
  double fValue = Utility::logdet(Omega + OmegaS, true);
  return std::make_pair(fValue, time_ms);
  // return blacklist;
}



std::pair<float, float> FeatureSelector::select_logdet_randomized_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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

    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    double fValue = Utility::logdet(information_matrix, true);
    return std::make_pair(fValue, time_ms);
    // return S;
}








std::pair<float, float> FeatureSelector::select_lambdamin_simple_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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


    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
    double minEigenvalue = es.eigenvalues()(0);
    double fValue =  minEigenvalue;
    return std::make_pair(fValue, time_ms);
    // return blacklist;
}



std::pair<float, float> FeatureSelector::select_lambdamin_lazy_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{  
  TicToc timer_ms{};
  timer_ms.tic();
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

  float time_ms = timer_ms.toc();
  omega_horizon_t information_matrix = Omega + OmegaS;
  Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
  double minEigenvalue = es.eigenvalues()(0);
  return std::make_pair(minEigenvalue, time_ms);

  // return blacklist;
}




std::pair<float, float> FeatureSelector::select_lambdamin_randomized_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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

    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega + OmegaS;
    Eigen::SelfAdjointEigenSolver<omega_horizon_t> es(information_matrix);
    double minEigenvalue = es.eigenvalues()(0);
    double fValue = minEigenvalue;
    return std::make_pair(fValue, time_ms);
    // return S;
}









std::map<double, int, std::greater<double>> FeatureSelector::sortedlambdaminUB_analysis(
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







std::pair<float, float> FeatureSelector::select_linearized_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
  {
    TicToc timer_ms{};
    timer_ms.tic();
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


    for(auto id : ids)
    {
      const auto& Delta_ell = Delta_ells.at(id);
      double p = image.at(id)[0].second.coeff(fPROB);
      Omega += p*Delta_ell;
    }

    float time_ms = timer_ms.toc();
    double fValue = Omega.llt().solve(omega_horizon_t::Identity()).trace();
    return std::make_pair(fValue, time_ms);
    // return ids;
  }


  





std::pair<float, float> FeatureSelector::select_actualrandom_analysis(image_t& subset,
            const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
            const std::map<int, omega_horizon_t>& Delta_ells,
            const std::map<int, omega_horizon_t>& Delta_used_ells)
{
    TicToc timer_ms{};
    timer_ms.tic();
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

    omega_horizon_t Omega = Omega_kkH;
    for (const auto& Delta : Delta_used_ells) {
      int feature_id = Delta.first;
      double p = subset.at(feature_id)[0].second.coeff(fPROB);
      Omega += Delta.second; // KIAN: shouldn't we put p*Delta.second??
    }

    std::vector<int> blacklist;
    blacklist.reserve(kappa);
    for(const auto& pair : pairs)
    {
    int id = pair.first;
    double p = image.at(id)[0].second.coeff(fPROB);
    Omega += p*Delta_ells.at(id);

    subset[id] = image.at(id);
    blacklist.push_back(id);
    }

    float time_ms = timer_ms.toc();
    omega_horizon_t information_matrix = Omega;
    double fValue = information_matrix.llt().solve(omega_horizon_t::Identity()).trace();
    return std::make_pair(fValue, time_ms);
    // return blacklist;
}