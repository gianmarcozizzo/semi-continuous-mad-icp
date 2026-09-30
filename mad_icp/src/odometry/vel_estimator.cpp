// Copyright 2024 R(obots) V(ision) and P(erception) group
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "vel_estimator.h"
#include <tools/constants.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// ─── SE3 utilities ────────────────────────────────────────────────────────────
// Convenzione [t; ω]: logMapSE3(T) = [T.translation(); logMapSO3(T.linear())]
// Queste funzioni sono standalone (non richiedono gsolver né GTSAM) e sono
// usate da computeQdFromKittiGT (sempre compilata).
static inline Eigen::Matrix<double, 6, 1> logMapSE3(const Eigen::Isometry3d& T) {
  Eigen::Matrix<double, 6, 1> x;
  x.head<3>() = T.translation();
  x.tail<3>() = logMapSO3(T.linear());
  return x;
}

static inline Eigen::Isometry3d expMapSE3(const Eigen::Matrix<double, 6, 1>& x) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation()     = x.head<3>();
  T.linear()          = expMapSO3(x.tail<3>());
  return T;
}

// ─── Costruttore ──────────────────────────────────────────────────────────────
// solver_type: "classic" | "gsolver" | "gtsam"
// gt_path: se non vuoto, calibra gp_cov_ dalla GT KITTI (una sola volta)
VelEstimator::VelEstimator(double sensor_hz,
                           const std::string& solver_type,
                           const std::string& gt_path,
                           double qc_default,
                           double prior_cov,
                           int gn_iterations) {
  X_.setZero();
  ts_                 = 1. / sensor_hz;
  solver_type_        = solver_type;
  qc_default_         = qc_default;
  prior_cov_          = prior_cov;
  gn_iterations_      = gn_iterations;

  // Fallback Qd: struttura WNOA isotropa con Qc = qc_default * I₆
  // (preserva la gerarchia dt³/3 : dt²/2 : dt rispetto al fallback precedente 1e-2·I₁₂)
  // La calibrazione reale viene sovrascritta da computeQdFromKittiGT() se gt_path è fornito.
  //constexpr double qc_default = 0.5;
  const Eigen::Matrix<double, 6, 6> Qc_default =
      Eigen::Matrix<double, 6, 6>::Identity() * qc_default;
  gp_cov_.setZero();
  gp_cov_.block<6, 6>(0, 0) = std::pow(ts_, 3) / 3.0 * Qc_default;
  gp_cov_.block<6, 6>(0, 6) = std::pow(ts_, 2) / 2.0 * Qc_default;
  gp_cov_.block<6, 6>(6, 0) = std::pow(ts_, 2) / 2.0 * Qc_default;
  gp_cov_.block<6, 6>(6, 6) = ts_ * Qc_default;

  if (solver_type_ != "classic" && !gt_path.empty()) {
    gp_cov_ = computeQdFromKittiGT(gt_path);
  } else if (solver_type_ != "classic") {
    std::cout << "[VelEstimator] Qc_diag usata (fallback, cap=" << qc_default_ << "): ["
              << Qc_default.diagonal().transpose() << "]\n"
              << "[VelEstimator] gp_cov_ (fallback WNOA):\n" << gp_cov_ << "\n";
  }
}

void VelEstimator::init(const Vector6d& velocity) {
  X_ = velocity;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeQdFromKittiGT — standalone, non richiede gsolver né GTSAM
//
//   1. Legge le pose dal file GT KITTI (12 valori per riga: [R|t])
//   2. v_i   = logMapSE3(T_i⁻¹·T_{i+1}) / ts_
//   3. a_i   = (Adj(R_{i,i+1})·v_{i+1} − v_i) / ts_
//   4. qc    = var(a)
//   5. Qd    = [[dt³/3·Qc, dt²/2·Qc], [dt²/2·Qc, dt·Qc]]   (formula WNOA)
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix<double, 12, 12> VelEstimator::computeQdFromKittiGT(const std::string& path) {
  std::cout << "[VelEstimator] Caricamento GT KITTI da: " << path << std::endl;

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[VelEstimator] ATTENZIONE: impossibile aprire GT '" << path
              << "'. Rimane il fallback WNOA (qc_default=" << qc_default_ << ").\n";
    return gp_cov_;
  }

  std::vector<Eigen::Isometry3d> poses;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    double r00, r01, r02, tx,
           r10, r11, r12, ty,
           r20, r21, r22, tz;
    if (!(iss >> r00 >> r01 >> r02 >> tx
              >> r10 >> r11 >> r12 >> ty
              >> r20 >> r21 >> r22 >> tz)) continue;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() << r00, r01, r02,
                  r10, r11, r12,
                  r20, r21, r22;
    T.translation() << tx, ty, tz;
    poses.push_back(T);
  }

  if (poses.size() < 3) {
    std::cerr << "[VelEstimator] GT ha meno di 3 pose. Rimane il fallback WNOA (qc_default).\n";
    return gp_cov_;
  }

  // 2. Velocità
  std::vector<Eigen::Matrix<double, 6, 1>> velocities;
  velocities.reserve(poses.size() - 1);
  for (size_t i = 0; i < poses.size() - 1; ++i)
    velocities.push_back(logMapSE3(poses[i].inverse() * poses[i + 1]) / ts_);

  // 3. Accelerazioni
  std::vector<Eigen::Matrix<double, 6, 1>> accelerations;
  accelerations.reserve(velocities.size() - 1);
  for (size_t i = 0; i < velocities.size() - 1; ++i) {
    const Eigen::Matrix3d R = poses[i].linear().transpose() * poses[i + 1].linear();
    Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();
    adj.block<3, 3>(0, 0) = R;
    adj.block<3, 3>(3, 3) = R;
    accelerations.push_back((adj * velocities[i + 1] - velocities[i]) / ts_);
  }

  // 4. Stima della varianza delle accelerazioni (varianza campionaria con
  //    sigma-clipping a 5σ per rimuovere solo outlier estremi del GT).
  //
  // Razionale: la MAD riduce troppo aggressivamente la varianza per dataset
  // dinamici (es. math_medium), rendendo il GP factor "troppo forte" rispetto
  // al prior ICP e peggiorando l'RMSE su traiettorie non-WNOA.
  // La varianza campionaria "grezza" riflette correttamente l'alta variabilità
  // del dataset; rimuoviamo SOLO i salti estremi del GT (loop-closure artifacts)
  // che sono qualitativamente diversi dal moto genuino.
  //
  // Algoritmo:
  //   1. Calcola media e deviazione standard per ogni componente
  //   2. Scarta le accelerazioni a distanza > 5σ dalla media (outlier estremi)
  //   3. Ricalcola la varianza campionaria sul sottoinsieme filtrato
  const int n_acc = static_cast<int>(accelerations.size());
  Eigen::Matrix<double, 6, 1> variance;

  for (int j = 0; j < 6; ++j) {
    // Pass 1: media e sigma di tutti i valori
    double sum = 0.0;
    for (int i = 0; i < n_acc; ++i) sum += accelerations[i][j];
    const double mean1 = sum / n_acc;
    double sq_sum = 0.0;
    for (int i = 0; i < n_acc; ++i) {
      const double d = accelerations[i][j] - mean1;
      sq_sum += d * d;
    }
    const double sigma1 = std::sqrt(sq_sum / n_acc);

    // Pass 2: varianza campionaria escludendo outlier > 5σ
    const double threshold = 5.0 * sigma1;
    double sum2 = 0.0, sq2 = 0.0;
    int cnt = 0;
    for (int i = 0; i < n_acc; ++i) {
      if (std::abs(accelerations[i][j] - mean1) <= threshold) {
        sum2 += accelerations[i][j];
        ++cnt;
      }
    }
    if (cnt < 2) { variance[j] = sq_sum / n_acc; continue; }
    const double mean2 = sum2 / cnt;
    for (int i = 0; i < n_acc; ++i) {
      if (std::abs(accelerations[i][j] - mean1) <= threshold) {
        const double d = accelerations[i][j] - mean2;
        sq2 += d * d;
      }
    }
    variance[j] = sq2 / (cnt - 1);
  }

  // 5. Qd (formula WNOA)
  // cwiseMax: floor numerico per evitare Qc singolare.
  // Il MAD è già sufficiente per gestire gli outlier del GT (es. salti di pose da loop closure).
  // Non si applica un cap superiore: Qc grande per dataset dinamici è statisticamente corretto.
  const Eigen::Matrix<double, 6, 6> Qc =
      variance.cwiseMax(1e-4).asDiagonal();
  std::cout << "[VelEstimator] Qc_diag MAD (" << poses.size() << " pose): ["
            << variance.transpose() << "]\n";
  std::cout << "[VelEstimator] Qc_diag usata (cap=" << qc_default_ << "): ["
            << Qc.diagonal().transpose() << "]\n";
  Eigen::Matrix<double, 12, 12> Qd = Eigen::Matrix<double, 12, 12>::Zero();
  Qd.block<6, 6>(0, 0) = std::pow(ts_, 3) / 3.0 * Qc;
  Qd.block<6, 6>(0, 6) = std::pow(ts_, 2) / 2.0 * Qc;
  Qd.block<6, 6>(6, 0) = std::pow(ts_, 2) / 2.0 * Qc;
  Qd.block<6, 6>(6, 6) = ts_ * Qc;
  std::cout << "[VelEstimator] gp_cov_ calibrata da GT:\n" << Qd << "\n";
  return Qd;
}

void VelEstimator::setOdometry(const std::vector<Eigen::Isometry3d>& odometry) {
  odometry_ = odometry;
}

void VelEstimator::setPriorCovariances(const std::vector<Matrix6d>& covs) {
  prior_covs_ = covs;
}

void VelEstimator::errorAndJacobian(Vector6d& e,
                                    Matrix6d& J,
                                    const Eigen::Isometry3d& T_now,
                                    const Eigen::Isometry3d& T_prev,
                                    const double delta_t) {
  Eigen::Isometry3d T_now_to_prev = T_prev.inverse() * T_now;
  e.block<3, 1>(0, 0) = delta_t * X_.block<3, 1>(0, 0) - T_now_to_prev.translation();

  Eigen::Vector3d angles;
  angles(0)           = atan2(-T_now_to_prev.linear()(1, 2), T_now_to_prev.linear()(2, 2));
  angles(1)           = asin(T_now_to_prev.linear()(0, 2));
  angles(2)           = atan2(-T_now_to_prev.linear()(0, 1), T_now_to_prev.linear()(0, 0));
  e.block<3, 1>(3, 0) = delta_t * X_.block<3, 1>(3, 0) - angles;

  J = Matrix6d::Identity() * delta_t;
}

void VelEstimator::update(const Eigen::Isometry3d& T_now,
                          const Eigen::Isometry3d& T_prev,
                          const double delta_t,
                          const double weight) {
  Vector6d e;
  Matrix6d J;
  errorAndJacobian(e, J, T_now, T_prev, delta_t);

  double scale      = 1.;
  const double chi2 = e.squaredNorm();
  const double chi  = sqrt(chi2);
  if (chi > E_THRESHOLD_VEL)
    scale = E_THRESHOLD_VEL / chi;

  H_adder_ += scale * weight * J.transpose() * J;
  b_adder_ += scale * weight * J.transpose() * e;
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────
void VelEstimator::oneRound() {
  static bool printed = false;
  if (!printed) {
    std::cout << "\n"
          << "========================================\n"
          << "   [VelEstimator] SOLVER ATTIVO: "
          << solver_type_
          << "\n"
          << "========================================\n"
          << std::endl;
    printed = true;
  }
  if      (solver_type_ == "gsolver") oneRoundGSolver();
  else if (solver_type_ == "gtsam")   oneRoundGTSAM();
  else                                oneRoundClassic();
}

// ─── oneRoundClassic ──────────────────────────────────────────────────────────
void VelEstimator::oneRoundClassic() {
  H_adder_.setZero();
  b_adder_.setZero();

  const Eigen::Isometry3d T_now = odometry_.back();

  for (size_t i = 0; i < odometry_.size() - 1; ++i) {
    const Eigen::Isometry3d T_prev = odometry_[i];
    const double delta_t           = (odometry_.size() - 1 - i) * ts_;
    const double weight = 1.f - double(odometry_.size() - 2 - i) / double(odometry_.size() - 1);
    update(T_now, T_prev, delta_t, weight);
  }

  Vector6d dx = H_adder_.ldlt().solve(-b_adder_);
  X_ += dx;
}

