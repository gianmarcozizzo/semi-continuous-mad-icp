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

#pragma once

#include <tools/lie_algebra.h>
#include <tools/utils.h>

#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct VelEstimator {
  // solver_type: seleziona il metodo di stima della velocità
  //   "classic" → Gauss-Newton classico (sempre disponibile, default)
  //   "gsolver" → Factor graph GP con GBP distribuito (richiede GSOLVER_AVAILABLE)
  //   "gtsam"   → Factor graph GP con GN centralizzato (richiede GTSAM_AVAILABLE)
  // gt_path: percorso GT KITTI per calibrare Qc (vuoto = usa default 1e-2*I)
  // prior_cov: covarianza del prior factor sulle pose (solver "gsolver" | "gtsam").
  //             Se negativo (sentinella): usa prior_covs_[i], la covarianza ICP reale
  //             per-posa impostata da setPriorCovariances() (fallback 1e-3*I se assente).
  // gn_iterations: numero di iterazioni GN/GBP del factor graph (solver "gsolver" | "gtsam")
  VelEstimator(double sensor_hz,
               const std::string& solver_type = "classic",
               const std::string& gt_path = "",
               double qc_default = 0.5,
               double prior_cov = 1e-3,
               int gn_iterations = 30);

  void setOdometry(const std::vector<Eigen::Isometry3d>& odometry);
  void init(const Vector6d& velocity = Vector6d::Zero());

  // Imposta la finestra di covarianze ICP (una per posa in odom_window).
  void setPriorCovariances(const std::vector<Matrix6d>& covs);

  // Dispatcher: chiama oneRoundClassic / oneRoundGSolver / oneRoundGTSAM
  // in base a solver_type_
  void oneRound();

  // Metodi ausiliari dell'implementazione classica (Gauss-Newton manuale)
  void errorAndJacobian(Vector6d& e,
                        Matrix6d& J,
                        const Eigen::Isometry3d& T_now,
                        const Eigen::Isometry3d& T_prev,
                        const double delta_t);
  void update(const Eigen::Isometry3d& T_now, const Eigen::Isometry3d& T_prev,
              const double delta_t, const double weight);

  // Stato interno
  Matrix6d H_adder_;
  Vector6d X_, b_adder_;
  std::vector<Eigen::Isometry3d> odometry_;
  double ts_;
  double qc_default_;  // intensità del rumore di processo WNOA
  double prior_cov_;  // covarianza del prior factor sulle pose (gsolver | gtsam); < 0 → usa prior_covs_[i]
  int gn_iterations_;  // numero di iterazioni GN/GBP del factor graph (gsolver | gtsam)
  std::string solver_type_;               // "classic" | "gsolver" | "gtsam"
  Eigen::Matrix<double, 12, 12> gp_cov_; // covarianza discreta Qd (WNOA), calcolata nel costruttore
  std::vector<Matrix6d> prior_covs_;      // covarianze ICP per-posa (una per frame in odom_window)

private:
  // Gauss-Newton classico: velocità costante su finestra scorrevole
  void oneRoundClassic();

  // Factor graph GP con GBP distribuito (gsolver)
  // Richiede: GSOLVER_AVAILABLE definito in CMakeLists.txt
  void oneRoundGSolver();

  // Factor graph GP con GN centralizzato (GTSAM)
  // Richiede: GTSAM_AVAILABLE definito in CMakeLists.txt
  void oneRoundGTSAM();

  // Calcola Qd (12×12) dalla ground truth KITTI
  // pose KITTI → velocità → accelerazioni → varianza → qc_diag → Qd (formula WNOA)
  // Standalone: non richiede gsolver né GTSAM.
  // ground truth KITTI seguendo la logica di gp_hyperparam_trainer.cpp:
  // pose KITTI → velocità → accelerazioni → varianza → qc_diag → Qd (formula WNOA)
  // Standalone: non richiede gsolver né GTSAM.
  Eigen::Matrix<double, 12, 12> computeQdFromKittiGT(const std::string& path);
};
