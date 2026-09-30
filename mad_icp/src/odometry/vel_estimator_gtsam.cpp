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

// ─── SE3 utilities ────────────────────────────────────────────────────────────
// Stessa convenzione di gsolver::geometry3d:
//   logMapSE3(T)     = [T.translation(); logMapSO3(T.linear())]   [t; ω]
//   expMapSE3([t;ω]) = { R=expMapSO3(ω), t }
//
// Questa convenzione è usata sia nelle funzioni di errore che nel calcolo
// numerico dei Jacobiani (perturbazione destra: T' = T · expMapSE3(δ)).

static inline Eigen::Matrix<double, 6, 1> logMapSE3_gtsam(const Eigen::Isometry3d& T) {
  Eigen::Matrix<double, 6, 1> x;
  x.head<3>() = T.translation();
  x.tail<3>() = logMapSO3(T.linear());
  return x;
}

static inline Eigen::Isometry3d expMapSE3_gtsam(const Eigen::Matrix<double, 6, 1>& x) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation()     = x.head<3>();
  T.linear()          = expMapSO3(x.tail<3>());
  return T;
}

// ─── Inclusi GTSAM ────────────────────────────────────────────────────────────
// Attivi solo se GTSAM_AVAILABLE è definito in CMakeLists.txt.
// Per abilitare:
//   cmake -DUSE_GP_VEL_ESTIMATOR=ON ..
// Richiede GTSAM installato:  sudo apt install libgtsam-dev
#ifdef GTSAM_AVAILABLE
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

// Forward declaration delle funzioni di errore (definite più in basso nel file).
static Eigen::Matrix<double, 12, 1> computeGPError(
    const Eigen::Isometry3d&, const Eigen::Matrix<double, 6, 1>&,
    const Eigen::Isometry3d&, const Eigen::Matrix<double, 6, 1>&, double);
static Eigen::Matrix<double, 6, 1> computePriorError(
    const Eigen::Isometry3d&, const Eigen::Isometry3d&);

// ─── SE3PosePriorFactor ───────────────────────────────────────────────────────
// Factor unario: ancora la posa del nodo i alla misura ICP.
//
// Stato:   pose_vec ∈ R^6 = perturbazione δ tale che T_i = z_i · expMapSE3(δ)
// Errore:  logMapSE3(z_i^{-1} · T_i) = logMapSE3(expMapSE3(δ)) ≈ δ       [6D]
// Jacobiano della perturbazione: coincide con la derivata destra usata in gsolver.
class SE3PosePriorFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector6> {
  Eigen::Isometry3d z_;  // posa ICP di riferimento (misura)
public:
  SE3PosePriorFactor(gtsam::Key key, const Eigen::Isometry3d& z,
                     const gtsam::SharedNoiseModel& model)
      : NoiseModelFactorN(model, key), z_(z) {}

  gtsam::Vector evaluateError(
      const gtsam::Vector6& pose_vec,
      boost::optional<gtsam::Matrix&> H = boost::none) const override {
    const Eigen::Isometry3d T = z_ * expMapSE3_gtsam(pose_vec);
    const gtsam::Vector6    e = computePriorError(z_, T);
    if (H) {
      H->resize(6, 6);
      constexpr double eps = 1e-5;
      for (int k = 0; k < 6; ++k) {
        gtsam::Vector6 d = gtsam::Vector6::Zero(); d[k] = eps;
        H->col(k) = (computePriorError(z_, z_ * expMapSE3_gtsam(pose_vec + d)) - e) / eps;
      }
    }
    return e;
  }
};

// ─── SE3PoseVelGPFactor ───────────────────────────────────────────────────────
// Factor quaternario: collegamento GP WNOA tra nodi consecutivi.
//
// Stato i: (pose_vec_i, vel_i),  Stato j: (pose_vec_j, vel_j)
// Errore:  computeGPError(Ti, vi, Tj, vj, dt)  [12D]
//   dove Ti = ref_i · expMapSE3(pose_vec_i)  (perturbazione destra dalla misura ICP)
class SE3PoseVelGPFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6,
                                      gtsam::Vector6, gtsam::Vector6> {
  Eigen::Isometry3d ref_i_, ref_j_;  // misure ICP dei nodi i e j
  double dt_;
public:
  SE3PoseVelGPFactor(gtsam::Key Pi, gtsam::Key Vi, gtsam::Key Pj, gtsam::Key Vj,
                     const Eigen::Isometry3d& ref_i, const Eigen::Isometry3d& ref_j,
                     double dt, const gtsam::SharedNoiseModel& model)
      : NoiseModelFactorN(model, Pi, Vi, Pj, Vj),
        ref_i_(ref_i), ref_j_(ref_j), dt_(dt) {}

  gtsam::Vector evaluateError(
      const gtsam::Vector6& xi, const gtsam::Vector6& vi,
      const gtsam::Vector6& xj, const gtsam::Vector6& vj,
      boost::optional<gtsam::Matrix&> Hi  = boost::none,
      boost::optional<gtsam::Matrix&> Hvi = boost::none,
      boost::optional<gtsam::Matrix&> Hj  = boost::none,
      boost::optional<gtsam::Matrix&> Hvj = boost::none) const override {
    const Eigen::Isometry3d          Ti = ref_i_ * expMapSE3_gtsam(xi);
    const Eigen::Isometry3d          Tj = ref_j_ * expMapSE3_gtsam(xj);
    const Eigen::Matrix<double,12,1> e  = computeGPError(Ti, vi, Tj, vj, dt_);
    constexpr double eps = 1e-5;
    if (Hi) {
      Hi->resize(12, 6);
      for (int k = 0; k < 6; ++k) {
        gtsam::Vector6 d = gtsam::Vector6::Zero(); d[k] = eps;
        Hi->col(k) = (computeGPError(ref_i_ * expMapSE3_gtsam(xi + d), vi, Tj, vj, dt_) - e) / eps;
      }
    }
    if (Hvi) {
      Hvi->resize(12, 6);
      for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<double,6,1> vi_p = vi; vi_p[k] += eps;
        Hvi->col(k) = (computeGPError(Ti, vi_p, Tj, vj, dt_) - e) / eps;
      }
    }
    if (Hj) {
      Hj->resize(12, 6);
      for (int k = 0; k < 6; ++k) {
        gtsam::Vector6 d = gtsam::Vector6::Zero(); d[k] = eps;
        Hj->col(k) = (computeGPError(Ti, vi, ref_j_ * expMapSE3_gtsam(xj + d), vj, dt_) - e) / eps;
      }
    }
    if (Hvj) {
      Hvj->resize(12, 6);
      for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<double,6,1> vj_p = vj; vj_p[k] += eps;
        Hvj->col(k) = (computeGPError(Ti, vi, Tj, vj_p, dt_) - e) / eps;
      }
    }
    return e;
  }
};

#endif  // GTSAM_AVAILABLE

// ─────────────────────────────────────────────────────────────────────────────
// computeGPError — funzione di errore del GP factor (identica a gsolver SE3PoseVelGP)
//
//   error_p = logMapSE3( exp(dt·v_i)⁻¹ · T_i⁻¹ · T_j )   [6D, conv. [t;ω]]
//   error_v = Adj(R_i^T·R_j)·v_j  −  v_i                  [6D]
//
// Prior WNOA (White Noise On Acceleration) su SE(3).
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::Matrix<double, 12, 1> computeGPError(
    const Eigen::Isometry3d&           Ti,
    const Eigen::Matrix<double, 6, 1>& vi,
    const Eigen::Isometry3d&           Tj,
    const Eigen::Matrix<double, 6, 1>& vj,
    double                             dt) {
  const Eigen::Matrix<double, 6, 1> error_p =
      logMapSE3_gtsam(expMapSE3_gtsam(dt * vi).inverse() * Ti.inverse() * Tj);

  const Eigen::Matrix3d R = Ti.linear().transpose() * Tj.linear();
  Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();
  adj.block<3, 3>(0, 0) = R;
  adj.block<3, 3>(3, 3) = R;
  const Eigen::Matrix<double, 6, 1> error_v = adj * vj - vi;

  Eigen::Matrix<double, 12, 1> e;
  e.head<6>() = error_p;
  e.tail<6>() = error_v;
  return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// computePriorError — funzione di errore del Prior factor (identica a gsolver SE3PoseVelPrior)
//
//   error = logMapSE3( z⁻¹ · T )   [6D]
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::Matrix<double, 6, 1> computePriorError(
    const Eigen::Isometry3d& z,
    const Eigen::Isometry3d& T) {
  return logMapSE3_gtsam(z.inverse() * T);
}

// ─────────────────────────────────────────────────────────────────────────────
// oneRoundGTSAM — Factor graph GP con ottimizzazione GN centralizzata (GTSAM)
//
// Struttura del grafo (identica a oneRoundGSolver ma ottimizzata con GTSAM GN):
//   - N prior factor unari sulle pose (SE3PosePriorFactor)
//   - N-1 GP factor binari tra stati consecutivi (SE3PoseVelGPFactor)
// ─────────────────────────────────────────────────────────────────────────────
void VelEstimator::oneRoundGTSAM() {
#ifdef GTSAM_AVAILABLE
  const int N = static_cast<int>(odometry_.size());
  if (N < 2) return; // Finestra troppo corta: non ci sono GP factor.

  // ─── Chiavi ───────────────────────────────────────────────────────────────
  // P(i): posa del nodo i  (6 DOF, perturbazione δ da odometry_[i])
  // V(i): velocità del nodo i  (6 DOF, vettore Euclideo)
  const auto P = [](int i) -> gtsam::Key { return gtsam::Symbol('p', i); };
  const auto V = [](int i) -> gtsam::Key { return gtsam::Symbol('v', i); };

  // ─── Noise models ─────────────────────────────────────────────────────────
  // prior_cov_ < 0 (sentinella) → usa la covarianza ICP reale per-posa (prior_covs_[i]);
  // altrimenti Identity * prior_cov_ per tutti i nodi (fallback 1e-3*I se prior_covs_ vuoto).
  constexpr double kPriorCovFallback = 1e-3;
  const auto gp_model = gtsam::noiseModel::Gaussian::Covariance(gp_cov_);

  // ─── STEP 1 — Inizializzazione stato ──────────────────────────────────────
  // pose_vec_i = 0  →  T_cur[i] = odometry_[i] · expMapSE3(0) = odometry_[i]
  // vel_i = X_      →  velocità dell'ultimo frame come punto di partenza
  gtsam::Values values;
  for (int i = 0; i < N; ++i) {
    values.insert(P(i), gtsam::Vector6(gtsam::Vector6::Zero()));
    values.insert(V(i), gtsam::Vector6(X_));
  }

  // ─── STEP 2 — Prior factor su ogni posa ───────────────────────────────────
  // Ancora ogni nodo alla posa stimata da MAD-ICP (misura z_i = odometry_[i]).
  // ─── STEP 3 — GP factor tra stati consecutivi ─────────────────────────────
  // Prior WNOA su SE(3): collega pose e velocità di frame consecutivi.
  //
  // Il grafo viene costruito UNA SOLA VOLTA qui (parallelo a gsolver STEP 1-3).
  gtsam::NonlinearFactorGraph nfg;

  for (int i = 0; i < N; ++i) {
    const Eigen::Matrix<double, 6, 6> prior_cov =
        (prior_cov_ < 0.0)
            ? (i < static_cast<int>(prior_covs_.size())
                   ? prior_covs_[i]
                   : Eigen::Matrix<double, 6, 6>::Identity() * kPriorCovFallback)
            : Eigen::Matrix<double, 6, 6>::Identity() * prior_cov_;
    const auto prior_model = gtsam::noiseModel::Gaussian::Covariance(prior_cov);
    nfg.add(boost::make_shared<SE3PosePriorFactor>(P(i), odometry_[i], prior_model));
  }

  for (int i = 0; i < N - 1; ++i)
    nfg.add(boost::make_shared<SE3PoseVelGPFactor>(
        P(i), V(i), P(i + 1), V(i + 1),
        odometry_[i], odometry_[i + 1],
        ts_, gp_model));

  // ─── STEP 4 — Ottimizzazione iterativa con Gauss-Newton ───────────────────
  // Parallelo a gsolver:  GbpSolver solver; for (iter) solver.performIteration(graph);
  // optimizer.iterate() ri-linearizza nfg al punto corrente e risolve con Cholesky,
  // senza ricostruire il grafo (le strutture dei factor rimangono invariate).
  gtsam::GaussNewtonParams params;
  gtsam::GaussNewtonOptimizer optimizer(nfg, values, params);
  for (int iter = 0; iter < gn_iterations_; ++iter)
    optimizer.iterate();
  const gtsam::Values result = optimizer.values();

  // ─── STEP 5 — Estrazione della velocità dell'ultimo nodo ─────────────────
  // Parallelo a gsolver:  X_ = last_node->mu_.vel;
  X_ = result.at<gtsam::Vector6>(V(N - 1));

#else
  // GTSAM non disponibile: X_ rimane invariata.
  // Per abilitare: installare GTSAM (sudo apt install libgtsam-dev)
  // e ricompilare con  cmake -DUSE_GP_VEL_ESTIMATOR=ON ..
  std::cerr << "[VelEstimator] ATTENZIONE: solver=\"gtsam\" ma GTSAM_AVAILABLE non definito.\n"
            << "               X_ rimane invariata. Verificare CMakeLists.txt.\n";
#endif
}
