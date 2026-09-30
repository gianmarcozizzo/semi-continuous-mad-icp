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

// Inclusi gsolver: attivi solo se GSOLVER_AVAILABLE è definito in CMakeLists.txt.
// Per abilitare:
//   1. FetchContent_Declare(gsolver GIT_REPOSITORY https://github.com/rvp-group/gsolver.git)
//   2. FetchContent_MakeAvailable(gsolver)
//   3. target_link_libraries(<target> PRIVATE gsolver)
//   4. target_compile_definitions(<target> PRIVATE GSOLVER_AVAILABLE)
#ifdef GSOLVER_AVAILABLE
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/variables/se3_posevel.h>
#include <gsolver/graph/types/factors/se3_posevel_gp.h>
#include <gsolver/graph/types/factors/se3_posevel_prior.h>
#include <gsolver/core/gbp_solver.h>
#endif

// Implementazione del factor graph GP con ottimizzazione GBP distribuita (gsolver).
// La struttura del grafo è:
//   - N variabili SE3PoseVel (posa + velocità, una per ogni frame nella finestra)
//   - N prior factor unari: ancorano ogni posa alla stima di MAD-ICP    ["quadrati" nel grafo]
//   - N-1 GP factor binari: impongono il prior WNOA tra stati consecutivi ["triangoli" nel grafo]
// L'ottimizzazione GBP su una catena converge alla soluzione GN esatta.
// X_ viene aggiornata con la velocità dell'ultimo stato ottimizzato.
void VelEstimator::oneRoundGSolver() {
#ifdef GSOLVER_AVAILABLE

  using namespace gsolver;

  const int N = static_cast<int>(odometry_.size());

  // -------------------------------------------------------------------------
  // STEP 1 — Aggiunta variabili SE3PoseVel al grafo
  // -------------------------------------------------------------------------
  FactorGraph graph;

  // Covarianza iniziale del nodo: valore grande → la variabile si affida
  // interamente ai factor (prior + GP) e non al punto di partenza.
  Eigen::Matrix<double, 12, 12> sigma_var =
      Eigen::Matrix<double, 12, 12>::Identity() * 1e6;

  for (int i = 0; i < N; ++i) {
    SE3PoseVelValue mu;
    mu.pose = odometry_[i]; // posa iniziale: stima di MAD-ICP
    mu.vel  = X_;           // velocità iniziale: stima del frame precedente

    // Timestamp relativo all'inizio della finestra (nodo 0 = posa più vecchia)
    const double node_time = i * ts_;

    auto node = std::make_shared<SE3PoseVel>(
        "POSE_" + std::to_string(i), mu, sigma_var, node_time);
    graph.addVariableNode(node);
  }

  // -------------------------------------------------------------------------
  // STEP 2 — Prior factor su ogni posa (factor unario, "quadrato" nel grafo)
  // -------------------------------------------------------------------------
  // Ancora ogni nodo variabile alla posa stimata da MAD-ICP.
  // prior_cov_ < 0 (sentinella, impostata da mad_params.py con "prior_cov" vuoto/None) →
  //   usa la covarianza ICP reale per-posa prior_covs_[i] (da icp_.H_adder_.inverse() in
  //   pipeline.cpp): piccola (ICP ben condizionato) → posa quasi fissa, si ottimizzano solo
  //   le velocità; grande (ICP mal condizionato) → posa libera di muoversi, smoothing completo.
  // prior_cov_ >= 0 → stesso valore scalare (Identity * prior_cov_) per tutti i nodi.
  // Fallback a 1e-3 * I se prior_covs_ non è stata impostata.
  constexpr double kPriorCovFallback = 1e-3;

  for (int i = 0; i < N; ++i) {
    const Eigen::Matrix<double, 6, 6> prior_cov =
        (prior_cov_ < 0.0)
            ? (i < static_cast<int>(prior_covs_.size())
                   ? prior_covs_[i]
                   : Eigen::Matrix<double, 6, 6>::Identity() * kPriorCovFallback)
            : Eigen::Matrix<double, 6, 6>::Identity() * prior_cov_;
    auto prior = std::make_shared<SE3PoseVelPrior>(
        "PRIOR_" + std::to_string(i),
        odometry_[i], // misura: posa da MAD-ICP
        prior_cov);
    graph.addFactorNode(prior,
        std::vector<std::string>{"POSE_" + std::to_string(i)});
  }

  // -------------------------------------------------------------------------
  // STEP 3 — GP factor tra coppie consecutive (factor binario, "triangolo" nel grafo)
  // -------------------------------------------------------------------------
  // Implementa il WNOA (White Noise On Acceleration) prior su SE(3):
  //   error_p = log( exp(dt * v_i)^{-1} * T_i^{-1} * T_{i+1} )   [6D]
  //   error_v = Adj(R) * v_{i+1} - v_i                            [6D]
  // gp_cov (Qc) piccola → traiettoria liscia (velocità quasi costante).
  // gp_cov (Qc) grande  → permette variazioni rapide di velocità.
  // gp_cov_: calcolata nel costruttore da GT (se gt_path fornito) oppure 1e-2 * I di default
  const Eigen::Matrix<double, 12, 12>& gp_cov = gp_cov_;

  for (int i = 0; i < N - 1; ++i) {
    auto gp_factor = std::make_shared<SE3PoseVelGP>(
        "GP_" + std::to_string(i) + "_" + std::to_string(i + 1),
        ts_,    // delta_t tra frame consecutivi (= 1 / sensor_hz)
        gp_cov);
    graph.addFactorNode(gp_factor, std::vector<std::string>{
        "POSE_" + std::to_string(i),
        "POSE_" + std::to_string(i + 1)});
  }

  // -------------------------------------------------------------------------
  // STEP 4 — Ottimizzazione con GBP (equivalente a GN su grafi a catena)
  // -------------------------------------------------------------------------
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int iter = 0; iter < gn_iterations_; ++iter) {
    solver.performIteration(graph);
  }

  // -------------------------------------------------------------------------
  // STEP 5 — Estrazione della velocità dall'ultimo nodo
  // -------------------------------------------------------------------------
  auto last_node = std::static_pointer_cast<SE3PoseVel>(
      graph.getVariableNode("POSE_" + std::to_string(N - 1)));
  X_ = last_node->mu_.vel; // velocità ottimizzata dell'ultimo stato → aggiorna X_

#else
  std::cerr << "[VelEstimator] ATTENZIONE: solver=\"gsolver\" ma GSOLVER_AVAILABLE non definito.\n"
            << "               X_ rimane invariata. Verificare CMakeLists.txt.\n";
#endif
}
