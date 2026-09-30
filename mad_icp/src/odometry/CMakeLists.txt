add_library(odometry STATIC
  vel_estimator.cpp
  vel_estimator_gsolver.cpp
  vel_estimator_gtsam.cpp
  pipeline.cpp
  mad_icp.cpp
)

target_link_libraries(odometry
  tools
  Eigen3::Eigen
)
target_compile_features(odometry PUBLIC)

# Collega gsolver + GTSAM e attiva la compilazione dei rispettivi solver se USE_GP_VEL_ESTIMATOR=ON.
# GSOLVER_AVAILABLE sblocca il codice dentro #ifdef GSOLVER_AVAILABLE in vel_estimator_gsolver.cpp.
# GTSAM_AVAILABLE  sblocca il codice dentro #ifdef GTSAM_AVAILABLE  in vel_estimator_gtsam.cpp.
if(USE_GP_VEL_ESTIMATOR)
    target_link_libraries(odometry gsolver)
    target_compile_definitions(odometry PRIVATE GSOLVER_AVAILABLE)
    message(STATUS "odometry: gsolver collegato, GSOLVER_AVAILABLE attivo")

    target_link_libraries(odometry gtsam)
    target_compile_definitions(odometry PRIVATE GTSAM_AVAILABLE)
    message(STATUS "odometry: gtsam collegato, GTSAM_AVAILABLE attivo")
endif()