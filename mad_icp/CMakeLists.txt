cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(mad_icp LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

# this is required only to compile apps in C++, if you work in Python this can be set to off
# if need to compile C++ apps, when executing CMake run `cmake -DCOMPILE_CPP_APPS=ON ..`
option(COMPILE_CPP_APPS "Set to ON to compile C++ applications" OFF)

# Eigen
find_package(Eigen3 3.3 NO_MODULE)

if(${Eigen3_DIR} STREQUAL "Eigen3_DIR-NOTFOUND")
    message(CHECK_START "Fetching Eigen3")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")

    include(FetchContent)
    FetchContent_Declare(
        Eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
    FetchContent_GetProperties(Eigen)

    if(NOT eigen_POPULATED)
        FetchContent_Populate(Eigen)
        add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(CHECK_PASS "fetched")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python bindings
find_package(pybind11)

if(NOT DEFINED pybind11_FOUND)
    message(CHECK_START "Fetching pybind11")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.12
    )

    FetchContent_GetProperties(pybind11)

    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()

    message(CHECK_PASS "fetched")
endif()

# ─── GP Velocity Estimator (gsolver) ────────────────────────────────────────
# Abilitare con:  cmake -DUSE_GP_VEL_ESTIMATOR=ON ..
# oppure impostando la variabile nel pyproject.toml / setup.py prima del build.
# Default: OFF → usa il metodo Gauss-Newton classico, nessuna dipendenza extra.
option(USE_GP_VEL_ESTIMATOR "Abilita il velocity estimator basato su factor graph GP" OFF)

if(USE_GP_VEL_ESTIMATOR)
    message(STATUS "USE_GP_VEL_ESTIMATOR=ON: scarico gsolver...")
    include(FetchContent)
    FetchContent_Declare(
        gsolver
        GIT_REPOSITORY https://github.com/rvp-group/gsolver.git
        GIT_TAG        main
    )
    FetchContent_GetProperties(gsolver)
    if(NOT gsolver_POPULATED)
        # CMP0169: permette FetchContent_Populate diretto (deprecato in CMake >= 3.30)
        # La policy esiste solo da CMake 3.30 in poi: su versioni precedenti
        # (es. 3.28.x) va saltata, altrimenti cmake_policy() genera un errore
        # "Policy is not known to this version of CMake".
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Populate(gsolver)

        # ── Patch: corregge il CMakeLists.txt interno di gsolver ──────────────
        # Il file usa ${CMAKE_SOURCE_DIR} per i percorsi di packaging, ma quando
        # gsolver è incluso come subdirectory quella variabile punta al progetto
        # genitore (mad-icp), non a gsolver stesso.
        set(_inner_cmake "${gsolver_SOURCE_DIR}/gsolver/CMakeLists.txt")
        file(READ "${_inner_cmake}" _gsolver_content)

        # Fix 1: percorso del template cmake → usa CMAKE_CURRENT_SOURCE_DIR
        # che è sempre relativo al file stesso, indipendentemente da chi lo include.
        # NOTA: gsolver usa ${PROJECT_SOURCE_DIR}, non ${CMAKE_SOURCE_DIR}.
        string(REPLACE
            [=[${PROJECT_SOURCE_DIR}/cmake/gsolverConfig.cmake.in]=]
            [=[${CMAKE_CURRENT_SOURCE_DIR}/../cmake/gsolverConfig.cmake.in]=]
            _gsolver_content "${_gsolver_content}")

        # Fix 2: rimuove yaml-cpp e rerun_sdk dal target_link_libraries di gsolver.
        # grep sui sorgenti conferma che nessun file .cpp del core le include:
        # sono dichiarate nel CMakeLists.txt ma non usate nel codice matematico.
        # Rimuoverle evita di dover installare yaml-cpp e scaricare Rerun SDK.
        string(REPLACE
            [=[    PRIVATE
        yaml-cpp
        rerun_sdk]=]
            [=[    # yaml-cpp e rerun_sdk rimossi: non usati nei sorgenti del core]=]
            _gsolver_content "${_gsolver_content}")

        file(WRITE "${_inner_cmake}" "${_gsolver_content}")
        # ──────────────────────────────────────────────────────────────────────

        # write_basic_package_version_file richiede PROJECT_VERSION: lo impostiamo
        # manualmente perché fuori dal progetto gsolver questa variabile è vuota.
        set(PROJECT_VERSION "0.1.0")

        # Aggiunge SOLO la libreria core, bypassando il root CMakeLists.txt
        # che scaricherebbe Rerun SDK e yaml-cpp (non necessari per il core).
        add_subdirectory(${gsolver_SOURCE_DIR}/gsolver ${gsolver_BINARY_DIR}/gsolver_core)

        # mad-icp compila con -Werror: i warning di gsolver diventerebbero errori.
        # Disabilitiamo -Werror solo per la libreria gsolver.
        if(TARGET gsolver)
            target_compile_options(gsolver PRIVATE -Wno-error)
        endif()
    endif()
    message(STATUS "gsolver pronto")

    # ─── GTSAM ────────────────────────────────────────────────────────────────
    # Richiede: sudo apt install libgtsam-dev libboost-all-dev libtbb-dev libsuitesparse-dev
    # Bug noto Ubuntu 24.04: GTSAM-exports.cmake referenzia libCppUnitLite.a inesistente.
    # Fix:  sudo touch /usr/lib/x86_64-linux-gnu/libCppUnitLite.a
    message(STATUS "USE_GP_VEL_ESTIMATOR=ON: ricerco GTSAM...")

    # Pre-flight checks: verifica che le dipendenze di GTSAM siano presenti.
    find_package(Boost REQUIRED COMPONENTS serialization filesystem system thread date_time regex timer chrono)
    if(NOT Boost_FOUND)
        message(FATAL_ERROR
            "Boost non trovato. Installare con:\n"
            "  sudo apt install libboost-all-dev")
    endif()

    find_package(TBB QUIET)
    if(NOT TBB_FOUND)
        message(WARNING
            "TBB non trovato (opzionale per GTSAM). Installare con:\n"
            "  sudo apt install libtbb-dev")
    endif()

    # GTSAM-exports.cmake su Ubuntu 24.04 referenzia libCppUnitLite.a che non esiste
    # se gtsam-point-cloud-support non è installato. Creiamo uno stub vuoto se serve.
    set(_cppu "/usr/lib/x86_64-linux-gnu/libCppUnitLite.a")
    if(NOT EXISTS "${_cppu}")
        message(STATUS "GTSAM: creo stub libCppUnitLite.a (${_cppu})")
        execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${_cppu}"
            RESULT_VARIABLE _cppu_res)
        if(NOT _cppu_res EQUAL 0)
            message(WARNING
                "Impossibile creare ${_cppu}. "
                "Se find_package(GTSAM) fallisce, eseguire manualmente:\n"
                "  sudo touch ${_cppu}")
        endif()
    endif()

    find_package(GTSAM REQUIRED)
    if(GTSAM_FOUND)
        message(STATUS "GTSAM trovato: ${GTSAM_VERSION}")
    endif()
    # ──────────────────────────────────────────────────────────────────────────
endif()
# ─────────────────────────────────────────────────────────────────────────────

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

add_subdirectory(src)

# TODO only if one wants to build cpp runners
add_subdirectory(apps/cpp_runners)