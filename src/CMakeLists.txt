set(SIMULATOR_SRCS
    sim.hpp sim.cpp
)

add_library(gridworld_madrona_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(gridworld_madrona_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

add_library(gridworld_madrona_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(gridworld_madrona_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    madrona_mw_cpu
    gridworld_madrona_cpu_impl
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gridworld_madrona_gpu_srcs
        SOURCES_DEFN
            SIMPLE_SRC_LIST
        FLAGS_DEFN
            SIMPLE_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(gridworld_madrona_mgr PRIVATE
        gridworld_madrona_gpu_srcs
        madrona_mw_gpu
    )
endif()

target_compile_definitions(gridworld_madrona_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(_gridworld_madrona
    bindings.cpp
)

target_link_libraries(_gridworld_madrona PRIVATE
    madrona_hdrs
    gridworld_madrona_mgr
)
