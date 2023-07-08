#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace madgrid {

template <typename T>
static void setRewards(Cell *cells,
                       T *rewards,
                       uint32_t grid_x,
                       uint32_t grid_y)
{
    for (uint32_t y = 0; y < grid_y; y++) {
        for (uint32_t x = 0; x < grid_x; x++) {
            uint32_t idx = y * grid_x + x;
            cells[idx].reward = static_cast<float>(rewards[idx]);
        }
    }
}

template <typename T>
static void tagWalls(Cell *cells,
                     T *walls,
                     uint32_t grid_x,
                     uint32_t grid_y)
{
    for (uint32_t y = 0; y < grid_y; y++) {
        for (uint32_t x = 0; x < grid_x; x++) {
            uint32_t idx = y * grid_x + x;

            if (!!walls[i]) {
                cells[idx].flags |= CellFlag::Wall;
            }
        }
    }
}

template <typename T>
static void tagEnd(Cell *cells,
                   T *end_cells,
                   uint32_t num_end_cells,
                   uint32_t grid_x,
                   uint32_t grid_y)
{
    for (uint32_t c = 0; c < num_end_cells; c++) {
        uint32_t idx = c * 2;
        uint32_t x = static_cast<uint32_t>(end_cells[idx]);
        uint32_t y = static_cast<uint32_t>(end_cells[idx + 1]);

        if (x >= grid_x || y >= grid_y) {
            FATAL("Out of range end cells");
        }

        cells[y * grid_x + x].flags |= CellFlag::End;
    }
}

static void setupCellData(
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
                      nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
                      nb::c_contig, nb::device::cpu> &rewards,
    const nb::ndarray<void, nb::shape<nb::any, 2>,
                      nb::c_contig, nb::device_cpu> &end_cells,
                      uint32_t grid_x,
                      uint32_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();
    
    if (rewards.dtype() == nb::dtype<float>()) {
        setRewards(cells, (float *)rewards.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<double>()) {
        setRewards(cells, (double *)rewards.data(), grid_x, grid_y);
    } else {
        FATAL("rewards: unsupported input type");
    }
    
    if (rewards.dtype() == nb::dtype<float>()) {
        tagWalls(cells, (float *)walls.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<double>()) {
        tagWalls(cells, (double *)walls.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<uint32_t>() ||
               rewards.dtype() == nb::dtype<int32_t>()) {
        tagWalls(cells, (uint32_t *)walls.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<uint64_t>() ||
               rewards.dtype() == nb::dtype<int64_t>()) {
        tagWalls(cells, (uint64_t *)walls.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<uint8_t>()) {
        tagWalls(cells, (uint8_t *)walls.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<bool>()) {
        tagWalls(cells, (bool *)walls.data(), grid_x, grid_y);
    } else {
        FATAL("walls: unsupported input type");
    }
    
    if (end_cells.dtype() == nb::dtype<uint32_t>() ||
            end_cells.dtype() == nb::dtype<int32_t>()) {
        tagEnd(cells, (uint32_t *)end_cells.data(), end_cells.shape(0),
               grid_x, grid_y);
    } else if (end_cells.dtype() == nb::dtype<uint64_t>() ||
               end_cells.dtype() == nb::dtype<int64_t>()) {
        tagEnd(cells, (uint64_t *)end_cells.data(), end_cells.shape(0),
               grid_x, grid_y);
    }

    return cells;
}

NB_MODULE(gridworld_madrona, m) {
    nb::module_::import_("madrona_python");

    nb::class_<Manager> (m, "GridWorldSimulator")
        .def("__init__", [](Manager *self,
                            nb::ndarray<void, nb::shape<nb::any, nb::any>,
                                        nb::c_contig, nb::device::cpu> walls,
                            nb::ndarray<void, nb::shape<nb::any, nb::any>,
                                        nb::c_contig, nb::device::cpu> rewards,
                            nb::ndarray<void, nb::shape<nb::any, 2>,
                                        nb::c_contig, nb::device_cpu> end_cells,
                            uint32_t start_x,
                            uint32_t start_y,
                            madrona::python::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {
            uint32_t grid_x = walls.shape(0);
            uint32_t grid_y = walls.shape(1);

            if (rewards.shape(0) != grid_x || rewards.shape(1) != grid_y) {
                FATAL("walls and rewards shapes don't match")
            }

            setupCellData(walls, rewards, end_cells, grid_x, grid_y);

            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
            }, Manager::GridState {
                .cells = cells,
                .startX = start_x,
                .startY = start_y,
                .width = grid_x,
                .height = grid_y,
            });

            delete[] cells;
        }, nb::arg("walls"),
           nb::arg("rewards"),
           nb::arg("start_x),
           nb::arg("start_y"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("gpu_id", -1))
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("position_tensor", &Manager::positionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
    ;
}

}
