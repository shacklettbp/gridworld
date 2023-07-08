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
                       int64_t grid_x,
                       int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;
            cells[idx].reward = static_cast<float>(rewards[idx]);
        }
    }
}

template <typename T>
static void tagWalls(Cell *cells,
                     T *walls,
                     int64_t grid_x,
                     int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;

            if (!!walls[idx]) {
                cells[idx].flags |= CellFlag::Wall;
            }
        }
    }
}

template <typename T>
static void tagEnd(Cell *cells,
                   T *end_cells,
                   int64_t num_end_cells,
                   int64_t grid_x,
                   int64_t grid_y)
{
    for (int64_t c = 0; c < num_end_cells; c++) {
        int64_t idx = c * 2;
        int64_t y = static_cast<int64_t>(end_cells[idx]);
        int64_t x = static_cast<int64_t>(end_cells[idx + 1]);

        if (x >= grid_x || y >= grid_y) {
            throw std::runtime_error("Out of range end cells");
        }

        cells[y * grid_x + x].flags |= CellFlag::End;
    }
}

static Cell * setupCellData(
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
        nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
        nb::c_contig, nb::device::cpu> &rewards,
    const nb::ndarray<void, nb::shape<nb::any, 2>,
        nb::c_contig, nb::device::cpu> &end_cells,
    int64_t grid_x,
    int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();
    
    if (rewards.dtype() == nb::dtype<float>()) {
        setRewards(cells, (float *)rewards.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<double>()) {
        setRewards(cells, (double *)rewards.data(), grid_x, grid_y);
    } else {
        throw std::runtime_error("rewards: unsupported input type");
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
        throw std::runtime_error("walls: unsupported input type");
    }
    
    if (end_cells.dtype() == nb::dtype<uint32_t>() ||
            end_cells.dtype() == nb::dtype<int32_t>()) {
        tagEnd(cells, (uint32_t *)end_cells.data(),
               (int64_t)end_cells.shape(0), grid_x, grid_y);
    } else if (end_cells.dtype() == nb::dtype<uint64_t>() ||
               end_cells.dtype() == nb::dtype<int64_t>()) {
        tagEnd(cells, (uint64_t *)end_cells.data(),
               (int64_t)end_cells.shape(0), grid_x, grid_y);
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
                                nb::c_contig, nb::device::cpu> end_cells,
                            int64_t start_x,
                            int64_t start_y,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {
            int64_t grid_y = (int64_t)walls.shape(0);
            int64_t grid_x = (int64_t)walls.shape(1);

            if ((int64_t)rewards.shape(0) != grid_y ||
                (int64_t)rewards.shape(1) != grid_x) {
                throw std::runtime_error("walls and rewards shapes don't match");
            }

            Cell *cells =
                setupCellData(walls, rewards, end_cells, grid_x, grid_y);

            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
            }, GridState {
                .cells = cells,
                .startX = (int32_t)start_x,
                .startY = (int32_t)start_y,
                .width = (int32_t)grid_x,
                .height = (int32_t)grid_y,
            });

            delete[] cells;
        }, nb::arg("walls"),
           nb::arg("rewards"),
           nb::arg("end_cells"),
           nb::arg("start_x"),
           nb::arg("start_y"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("gpu_id") = -1)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
    ;
}

}
