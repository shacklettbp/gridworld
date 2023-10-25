#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace madrona;
using namespace madrona::py;

namespace madgrid {

struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;
    GridState *gridData;
    Reset *worldResetBuffer;

    inline Impl(const Config &c,
                EpisodeManager *ep_mgr,
                GridState *grid_data)
        : cfg(c),
          episodeMgr(ep_mgr),
          gridData(grid_data),
          worldResetBuffer(nullptr)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuRollout(cudaStream_t strm,
                            void **buffers,
                            const TrainInterface &train_iface) = 0;
#endif

    virtual Tensor exportTensor(ExportID slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;

    static inline Impl * init(const Config &cfg, const GridState &src_grid);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   GridState *grid_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports,
              }, sim_cfg, world_inits)
    {
        worldResetBuffer = (Reset *)cpuExec.getExported(
            (uint32_t)ExportID::Reset);
    }

    inline virtual ~CPUImpl() final {
        delete episodeMgr;
        free(gridData);
    }

    inline virtual void run() final { cpuExec.run(); }

#ifdef MADRONA_CUDA_SUPPORT
    inline virtual void gpuRollout(cudaStream_t strm,
                                   void **buffers,
                                   const TrainInterface &train_iface)
    {
        (void)strm;
        (void)buffers;
        (void)train_iface;
        assert(false);
    }
#endif
    
    inline virtual Tensor exportTensor(ExportID slot,
                                       Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;

    inline GPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   GridState *grid_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data),
          gpuExec({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&sim_cfg,
                  .numUserConfigBytes = sizeof(Sim::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports, 
                  .gpuID = (uint32_t)mgr_cfg.gpuID,
              }, {
                  { SIMPLE_SRC_LIST },
                  { SIMPLE_COMPILE_FLAGS },
                  CompileConfig::OptMode::LTO,
              })
    {
        worldResetBuffer = (Reset *)gpuExec.getExported(
            (uint32_t)ExportID::Reset);
    }

    inline virtual ~GPUImpl() final {
        REQ_CUDA(cudaFree(episodeMgr));
        REQ_CUDA(cudaFree(gridData));
    }

    inline virtual void run() final { gpuExec.run(); }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuRollout(cudaStream_t strm,
                            void **buffers,
                            const TrainInterface &train_iface)
    {
        auto numTensorBytes = [](const Tensor &t) {
            uint64_t num_items = 1;
            uint64_t num_dims = t.numDims();
            for (uint64_t i = 0; i < num_dims; i++) {
                num_items *= t.dims()[i];
            }

            return num_items * (uint64_t)t.numBytesPerItem();
        };

        auto copyToSim = [&strm, &numTensorBytes](const Tensor &dst, void *src) {
            uint64_t num_bytes = numTensorBytes(dst);

            REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        auto copyFromSim = [&strm, &numTensorBytes](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        Span<const TrainInterface::NamedTensor> src_obs =
            train_iface.observations();
        Span<const TrainInterface::NamedTensor> src_stats =
            train_iface.stats();
        auto policy_assignments = train_iface.policyAssignments();

        void **input_buffers = buffers;
        void **output_buffers = buffers +
            src_obs.size() + src_stats.size() + 4;

        if (policy_assignments.has_value()) {
            output_buffers += 1;
        }

        CountT cur_idx = 0;

        copyToSim(train_iface.actions(), input_buffers[cur_idx++]);
        copyToSim(train_iface.resets(), input_buffers[cur_idx++]);

        gpuExec.runAsync(strm);

        copyFromSim(output_buffers[cur_idx++], train_iface.rewards());
        copyFromSim(output_buffers[cur_idx++], train_iface.dones());

        if (policy_assignments.has_value()) {
            copyFromSim(output_buffers[cur_idx++], *policy_assignments);
        }

        for (const TrainInterface::NamedTensor &t : src_obs) {
            copyFromSim(output_buffers[cur_idx++], t.hdl);
        }

        for (const TrainInterface::NamedTensor &t : src_stats) {
            copyFromSim(output_buffers[cur_idx++], t.hdl);
        }
    }
#endif
    
    virtual inline Tensor exportTensor(ExportID slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                               EpisodeManager *episode_mgr,
                                               const GridState *grid)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            grid,
        };
    }

    return world_inits;
}

Manager::Impl * Manager::Impl::init(const Config &cfg,
                                    const GridState &src_grid)
{
    static_assert(sizeof(GridState) % alignof(Cell) == 0);

    Sim::Config sim_cfg {
        .maxEpisodeLength = cfg.maxEpisodeLength,
        .enableViewer = false,
    };

    switch (cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)malloc(sizeof(GridState) + num_cell_bytes);
        Cell *cpu_cell_data = (Cell *)(grid_data + sizeof(GridState));

        GridState *cpu_grid = (GridState *)grid_data;
        *cpu_grid = GridState {
            .cells = cpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        memcpy(cpu_cell_data, src_grid.cells, num_cell_bytes);

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, cpu_grid);

        return new CPUImpl(cfg, sim_cfg, episode_mgr, cpu_grid,
                           world_inits.data());
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        FATAL("CUDA support not compiled in!");
#else
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)cu::allocGPU(sizeof(GridState) + num_cell_bytes);

        Cell *gpu_cell_data = (Cell *)(grid_data + sizeof(GridState));
        GridState grid_staging {
            .cells = gpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        cudaMemcpy(grid_data, &grid_staging, sizeof(GridState),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_cell_data, src_grid.cells, num_cell_bytes,
                   cudaMemcpyHostToDevice);

        GridState *gpu_grid = (GridState *)grid_data;

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, gpu_grid);

        return new GPUImpl(cfg, sim_cfg, episode_mgr, gpu_grid,
                           world_inits.data());
#endif
    } break;
    default: return nullptr;
    }
}

Manager::Manager(const Config &cfg,
                 const GridState &src_grid)
    : impl_(Impl::init(cfg, src_grid))
{
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuRolloutStep(cudaStream_t strm, void **rollout_buffers)
{
    TrainInterface iface = trainInterface();
    impl_->gpuRollout(strm, rollout_buffers, iface);
}
#endif

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, Tensor::ElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::observationTensor() const
{
    return impl_->exportTensor(ExportID::GridPos, Tensor::ElementType::Int32,
        {impl_->cfg.numWorlds, 2});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, Tensor::ElementType::Float32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, Tensor::ElementType::Float32,
        {impl_->cfg.numWorlds, 1});
}

TrainInterface Manager::trainInterface() const
{
    return TrainInterface {
        {
            { "self", observationTensor() },
        },
        actionTensor(),
        rewardTensor(),
        doneTensor(),
        resetTensor(),
        Optional<Tensor>::none(),
    }; 
}

void Manager::triggerReset(int32_t world_idx)
{
    Reset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(Reset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

}
