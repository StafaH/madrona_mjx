#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madMJX {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    if (viz_gpu_hdls.has_value()) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.useRT && !viz_gpu_hdls.has_value()) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        assert(viz_gpu_hdls.has_value());

        render_api = viz_gpu_hdls->renderAPI;
        render_dev = viz_gpu_hdls->renderDev;
    }

    uint32_t max_instances_per_world = mjx_model.numGeoms;
    if (mgr_cfg.addCamDebugGeometry) {
        max_instances_per_world += mjx_model.numCams;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = true,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = mjx_model.numCams,
        .maxInstancesPerWorld = max_instances_per_world,
        .execMode = ExecMode::CUDA,
        .voxelCfg = {},
    });
}

struct JAXIO {
    Vector3 *geomPositions;
    Quat *geomRotations;
    Vector3 *camPositions;
    Quat *camRotations;

    uint8_t *rgbOut;
    float *depthOut;

    static inline JAXIO make(void **buffers)
    {
        CountT buf_idx = 0;
        auto geom_positions = (Vector3 *)buffers[buf_idx++];
        auto geom_rotations = (Quat *)buffers[buf_idx++];
        auto cam_positions = (Vector3 *)buffers[buf_idx++];
        auto cam_rotations = (Quat *)buffers[buf_idx++];
        auto rgb_out = (uint8_t *)buffers[buf_idx++];
        auto depth_out = (float *)buffers[buf_idx++];

        return JAXIO {
            .geomPositions = geom_positions,
            .geomRotations = geom_rotations,
            .camPositions = cam_positions,
            .camRotations = cam_rotations,
            .rgbOut = rgb_out,
            .depthOut = depth_out,
        };
    }
};

struct Manager::Impl {
    Config cfg;
    uint32_t numGeoms;
    uint32_t numCams;

    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph renderGraph;

    Optional<MWCudaLaunchGraph> raytraceGraph;

    static inline Impl * make(
        const Config &cfg,
        const MJXModel &mjx_model,
        const Optional<VisualizerGPUHandles> &viz_gpu_hdls);

    inline Impl(const Manager::Config &mgr_cfg,
                uint32_t num_geoms,
                uint32_t num_cams,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                MWCudaExecutor &&gpu_exec,
                Optional<MWCudaLaunchGraph> &&raytrace_graph)

        : cfg(mgr_cfg),
          numGeoms(num_geoms),
          numCams(num_cams),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          renderGraph(gpuExec.buildLaunchGraph(TaskGraphID::Render)),
          raytraceGraph(std::move(raytrace_graph))
    {}

    inline ~Impl() {}

    inline void renderImpl()
    {
        if (renderMgr.has_value()) {
            renderMgr->readECS();
            renderMgr->batchRender();
        }

        if (cfg.useRT) {
            gpuExec.run(*raytraceGraph);
        }
    }

    inline void copyInTransforms(Vector3 *geom_positions,
                                 Quat *geom_rotations,
                                 Vector3 *cam_positions,
                                 Quat *cam_rotations,
                                 cudaStream_t strm)
    {
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstancePositions),
            geom_positions,
            sizeof(Vector3) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceRotations),
            geom_rotations,
            sizeof(Quat) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);

        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraPositions),
            cam_positions,
            sizeof(Vector3) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraRotations),
            cam_rotations,
            sizeof(Quat) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
    }

    inline void init(Vector3 *geom_positions,
                             Quat *geom_rotations,
                             Vector3 *cam_positions,
                             Quat *cam_rotations)
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);

        gpuExec.run(init_graph);

        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations, 0);

        gpuExec.run(renderGraph);
        renderImpl();
    }

    inline void render(Vector3 *geom_positions,
                               Quat *geom_rotations,
                               Vector3 *cam_positions,
                               Quat *cam_rotations)
    {
        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations, 0);

        gpuExec.run(renderGraph);

        renderImpl();
    }

    inline const float * getDepthOut() const
    {
        if (cfg.useRT) {
            return (float *)gpuExec.getExported((uint32_t)ExportID::RaycastDepth);
        } else {
            return renderMgr->batchRendererDepthOut();
        }
    }

    inline const uint8_t * getRGBOut() const
    {
        if (cfg.useRT) {
            return (uint8_t *)gpuExec.getExported((uint32_t)ExportID::RaycastRGB);
        } else {
            return renderMgr->batchRendererRGBOut();
        }
    }

    inline void copyOutRendered(uint8_t *rgb_out, float *depth_out,
                                cudaStream_t strm)
    {  
        cudaMemcpyAsync(depth_out, getDepthOut(),
                        sizeof(float) *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams,
                        cudaMemcpyDeviceToDevice, strm);

        cudaMemcpyAsync(rgb_out, getRGBOut(),
                        sizeof(uint8_t) *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams * 4,
                        cudaMemcpyDeviceToDevice, strm);
    }

    inline void gpuStreamInit(cudaStream_t strm, void **buffers)
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);

        JAXIO jax_io = JAXIO::make(buffers);

        gpuExec.runAsync(init_graph, strm);

        copyInTransforms(jax_io.geomPositions, jax_io.geomRotations,
                         jax_io.camPositions, jax_io.camRotations, strm);

        gpuExec.runAsync(renderGraph, strm);

        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));

        renderImpl();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
    }

    inline void gpuStreamRender(cudaStream_t strm, void **buffers)
    {
        JAXIO jax_io = JAXIO::make(buffers);

        copyInTransforms(jax_io.geomPositions, jax_io.geomRotations,
                         jax_io.camPositions, jax_io.camRotations, strm);

#if 0
        Vector3 *readback_pos = (Vector3 *)malloc(sizeof(Vector3) * numGeoms * cfg.numWorlds);
        cudaMemcpy(jax_io.geomPositions,
                   readback_pos,
                   sizeof(Vector3) * numGeoms * cfg.numWorlds,
                   cudaMemcpyHostToDevice);
        printf("%f %f %f\n",
            readback_pos[1].x,
            readback_pos[1].y,
            readback_pos[1].z);
#endif

        gpuExec.runAsync(renderGraph, strm);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));

        renderImpl();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
    }

    inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};

struct RTAssets {
    render::MeshBVHData bvhData;
    render::MaterialData matData;
};

static RTAssets loadRenderObjects(
    const MJXModel &model,
    Optional<render::RenderManager> &render_mgr,
    bool use_rt)
{
    using namespace imp;

    std::array<std::string, (size_t)RenderPrimObjectIDs::NumPrims> 
        render_asset_paths;
    render_asset_paths[(size_t)RenderPrimObjectIDs::DebugCam] =
        (std::filesystem::path(DATA_DIR) / "debugcam.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Box] =
        (std::filesystem::path(DATA_DIR) / "box.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Cylinder] =
        (std::filesystem::path(DATA_DIR) / "cylinder.obj").string();

    std::array<const char *, render_asset_paths.size()> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    imp::AssetImporter asset_importer;

    std::array<char, 1024> import_err;
    auto disk_render_assets = asset_importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!disk_render_assets.has_value()) {
        FATAL("Failed to load render assets from disk: %s", import_err);
    }

    HeapArray<SourceMesh> meshes(
        model.meshGeo.numMeshes + disk_render_assets->objects.size());
    const CountT num_meshes = (CountT)model.meshGeo.numMeshes;

    for (CountT i = 0; i < disk_render_assets->objects.size(); i++) {
        meshes[i] = disk_render_assets->objects[i].meshes[0];
    }
    
    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        uint32_t mesh_vert_offset = model.meshGeo.vertexOffsets[mesh_idx];
        uint32_t next_vert_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.vertexOffsets[mesh_idx + 1] : model.meshGeo.numVertices;

        uint32_t mesh_tri_offset = model.meshGeo.triOffsets[mesh_idx];
        uint32_t next_tri_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.triOffsets[mesh_idx + 1] : model.meshGeo.numTris;

        uint32_t mesh_num_verts = next_vert_offset - mesh_vert_offset;
        uint32_t mesh_num_tris = next_tri_offset - mesh_tri_offset;
        uint32_t mesh_idx_offset = mesh_tri_offset * 3;

        meshes[mesh_idx + disk_render_assets->objects.size()] = {
            .positions = model.meshGeo.vertices + mesh_vert_offset,
            .normals = nullptr,
            .tangentAndSigns = nullptr,
            .uvs = nullptr,
            .indices = model.meshGeo.indices + mesh_idx_offset,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = mesh_num_verts,
            .numFaces = mesh_num_tris,
            .materialIDX = 0,
        };
    }

    std::vector<imp::SourceMaterial> materials;
    for (CountT i = 0; i < model.numMats; i++) {
        SourceMaterial mat = {
            .color = math::Vector4{
                model.matRGBA[i].x, model.matRGBA[i].y,
                model.matRGBA[i].z, model.matRGBA[i].w},
            .textureIdx = -1,
            .roughness = 0.0f,
            .metalness = 0.0f};
        materials.push_back(mat);
    }

    // Create materials for geoms that do not have one assigned
    for (CountT i = 0; i < model.numGeoms; i++) {
        if (model.geomMatIDs[i] == -1) {
            SourceMaterial mat = {
                .color = math::Vector4{
                    model.geomRGBA[i].x, model.geomRGBA[i].y,
                    model.geomRGBA[i].z, model.geomRGBA[i].w},
                .textureIdx = -1,
                .roughness = 0.8f,
                .metalness = 0.2f,
            };
            materials.push_back(mat);
            model.geomMatIDs[i] = materials.size() - 1;

            for (CountT j = i + 1; j < model.numGeoms; j++) {
                // FIX: Should probably implement == op for Vector4
                if (model.geomMatIDs[j] == -1 && 
                    model.geomRGBA[i].x == model.geomRGBA[j].x &&
                    model.geomRGBA[i].y == model.geomRGBA[j].y &&
                    model.geomRGBA[i].z == model.geomRGBA[j].z &&
                    model.geomRGBA[i].w == model.geomRGBA[j].w) 
                {
                    model.geomMatIDs[j] = materials.size() - 1;
                }
            }
        }
    }

    HeapArray<SourceObject> objs(model.numGeoms + 1);

    // Create a new mesh for each geom to avoid geoms that share the same mesh
    // from pointing to the same source mesh
    HeapArray<SourceMesh> dest_meshes(model.numGeoms + 1);
    
    for (CountT i = 0; i < model.numGeoms; i++) {
        int source_mesh_idx = 0;
        switch ((MJXGeomType)model.geomTypes[i]) {
        case MJXGeomType::Plane: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Plane;
        } break;
        case MJXGeomType::Sphere: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Sphere;
        } break;
        case MJXGeomType::Capsule: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Sphere;
        } break;
        case MJXGeomType::Box: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Box;
        } break;
        case MJXGeomType::Cylinder: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Cylinder;
        } break;
        case MJXGeomType::Mesh: {
            source_mesh_idx = (int)RenderPrimObjectIDs::NumPrims + model.geomDataIDs[i];
        } break;
        case MJXGeomType::Heightfield:
        case MJXGeomType::Ellipsoid:
        default:
            FATAL("Unsupported geom type");
            break;
        }
        const SourceMesh& source_mesh = meshes[source_mesh_idx];
        dest_meshes[i] = {
            .positions = source_mesh.positions,
            .normals = source_mesh.normals,
            .tangentAndSigns = source_mesh.tangentAndSigns,
            .uvs = source_mesh.uvs,
            .indices = source_mesh.indices,
            .faceCounts = source_mesh.faceCounts,
            .faceMaterials = source_mesh.faceMaterials,
            .numVertices = source_mesh.numVertices,
            .numFaces = source_mesh.numFaces,
            .materialIDX = static_cast<uint32_t>(model.geomMatIDs[i]),
        };

        objs[i] = {
            .meshes = Span<SourceMesh>(&dest_meshes[i], 1),
        };

        model.geomDataIDs[i] = -1;
        for (CountT geom_i = 0; geom_i < model.numEnabledGeomGroups; geom_i++)
        {
            if (model.geomGroups[i] == model.enabledGeomGroups[geom_i])
            {
                model.geomDataIDs[i] = i;
                break;
            }
        }
    }
    
    objs[model.numGeoms] = disk_render_assets->objects[(int)RenderPrimObjectIDs::DebugCam];
    for (auto &mesh : objs[model.numGeoms].meshes) {
        mesh.materialIDX = 0;
    }

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(objs, materials, {});

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    if (use_rt) {
        return {
            render::AssetProcessor::makeBVHData(objs),
            render::AssetProcessor::initMaterialData(materials.data(),
                                     materials.size(),
                                     nullptr,
                                     0)
        };
    } else {
        return {};
    }
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    bool use_rt = mgr_cfg.useRT;

    if (use_rt) {
        printf("Using raytracer\n");
    } else {
        printf("Using rasterizer\n");
    }

    Sim::Config sim_cfg;
    sim_cfg.numGeoms = mjx_model.numGeoms;
    sim_cfg.numCams = mjx_model.numCams;
    sim_cfg.useDebugCamEntity = mgr_cfg.addCamDebugGeometry;
    sim_cfg.useRT = use_rt;

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

    Optional<RenderGPUState> render_gpu_state =
        initRenderGPUState(mgr_cfg, viz_gpu_hdls);

    Optional<render::RenderManager> render_mgr =
        initRenderManager(mgr_cfg, mjx_model,
                          viz_gpu_hdls, render_gpu_state);

    RTAssets rt_assets = loadRenderObjects(
            mjx_model, render_mgr, use_rt);
    if (render_mgr.has_value()) {
        sim_cfg.renderBridge = render_mgr->bridge();
    } else {
        sim_cfg.renderBridge = nullptr;
    }

    int32_t *geom_types_gpu = (int32_t *)cu::allocGPU(
        sizeof(int32_t) * mjx_model.numGeoms);
    int32_t *geom_data_ids_gpu = (int32_t *)cu::allocGPU(
        sizeof(int32_t) * mjx_model.numGeoms);
    Vector3 *geom_sizes_gpu = (Vector3 *)cu::allocGPU(
        sizeof(Vector3) * mjx_model.numGeoms);

    REQ_CUDA(cudaMemcpy(geom_types_gpu, mjx_model.geomTypes,
        sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_data_ids_gpu, mjx_model.geomDataIDs,
        sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_sizes_gpu, mjx_model.geomSizes,
        sizeof(Vector3) * mjx_model.numGeoms, cudaMemcpyHostToDevice));

    sim_cfg.geomTypes = geom_types_gpu;
    sim_cfg.geomDataIDs = geom_data_ids_gpu;
    sim_cfg.geomSizes = geom_sizes_gpu;

    HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

    Optional<CudaBatchRenderConfig> render_cfg = 
        Optional<CudaBatchRenderConfig>::none();
    if (use_rt) {
        render_cfg = {
            .renderMode = CudaBatchRenderConfig::RenderMode::RGBD,
            .geoBVHData = rt_assets.bvhData,
            .materialData = rt_assets.matData,
            .renderResolution = mgr_cfg.batchRenderViewWidth,
            .nearPlane = 0.001f,
            .farPlane = 1000.0f,
        };
    }

    MWCudaExecutor gpu_exec({
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(Sim::WorldInit),
        .userConfigPtr = (void *)&sim_cfg,
        .numUserConfigBytes = sizeof(Sim::Config),
        .numWorldDataBytes = sizeof(Sim),
        .worldDataAlignment = alignof(Sim),
        .numWorlds = mgr_cfg.numWorlds,
        .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
        .numExportedBuffers = (uint32_t)ExportID::NumExports, 
    }, {
        { GPU_HIDESEEK_SRC_LIST },
        { GPU_HIDESEEK_COMPILE_FLAGS },
        CompileConfig::OptMode::LTO,
    }, cu_ctx, render_cfg);

    Optional<MWCudaLaunchGraph> raytrace_graph =
        Optional<MWCudaLaunchGraph>::none();

    if (use_rt) {
        raytrace_graph = gpu_exec.buildRenderGraph();
    }

    cu::deallocGPU(geom_types_gpu);
    cu::deallocGPU(geom_data_ids_gpu);
    cu::deallocGPU(geom_sizes_gpu);

    return new Impl {
        mgr_cfg,
        mjx_model.numGeoms,
        mjx_model.numCams,
        std::move(render_gpu_state),
        std::move(render_mgr),
        std::move(gpu_exec),
        std::move(raytrace_graph)
    };
}

Manager::Manager(const Config &cfg,
                 const MJXModel &mjx_model,
                 Optional<VisualizerGPUHandles> viz_gpu_hdls)
    : impl_(Impl::make(cfg, mjx_model, viz_gpu_hdls))
{}

Manager::~Manager() {}

void Manager::init(math::Vector3 *geom_pos, math::Quat *geom_rot,
                   math::Vector3 *cam_pos, math::Quat *cam_rot)
{
    impl_->init(geom_pos, geom_rot, cam_pos, cam_rot);
}

void Manager::render(math::Vector3 *geom_pos, math::Quat *geom_rot,
                     math::Vector3 *cam_pos, math::Quat *cam_rot)
{
    impl_->render(geom_pos, geom_rot, cam_pos, cam_rot);
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers);
}

void Manager::gpuStreamRender(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamRender(strm, buffers);
}
#endif

Tensor Manager::instancePositionsTensor() const
{
    return impl_->exportTensor(ExportID::InstancePositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::instanceRotationsTensor() const
{
    return impl_->exportTensor(ExportID::InstanceRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::cameraPositionsTensor() const
{
    return impl_->exportTensor(ExportID::CameraPositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::cameraRotationsTensor() const
{
    return impl_->exportTensor(ExportID::CameraRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::rgbTensor() const
{
    FATAL("No RGB support currently");
#if 0
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
#endif
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->getDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

uint32_t Manager::numWorlds() const
{
    return impl_->cfg.numWorlds;
}

uint32_t Manager::numCams() const
{
    return impl_->numCams;
}

uint32_t Manager::batchViewWidth() const
{
    return impl_->cfg.batchRenderViewWidth;
}

uint32_t Manager::batchViewHeight() const
{
    return impl_->cfg.batchRenderViewHeight;
}

render::RenderManager & Manager::getRenderManager()
{
    return *(impl_->renderMgr);
}

}
