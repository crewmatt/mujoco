#include "third_party/mujoco/src/usd/circus_experimental.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"
// #include "googlex/proxy/simulation/perception_renderer/threejs/offscreen_context.h"
#include "googlex/proxy/simulation/scene_builder/bundles/asset_parsers/reference_collector.h"
#include "googlex/proxy/simulation/utils/file_bundle/file_bundle.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/time/time.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/mujoco/src/usd/circus_util.h"
#include "third_party/mujoco/src/user/user_api.h"
#include "third_party/mujoco/src/xml/xml.h"
#include "third_party/mujoco/src/xml/xml_native_reader.h"
#include "third_party/tinyxml2/tinyxml2.h"
#include "third_party/usd/v23/google/plugin_registration.h"
#include "third_party/usd/v23/google/imaging/engine.h"
#include "third_party/usd/v23/pxr/base/gf/matrix4d.h"
#include "third_party/usd/v23/pxr/base/gf/quatd.h"
#include "third_party/usd/v23/pxr/base/gf/rotation.h"
#include "third_party/usd/v23/pxr/base/gf/vec3d.h"
#include "third_party/usd/v23/pxr/base/gf/vec3f.h"
#include "third_party/usd/v23/pxr/base/tf/token.h"
#include "third_party/usd/v23/pxr/base/vt/array.h"
#include "third_party/usd/v23/pxr/imaging/hd/renderBuffer.h"
#include "third_party/usd/v23/pxr/imaging/hd/renderDelegate.h"
#include "third_party/usd/v23/pxr/imaging/hd/rendererPlugin.h"
#include "third_party/usd/v23/pxr/imaging/hd/rendererPluginRegistry.h"
#include "third_party/usd/v23/pxr/imaging/hd/tokens.h"
#include "third_party/usd/v23/pxr/imaging/hd/types.h"
#include "third_party/usd/v23/pxr/usd/sdf/path.h"
#include "third_party/usd/v23/pxr/usd/usd/common.h"
#include "third_party/usd/v23/pxr/usd/usd/primRange.h"
#include "third_party/usd/v23/pxr/usd/usd/stage.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/capsule.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cone.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cube.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cylinder.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/mesh.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/metrics.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/points.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/sphere.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/tokens.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xform.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xformCache.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xformable.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/collisionAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/joint.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/limitAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/massAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/rigidBodyAPI.h"
#include "third_party/usd/v23/pxr/usd/usdRender/settings.h"
#include "third_party/usd/v23/pxr/usdImaging/usdImagingGL/engine.h"
#include "third_party/usd/v23/pxr/usdImaging/usdImagingGL/renderParams.h"
#include "util/task/contrib/status_macros/ret_check.h"
#include "util/task/status_macros.h"
#include "util/tuple/components/dump_vars.h"

namespace mujoco {

namespace {

absl::StatusOr<pxr::HdRenderSettingsMap> ReadRenderSettings(
    pxr::UsdStageRefPtr world_stage) {
  pxr::HdRenderSettingsMap render_settings;
  for (const auto& prim : world_stage->TraverseAll()) {
    if (prim.GetTypeName() == "RenderSettings") {
      pxr::UsdRenderSettings settings =
          pxr::UsdRenderSettings::Get(world_stage, prim.GetPrimPath());

      // convert attributes to Render Settings Map
      std::vector<pxr::UsdAttribute> attributes =
          settings.GetPrim().GetAuthoredAttributes();
      for (const auto& a : attributes) {
        pxr::VtValue value;
        a.Get(&value);
        render_settings[a.GetName()] = value;
      }

      // camera rel
      pxr::UsdRelationship cam_rel = settings.GetCameraRel();
      if (cam_rel) {
        pxr::SdfPathVector targets;
        cam_rel.GetTargets(&targets);
        if (!targets.empty()) {
          std::string cam_rel_path = targets[0].GetString();
          render_settings[pxr::HdTokens->camera] = cam_rel_path;
        }
      }
    }
  }

  return render_settings;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CircusExperimental>> CircusExperimental::Create(
    const std::vector<mjCircusLayer>& circus_layers, absl::Time start_time) {

  RETURN_IF_ERROR(pxr::google::RegisterUsdPlugins());

  // Create an empty world stage.
  pxr::UsdStageRefPtr world_stage = pxr::UsdStage::CreateInMemory();
  RET_CHECK(world_stage != nullptr) << "Failed to create world UsdStage.";

  // Orient the Z access as up.
  RET_CHECK(pxr::UsdGeomSetStageUpAxis(world_stage, pxr::UsdGeomTokens->z))
      << "Failed to set stage up axis.";

  auto circus =
      absl::WrapUnique<CircusExperimental>(new CircusExperimental(world_stage));

  // Configure Offscreen Context.
  /* circus->offscreen_context_ = googlex::proxy::simulation::
      perception_renderer ::OffscreenContext::Create();
  RETURN_IF_ERROR(circus->offscreen_context_->MakeCurrent()); */

  circus->simulation_clock_.SetTime(start_time);
  RETURN_IF_ERROR(circus->AddLayers(circus_layers));

  return circus;
}

absl::StatusOr<std::vector<mjImageOutputBuffer>>
CircusExperimental::RenderCamera(const std::string& scene_path) {
  // Try this
  pxr::HdRendererPluginRegistry& plugin_registry =
      pxr::HdRendererPluginRegistry::GetInstance();
  auto* hd_renderer_plugin =
      plugin_registry.GetRendererPlugin(pxr::TfToken{"HdCyclesRendererPlugin"});

  google::UsdImagingGoogleEngine usd_imaging_google_engine;
  usd_imaging_google_engine.SetUsdStage(world_stage_);

  ASSIGN_OR_RETURN(pxr::HdRenderSettingsMap render_settings_map,
                   ReadRenderSettings(world_stage_));

  usd_imaging_google_engine.CreateDelegates(*hd_renderer_plugin,
                                            render_settings_map);

  auto camera_usd_path = pxr::SdfPath(scene_path);
  if (!usd_imaging_google_engine.SetCamera(camera_usd_path)) {
    return absl::InvalidArgumentError(
        "Unable to set camera to: " + camera_usd_path.GetAsString() +
        " available prims: " + PrintAllRealPrims(world_stage_));
  }

  usd_imaging_google_engine.SetResolution(640, 480);
  usd_imaging_google_engine.SetRefineLevelFallback(4);
  usd_imaging_google_engine.SetTime(1);

  usd_imaging_google_engine.Render();

  const size_t render_buffer_count =
      usd_imaging_google_engine.GetRenderBufferCount();
  if (render_buffer_count == 0) {
    return absl::InternalError("Engine has no output renderbuffers");
  }

  // Copy render buffers to output buffers.
  std::vector<mjImageOutputBuffer> results;
  for (size_t i = 0; i < render_buffer_count; i++) {
    results.emplace_back();
    auto output_buffer = results.rbegin();
    pxr::HdRenderBuffer* render_buffer =
        usd_imaging_google_engine.GetRenderBuffer(i);
    output_buffer->width = render_buffer->GetWidth();
    output_buffer->height = render_buffer->GetHeight();
    output_buffer->format = render_buffer->GetFormat();
    auto& aovId = usd_imaging_google_engine.GetAovId(i);
    output_buffer->aovId = aovId.channel_name;
    output_buffer->type = aovId.render_var.dataType.GetString();
    const size_t format_byte_size = HdDataSizeOfFormat(output_buffer->format);
    const size_t data_byte_size =
        output_buffer->width * output_buffer->height * format_byte_size;
    output_buffer->buffer.resize(data_byte_size);

    void* data = render_buffer->Map();
    memcpy(output_buffer->buffer.data(), data, data_byte_size);
    render_buffer->Unmap();
  }

  return results;
}

absl::Status CircusExperimental::Step() {
  // For now only consider time advanced by MuJoCo. No advancing of the stage
  // as far as animations go.
  if (mjmodel_ == nullptr) {
    return absl::OkStatus();
  }

  mj_step(mjmodel_, mjdata_);
  RETURN_IF_ERROR(SyncMujocoToUsdStage());
  simulation_clock_.AdvanceTime(absl::Seconds(mjmodel_->opt.timestep));

  return absl::OkStatus();
}

}  // namespace mujoco
