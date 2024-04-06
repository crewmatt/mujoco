#ifndef THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_EXPERIMENTAL_H_
#define THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_EXPERIMENTAL_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlex/proxy/simulation/utils/file_bundle/file_bundle.proto.h"
#include "third_party/OpenCV/videoio.hpp"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/time/time.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/mujoco/src/usd/circus.h"
#include "third_party/mujoco/src/usd/circus_util.h"
#include "third_party/mujoco/src/xml/xml_urdf.h"
#include "third_party/usd/v23/google/plugin_registration.h"
#include "third_party/usd/v23/pxr/base/gf/matrix4d.h"
#include "third_party/usd/v23/pxr/base/tf/token.h"
#include "third_party/usd/v23/pxr/base/tf/weakPtr.h"
#include "third_party/usd/v23/pxr/imaging/hd/driver.h"
#include "third_party/usd/v23/pxr/imaging/hd/primTypeIndex.h"
#include "third_party/usd/v23/pxr/imaging/hd/renderIndex.h"
#include "third_party/usd/v23/pxr/imaging/hgi/hgi.h"
#include "third_party/usd/v23/pxr/usd/sdf/path.h"
#include "third_party/usd/v23/pxr/usd/usd/common.h"
#include "third_party/usd/v23/pxr/usd/usd/prim.h"
#include "third_party/usd/v23/pxr/usd/usd/references.h"
#include "third_party/usd/v23/pxr/usd/usd/stage.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/camera.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/capsule.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cone.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cube.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cylinder.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/mesh.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/metrics.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/sphere.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/tokens.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xform.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xformCache.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xformOp.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/joint.h"
#include "third_party/usd/v23/pxr/usdImaging/usdImaging/delegate.h"
#include "third_party/usd/v23/pxr/usdImaging/usdImagingGL/engine.h"
#include "util/time/clock.h"

namespace mujoco {

struct mjImageOutputBuffer {
  size_t width = 0;
  size_t height = 0;
  pxr::HdFormat format = pxr::HdFormatInvalid;
  std::string aovId;
  std::string type;
  std::vector<uint8_t> buffer;
};

class CircusExperimental : public Circus {
 public:
  // Creates CircusExperimental for running a simulation.
  //
  // Args:
  //   circus_layers: A list of layers to initialize the simulation with
  //   start_time: The time the simulation clock will be initialized to
  //
  // Returns:
  //   A Status or unique pointer to the circus.
  static absl::StatusOr<std::unique_ptr<CircusExperimental>> Create(
      const std::vector<mjCircusLayer>& circus_layers,
      absl::Time start_time);

  static absl::StatusOr<std::unique_ptr<CircusExperimental>> Create2(
      const std::vector<mjCircusLayer>& circus_layers) {
    return Create(circus_layers, absl::UnixEpoch());
  };

  static CircusExperimental* Create3(
      const mjCircusLayer circus_layer) {
    return  Create({circus_layer}, absl::UnixEpoch()).value().release();
  };

  // Renders a given camera.
  absl::StatusOr<std::vector<mjImageOutputBuffer>> RenderCamera(
      const std::string& scene_path);

  // Advances time of the simulation.
  absl::Status Step();

  bool Step2() {
    auto step_status = Step();
    if (!step_status.ok()) {
      std::cerr << "MATTT Step2 failed: " << step_status;
      return false;
    }
    return true;
  }

  pxr::UsdStageRefPtr GetWorldStage() {
    return world_stage_;
  }

  ~CircusExperimental() {}

 private:
  explicit CircusExperimental(const pxr::UsdStageRefPtr& world_stage)
      : Circus(world_stage) {}

  // Simulated Clock to track time in the simulation.
  util::SimulatedClock simulation_clock_;

  // Offscreen Context to launch and configure Mesa Renderer.
  /* std::unique_ptr<
      googlex::proxy::simulation::perception_renderer::OffscreenContext>
      offscreen_context_; */

  // VideoWriters to output video if desired.
  std::map<std::string, std::unique_ptr<cv::VideoWriter>> video_writers_map_;
};

}  // namespace mujoco

#endif  // THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_EXPERIMENTAL_H_
