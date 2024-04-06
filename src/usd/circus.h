#ifndef THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_H_
#define THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_H_

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
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/mujoco/src/usd/circus_util.h"
#include "third_party/mujoco/src/usd/importers/urdf/urdf_sdf_file_format.h"
#include "third_party/mujoco/src/user/user_model.h"
#include "third_party/mujoco/src/xml/xml_urdf.h"
#include "third_party/usd/v23/google/plugin_registration.h"
#include "third_party/usd/v23/pxr/base/gf/matrix4d.h"
#include "third_party/usd/v23/pxr/base/tf/token.h"
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

class Circus {
 public:
  // Creates Circus for running a simulation.
  //
  // Args:
  //   circus_layers: A list of layers to initialize the simulation with
  //   start_time: The time the simulation clock will be initialized to
  //
  // Returns:
  //   A Status or unique pointer to the circus.
  static absl::StatusOr<std::unique_ptr<Circus>> Create(
      const std::vector<mjCircusLayer>& circus_layers);

  // Add layers to the existing circus
  //
  // Args:
  //   circus_layers: A vector of layers to add to the existing Circus.
  absl::Status AddLayers(const std::vector<mjCircusLayer>& layers);

  // Set pose relative to parent frame of the object.
  //
  // Args:
  //   scene_path: A string scene path of the object.
  //   parent_pose_object: The pose relative to the parent frame.
  absl::Status SetLocalPose(const std::string& scene_path_of_object,
                            const mjCircusPose& parent_pose_object);

  // Set pose relative to the world.
  //
  // Args:
  //   scene_path: A string scene path of the object.
  //   world_pose_object: The pose relative to the world frame.
  absl::Status SetWorldPose(const std::string& scene_path_of_object,
                            const mjCircusPose& world_pose_object);

  // Gets the pose in parent frame of a given object.
  //
  // Args:
  //   scene_path: A string scene path of the object.
  //
  // Returns:
  //   The pose of the object relative to its parent frame.
  absl::StatusOr<mjCircusPose> GetLocalPose(
      const std::string& scene_path_of_object);

  // Gets the pose in world frame of a given object.
  //
  // Args:
  //   scene_path: A string scene path of the object.
  //
  // Returns:
  //   The pose of the object relative to its world frame.
  absl::StatusOr<mjCircusPose> GetWorldPose(
      const std::string& scene_path_of_object);

  // Creates an mjModel and mjData for the physical objects in the list of USD
  // layers. Caller assumes ownership of the raw pointer.
  absl::StatusOr<mjModel*> CreateMjModel();

  // Exports an MJCF of the current MuJoCo model.
  absl::StatusOr<googlex::proxy::simulation::FileBundle> ExportMjcf();

  // Exports a USD of the current world_state_.
  absl::StatusOr<std::string> ExportUsd();

  ~Circus() {
    mj_deleteData(mjdata_);
    mj_deleteModel(mjmodel_);
    world_stage_.Reset();
  }

  // This is meant to be private, only protected for CircusExperimental.
 protected:
  explicit Circus(const pxr::UsdStageRefPtr& world_stage)
      : world_stage_(world_stage) {}

  // Adds a USD Layer to the existing Circus.
  absl::Status AddUsdLayer(const mjCircusLayer& layer);

  // Adds an MJCF Layer to the existing Circus. For now this only adds a single
  // MJCF file, and it is not available to be rendered via USD. Longer term this
  // will be corrected to leverage an MJCF->USD converter enabling multiple MJCF
  // files and full rendering.
  absl::Status AddMjcfLayer(const mjCircusLayer& layer);

  // Gets the USD Stage pose of a given prim relative to its parent.
  absl::StatusOr<pxr::GfMatrix4d> GetUsdLocalPose(
      const pxr::SdfPath& prim_path);

  // Gets the USD Stage pose of a given prim relative to the World.
  absl::StatusOr<pxr::GfMatrix4d> GetUsdWorldPose(
      const pxr::SdfPath& prim_path);

  // Initializes a MuJoCo Capsule Geom given a prim.
  absl::Status InitializeMujocoGeomCapsule(
      const pxr::UsdGeomCapsule& usd_geom_capsule, mjsBody* body,
      mjsGeom* geom);

  // Initializes a MuJoCo Cone Geom given a prim.
  absl::Status InitializeMujocoGeomCone(const pxr::UsdGeomCone& usd_geom_cone,
                                     mjsBody* body, mjsGeom* geom);

  // Initializes a MuJoCo Cube Geom given a prim.
  absl::Status InitializeMujocoGeomCube(const pxr::UsdGeomCube& usd_geom_cube,
                                     mjsBody* body, mjsGeom* geom);

  // Initializes a MuJoCo Capsule Geom given a prim.
  absl::Status InitializeMujocoGeomCylinder(
      const pxr::UsdGeomCylinder& usd_geom_cylinder, mjsBody* body,
      mjsGeom* geom);

  // Initializes a MuJoCo Mesh Geom given a prim. Note this also adds a mesh
  // to the mjcmodel_.
  absl::Status InitializeMujocoGeomMesh(
      const pxr::UsdGeomMesh& usd_geom_mesh, mjsBody* body, mjsGeom* geom);

  // Initializes a MuJoCo Capsule Geom given a prim.
  absl::Status InitializeMujocoGeomSphere(
      const pxr::UsdGeomSphere& usd_geom_sphere, mjsBody* body, mjsGeom* geom);

  // Initializes a MuJoCo Body and its child Geoms given a USD prim.
  //
  // Args:
  //   child_body_prim: The prim of the child body to be created.
  //   usd_world_pose_parent_body: The pose of the parent body in world frame.
  //   parent_body: The body the new body should be added to. This can be the
  //       World Body if needed.
  //   child_body:  Output the new MuJoCo child body which is created.
  //   usd_world_pose_child_body: Output the absolute pose of the child body
  //       that is created.
  //   body_mass: Output the mass of the body that is created.
  absl::Status InitializeMujocoBody(
    const pxr::UsdPrim& child_body_prim,
    const pxr::GfMatrix4d& usd_world_pose_parent_body,
    mjsBody* parent_body,
    mjsBody** child_body,
    pxr::GfMatrix4d* usd_world_pose_child_body,
    double* body_mass);

  // Initializes a MuJoCo Joint given a prim.
  absl::Status InitializeMujocoJoint(
      const pxr::UsdPhysicsJoint& usd_physics_joint, mjsBody* parent_body,
      mjsBody* child_body);

  // Recursively initializes a full kinematic chain of a Multi-Body by following
  // the joints.
  absl::Status InitializeKinematicChain(
      const pxr::SdfPath& sdf_path,
      const std::map<pxr::SdfPath, std::vector<pxr::SdfPath>>& all_nodes_map,
      const pxr::GfMatrix4d& usd_world_pose_parent_body,
      mjsBody* parent_body);

  // Copies the poses of all bodies being physically simulated back to the USD
  // circus.
  absl::Status SyncMujocoToUsdStage();

  // Uses the mjdata_ information to update the keyframe in the mjmodel_.
  absl::Status UpdateMujocoModelKeyframe();

  // Takes the USD Circus and initializes a MuJoCo simulation of it.
  absl::Status InitializeMujocoFromWorldStage();

  // The singular world USD circus that contains all relevant world state.
  pxr::UsdStageRefPtr world_stage_;

  // MuJoCo model and data for physics simulation.
  mjModel* mjmodel_ = nullptr;
  mjData* mjdata_ = nullptr;

  // Layers that are loaded onto this circus.
  std::map<std::string, mjCircusLayer> circus_layer_map_;

  // Single MJCF File to be merged into the simulation.
  std::string mjcf_file_to_merge_;

  // mjCModel used to compile the simulation.
  std::unique_ptr<mjCModel> mjcmodel_;

  // The keyframe id to store the current state of the simulation when
  // exporting.
  size_t mjcmodel_keyframe_id_;
};

}  // namespace mujoco

#endif  // THIRD_PARTY_MUJOCO_SRC_USD_CIRCUS_H_
