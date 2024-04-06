#ifndef THIRD_PARTY_MUJOCO_SRC_USD_IMPORTERS_MJCF_MUJOCO_TO_USD_H_
#define THIRD_PARTY_MUJOCO_SRC_USD_IMPORTERS_MJCF_MUJOCO_TO_USD_H_

#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/usd/v23/pxr/base/tf/staticTokens.h"
#include "third_party/usd/v23/pxr/pxr.h"
#include "third_party/usd/v23/pxr/usd/sdf/declareHandles.h"
#include "third_party/usd/v23/pxr/usd/sdf/fileFormat.h"
#include "third_party/usd/v23/pxr/usd/sdf/path.h"
#include "third_party/usd/v23/pxr/usd/usd/api.h"
#include "third_party/usd/v23/pxr/usd/usd/common.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xform.h"

namespace mujoco {

class MujocoToUsd {
 public:
  // Given an mjModel and mjData, converts it to a USD Layer. This can then be
  // used directly in a stage or exported to a file.
  //
  // Args:
  //   model: an mjModel of the simulation
  //   data: an mjData representing the current state of the simulation
  //   output_layer: an output SdfLayer that will be appended to.
  static absl::Status AddToSdfLayer(const mjModel* model, const mjData* data,
                                                pxr::SdfLayer* output_layer);

  // Given an mjModel and mjData, converts it to a USD Stage. This can then be
  // used directly in a stage or exported to a file.
  //
  // Args:
  //   model: an mjModel of the simulation
  //   data: an mjData representing the current state of the simulation
  //   output_stage: an output SdfLayer that will be appended to.
  static absl::Status AddToUsdStage(
      const mjModel* model, const mjData* data, const pxr::SdfPath& path,
      pxr::UsdStageRefPtr* output_stage);

 private:
  MujocoToUsd(const mjModel* mjmodel, const mjData* mjdata,
              pxr::UsdStageRefPtr stage) {
    mjmodel_ = mjmodel;
    mjdata_ = mjdata;
    stage_ = stage;
  }

  std::string GetUsdName(
      const size_t name_index, const std::string& type,
      const size_t type_index);

  absl::Status InitializeUsdGeomXformable(
      const size_t mujoco_body_index, const size_t mujoco_geom_index,
      const pxr::SdfPath& geom_sdf_path,
      pxr::UsdGeomXformable* usd_geom_xformable);

  absl::Status InitializeUsdGeomMesh(const size_t mujoco_body_index,
                                     const size_t mujoco_geom_index,
                                     const pxr::SdfPath& geom_sdf_path);

  absl::Status InitializeUsdGeomCapsule(const size_t mujoco_body_index,
                                        const size_t mujoco_geom_index,
                                        const pxr::SdfPath& geom_sdf_path);

  absl::Status InitializeUsdGeomCube(const size_t mujoco_body_index,
                                     const size_t mujoco_geom_index,
                                     const pxr::SdfPath& geom_sdf_path);

  absl::Status InitializeUsdGeomCylinder(const size_t mujoco_body_index,
                                         const size_t mujoco_geom_index,
                                         const pxr::SdfPath& geom_sdf_path);

  absl::Status InitializeUsdGeomSphere(const size_t mujoco_body_index,
                                       const size_t mujoco_geom_index,
                                       const pxr::SdfPath& geom_sdf_path);

  absl::Status InitializeUsdGeoms(const size_t mujoco_body_index,
                                  const pxr::SdfPath& body_sdf_path);

  absl::Status InitializeUsdPhysicsJoint(
      const size_t mujoco_joint_index,
      const pxr::SdfPath& joint_sdf_path,
      const pxr::SdfPath& parent_body_sdf_path,
      const pxr::SdfPath& child_body_sdf_path);

  absl::Status InitializeUsdPhysicsPrismaticJoint(
      const size_t mujoco_joint_index,
      const pxr::SdfPath& joint_sdf_path);

  absl::Status InitializeUsdPhysicsRevoluteJoint(
      const size_t mujoco_joint_index,
      const pxr::SdfPath& joint_sdf_path);

  absl::Status InitializeUsdPhysicsSphericalJoint(
      const size_t mujoco_joint_index,
      const pxr::SdfPath& joint_sdf_path);

  absl::Status InitializeUsdJoints(
      const size_t mujoco_body_index,
      const pxr::SdfPath& parent_sdf_path);

  absl::Status InitializeUsdRigidBodyRecursive(
      const size_t mujoco_parent_body_index,
      const pxr::SdfPath& input_parent_sdf_path,
      const pxr::GfMatrix4d root_pose_parent_transform);

  absl::Status AddToUsdStageInternal(const pxr::SdfPath& path);

  const mjModel* mjmodel_;
  const mjData* mjdata_;
  pxr::UsdStageRefPtr stage_;
};

}  // namespace mujoco

#endif  // THIRD_PARTY_MUJOCO_SRC_USD_IMPORTERS_MJCF_MUJOCO_TO_USD_H_
