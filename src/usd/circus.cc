#include "third_party/mujoco/src/usd/circus.h"

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
#include "googlex/proxy/simulation/scene_builder/bundles/asset_parsers/reference_collector.h"
#include "googlex/proxy/simulation/simian/utils/file_resolver.h"
#include "googlex/proxy/simulation/utils/file_bundle/file_bundle.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/strip.h"
#include "third_party/absl/time/time.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/mujoco/src/usd/circus_util.h"
#include "third_party/mujoco/src/user/user_api.h"
#include "third_party/mujoco/src/xml/xml.h"
#include "third_party/mujoco/src/xml/xml_native_reader.h"
#include "third_party/tinyxml2/tinyxml2.h"
#include "third_party/usd/v23/google/plugin_registration.h"
#include "third_party/usd/v23/pxr/base/gf/matrix4d.h"
#include "third_party/usd/v23/pxr/base/gf/quatd.h"
#include "third_party/usd/v23/pxr/base/gf/rotation.h"
#include "third_party/usd/v23/pxr/base/gf/vec3d.h"
#include "third_party/usd/v23/pxr/base/gf/vec3f.h"
#include "third_party/usd/v23/pxr/base/plug/registry.h"
#include "third_party/usd/v23/pxr/base/tf/token.h"
#include "third_party/usd/v23/pxr/base/vt/array.h"
#include "third_party/usd/v23/pxr/imaging/hd/rendererPlugin.h"
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
#include "third_party/usd/v23/pxr/usd/usdGeom/primvarsAPI.h"
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
#include "util/task/contrib/status_macros/ret_check.h"
#include "util/task/status_macros.h"
#include "util/tuple/components/dump_vars.h"

namespace mujoco {

absl::StatusOr<std::unique_ptr<Circus>> Circus::Create(
    const std::vector<mjCircusLayer>& circus_layers) {
  RETURN_IF_ERROR(pxr::google::RegisterUsdPlugins());

  // Create an empty world stage.
  pxr::UsdStageRefPtr world_stage = pxr::UsdStage::CreateInMemory();
  RET_CHECK(world_stage != nullptr) << "Failed to create world UsdStage.";

  // Orient the Z access as up.
  RET_CHECK(pxr::UsdGeomSetStageUpAxis(world_stage, pxr::UsdGeomTokens->z))
      << "Failed to set stage up axis.";

  auto circus = absl::WrapUnique<Circus>(new Circus(world_stage));

  RETURN_IF_ERROR(circus->AddLayers(circus_layers));

  return circus;
}

/////////////////////// Initialize USD Geometries in MuJoCo ////////////////////

absl::Status Circus::InitializeMujocoGeomMesh(
    const pxr::UsdGeomMesh& usd_geom_mesh, mjsBody* body, mjsGeom* geom) {
  geom->type = mjGEOM_MESH;

  // Extract all vertices from the USD mesh.
  const pxr::UsdGeomPoints& points =
      pxr::UsdGeomPoints(usd_geom_mesh.GetPointsAttr().GetPrim());
  pxr::VtArray<pxr::GfVec3f> usd_points;
  usd_geom_mesh.GetPointsAttr().Get<pxr::VtArray<pxr::GfVec3f>>(&usd_points);
  std::vector<float> mujoco_vertices;
  for (const pxr::GfVec3f& p : usd_points) {
    mujoco_vertices.push_back(p[0]);
    mujoco_vertices.push_back(p[1]);
    mujoco_vertices.push_back(p[2]);
  }

  // Confirm all faces are triangular.
  pxr::VtArray<int> face_vertex_counts;
  usd_geom_mesh.GetFaceVertexCountsAttr().Get<pxr::VtArray<int>>(
      &face_vertex_counts);

  for (int vert_count : face_vertex_counts) {
    RET_CHECK_EQ(vert_count, 3)
        << "Only face vertex count of 3 is supported, got "
        << vert_count;
  }

  // Extract faces from the USD mesh.
  pxr::VtArray<int> usd_faces;
  usd_geom_mesh.GetFaceVertexIndicesAttr().Get<pxr::VtArray<int>>(&usd_faces);
  std::vector<int> mujoco_faces;
  int face_index = 0;

  for (int vert_count : face_vertex_counts) {
    mujoco_faces.push_back(usd_faces[face_index]);
    mujoco_faces.push_back(usd_faces[face_index + 1]);
    mujoco_faces.push_back(usd_faces[face_index + 2]);
    face_index += vert_count;
  }

  auto* mesh = mjs_addMesh(&mjcmodel_->spec, 0);
  std::string mesh_name = usd_geom_mesh.GetPath().GetAsString() + "_mesh";
  mjs_setString(mesh->name, mesh_name.c_str());
  mjs_setFloat(mesh->uservert, mujoco_vertices.data(), mujoco_vertices.size());
  mjs_setInt(mesh->userface, mujoco_faces.data(), mujoco_faces.size());

  mjs_setString(geom->meshname, mesh_name.c_str());

  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoGeomCapsule(
    const pxr::UsdGeomCapsule& usd_geom_capsule, mjsBody* body, mjsGeom* geom) {
  geom->type = mjGEOM_CAPSULE;
  double radius;
  if (!usd_geom_capsule.GetRadiusAttr().Get<double>(&radius)) {
    return absl::InternalError("Unable to get size for " +
                               usd_geom_capsule.GetPath().GetAsString());
  };

  double height;
  if (!usd_geom_capsule.GetHeightAttr().Get<double>(&height)) {
    return absl::InternalError("Unable to get height for " +
                               usd_geom_capsule.GetPath().GetAsString());
  }

  ASSIGN_OR_RETURN(auto capsule_matrix4d,
                   GetUsdWorldPose(usd_geom_capsule.GetPath()));

  ASSIGN_OR_RETURN(auto scale, GetScale(capsule_matrix4d));

  // Radius
  geom->size[0] = scale[0] * radius;
  // Height
  // MuJoCo uses half height, USD uses full.
  geom->size[1] = scale[1] * height / 2;

  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoGeomCone(
    const pxr::UsdGeomCone& usd_geom_cone,mjsBody* body, mjsGeom* geom) {
  return absl::UnimplementedError("Cone shape is unsupported by MuJoCo");
}

absl::Status Circus::InitializeMujocoGeomCube(
    const pxr::UsdGeomCube& usd_geom_cube, mjsBody* body, mjsGeom* geom) {
  geom->type = mjGEOM_BOX;
  double size;
  if (!usd_geom_cube.GetSizeAttr().Get<double>(&size)) {
    return absl::InternalError("Unable to get size for " +
                               usd_geom_cube.GetPath().GetAsString());
  };

  ASSIGN_OR_RETURN(auto cube_matrix4d,
                   GetUsdWorldPose(usd_geom_cube.GetPath()));

  ASSIGN_OR_RETURN(auto scale, GetScale(cube_matrix4d));

  // MuJoCo uses half-length
  geom->size[0] = scale[0] * size / 2;
  geom->size[1] = scale[1] * size / 2;
  geom->size[2] = scale[2] * size / 2;

  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoGeomCylinder(
    const pxr::UsdGeomCylinder& usd_geom_cylinder, mjsBody* body,
    mjsGeom* geom) {
  geom->type = mjGEOM_CYLINDER;
  double radius;
  if (!usd_geom_cylinder.GetRadiusAttr().Get<double>(&radius)) {
    return absl::InternalError("Unable to get size for " +
                               usd_geom_cylinder.GetPath().GetAsString());
  };

  double height;
  if (!usd_geom_cylinder.GetHeightAttr().Get<double>(&height)) {
    return absl::InternalError("Unable to get height for " +
                               usd_geom_cylinder.GetPath().GetAsString());
  }

  ASSIGN_OR_RETURN(auto cylinder_matrix4d,
                   GetUsdWorldPose(usd_geom_cylinder.GetPath()));

  ASSIGN_OR_RETURN(auto scale, GetScale(cylinder_matrix4d));

  // Radius
  geom->size[0] = scale[0] * radius;
  // Height
  // MuJoCo uses half height
  geom->size[1] = scale[1] * height / 2;

  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoGeomSphere(
    const pxr::UsdGeomSphere& usd_geom_sphere, mjsBody* body, mjsGeom* geom) {
  geom->type = mjGEOM_SPHERE;
  double radius;
  if (!usd_geom_sphere.GetRadiusAttr().Get<double>(&radius)) {
    return absl::InternalError("Unable to get size for " +
                               usd_geom_sphere.GetPath().GetAsString());
  };

  ASSIGN_OR_RETURN(auto sphere_matrix4d,
                   GetUsdWorldPose(usd_geom_sphere.GetPath()));

  ASSIGN_OR_RETURN(auto scale, GetScale(sphere_matrix4d));

  geom->size[0] = scale[0] * radius;
  geom->size[1] = scale[1] * radius;
  geom->size[2] = scale[2] * radius;

  return absl::OkStatus();
}

/////////////////////// Initialize USD Body and Geoms in MuJoCo ////////////////
absl::Status Circus::InitializeMujocoBody(
    const pxr::UsdPrim& child_body_prim,
    const pxr::GfMatrix4d& usd_world_pose_parent_body,
    mjsBody* parent_body,
    mjsBody** child_body,
    pxr::GfMatrix4d* usd_world_pose_child_body,
    double* body_mass) {
  *child_body = mjs_addBody(parent_body, 0);
  mjs_setString(
      (*child_body)->name, child_body_prim.GetPath().GetAsString().c_str());
  ASSIGN_OR_RETURN(*usd_world_pose_child_body,
                   GetUsdWorldPose(child_body_prim.GetPath()));

  pxr::GfMatrix4d usd_parent_body_pose_child_body =
      usd_world_pose_parent_body.GetInverse() * (*usd_world_pose_child_body);
  TransformToMujocoPosQuat(
      usd_parent_body_pose_child_body, (*child_body)->pos, (*child_body)->quat);

  std::stack<pxr::UsdPrim> geom_candidate_prims;
  geom_candidate_prims.push(child_body_prim);
  *body_mass = 0.0;
  while (!geom_candidate_prims.empty()) {
    auto geom_candidate_prim = geom_candidate_prims.top();
    geom_candidate_prims.pop();

    for (const auto& geom_candidate_child_prim :
             geom_candidate_prim.GetChildren()) {
      geom_candidate_prims.push(geom_candidate_child_prim);
    }

    if (!IsGeom(geom_candidate_prim)) {
      continue;
    }

    auto* geom = mjs_addGeom((*child_body), 0);
    mjs_setString(
        geom->name,
        (geom_candidate_prim.GetPath().GetAsString() + "_geom").c_str());

    if (geom_candidate_prim.HasAPI<pxr::UsdPhysicsCollisionAPI>()) {
      geom->contype = 1;
      geom->conaffinity = 1;
    } else {
      geom->contype = 0;
      geom->conaffinity = 0;
      geom->group = kVisualGeomGroup;
      geom->density = 0;
    }

    if (geom_candidate_prim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
      auto usd_physics_mass = pxr::UsdPhysicsMassAPI(geom_candidate_prim);
      pxr::VtValue mass_value;
      usd_physics_mass.GetMassAttr().Get(&mass_value);
      geom->mass = mass_value.Get<float>();
      *body_mass += geom->mass;
    }

    auto xformable = pxr::UsdGeomXformable::Get(
        world_stage_, geom_candidate_prim.GetPath());
    pxr::TfToken visibility;
    xformable.GetVisibilityAttr().Get<pxr::TfToken>(&visibility);
    if (visibility != pxr::UsdGeomTokens->invisible) {
      RETURN_IF_ERROR(ApplyColorIfPresent(xformable, geom));
    }

    if (geom_candidate_prim.IsA<pxr::UsdGeomMesh>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomMesh(
          pxr::UsdGeomMesh::Get(world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else if (geom_candidate_prim.IsA<pxr::UsdGeomCapsule>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomCapsule(
          pxr::UsdGeomCapsule::Get(world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else if (geom_candidate_prim.IsA<pxr::UsdGeomCone>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomCone(
          pxr::UsdGeomCone::Get(world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else if (geom_candidate_prim.IsA<pxr::UsdGeomCube>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomCube(
          pxr::UsdGeomCube::Get(world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else if (geom_candidate_prim.IsA<pxr::UsdGeomCylinder>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomCylinder(
          pxr::UsdGeomCylinder::Get(
              world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else if (geom_candidate_prim.IsA<pxr::UsdGeomSphere>()) {
      RETURN_IF_ERROR(InitializeMujocoGeomSphere(
          pxr::UsdGeomSphere::Get(world_stage_, geom_candidate_prim.GetPath()),
          *child_body, geom));
    } else {
      return absl::UnimplementedError(
          "Unimplemented prim type: " +
          std::string(
              geom_candidate_prim.GetPrimTypeInfo().GetSchemaTypeName()));
    }
  }

  if (*body_mass == 0 &&
    child_body_prim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
    auto usd_physics_mass = pxr::UsdPhysicsMassAPI(child_body_prim);
    pxr::VtValue mass_value;
    usd_physics_mass.GetMassAttr().Get(&mass_value);
    *body_mass = mass_value.Get<float>();
  }

  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoJoint(
    const pxr::UsdPhysicsJoint& usd_physics_joint, mjsBody* parent_body,
    mjsBody* child_body) {

  std::map<std::string, std::vector<double>> joint_types = {
    {"rotX", {1.0, 0.0, 0.0}},
    {"rotY", {0.0, 1.0, 0.0}},
    {"rotZ", {0.0, 0.0, 1.0}},
    {"transX", {1.0, 0.0, 0.0}},
    {"transY", {0.0, 1.0, 0.0}},
    {"transZ", {0.0, 0.0, 1.0}},
  };


  for (const auto& joint_type : joint_types) {
    auto joint_property =
        pxr::TfToken(absl::StrCat("limit:", joint_type.first));

    auto usd_physics_limit_api = pxr::UsdPhysicsLimitAPI::Get(
        world_stage_,
        usd_physics_joint.GetPath().AppendProperty(joint_property));
    float high_attr;
    float low_attr;
    usd_physics_limit_api.GetHighAttr().Get<float>(&high_attr);
    usd_physics_limit_api.GetLowAttr().Get<float>(&low_attr);
    if (high_attr == low_attr) {
      continue;
    }

    auto* joint = mjs_addJoint(child_body, 0);
    mjs_setString(
        joint->name,
        absl::StrCat(
            usd_physics_joint.GetPath().GetAsString(), joint_type.first)
        .c_str());

    if (absl::StartsWith(joint_type.first, "rot")) {
      joint->type = mjJNT_HINGE;
    } else {
      joint->type = mjJNT_SLIDE;
    }
    // MATTT FIX THIS
    // While this is correct it does not take into account any rotation
    // of the joint itself.
    joint->axis[0] = joint_type.second[0];
    joint->axis[1] = joint_type.second[1];
    joint->axis[2] = joint_type.second[2];
    joint->range[0] = low_attr;
    joint->range[1] = high_attr;

    // MuJoCo joint poses are defined relative to the child body.
    // The Parent Pose should just be the inverse.
    pxr::GfVec3f local_pos1;
    usd_physics_joint.GetLocalPos1Attr().Get<pxr::GfVec3f>(&local_pos1);
    joint->pos[0] = local_pos1[0];
    joint->pos[1] = local_pos1[1];
    joint->pos[2] = local_pos1[2];
  }

  return absl::OkStatus();
}

absl::Status Circus::InitializeKinematicChain(
    const pxr::SdfPath& parent_sdf_path,
    const std::map<pxr::SdfPath, std::vector<pxr::SdfPath>>& all_nodes_map,
    const pxr::GfMatrix4d& usd_world_pose_parent_body,
    mjsBody* parent_body) {
  // Add Node to parent_body.
  auto parent_node_it = all_nodes_map.find(parent_sdf_path);
  if (parent_node_it == all_nodes_map.end()) {
    return absl::InvalidArgumentError("Unable to find parent node: " +
                                      parent_sdf_path.GetAsString());
  }

  for (const auto& joint_sdf_path : parent_node_it->second) {
    auto usd_physics_joint =
        pxr::UsdPhysicsJoint::Get(world_stage_, joint_sdf_path);
    pxr::SdfPathVector body1_targets;
    usd_physics_joint.GetBody1Rel().GetTargets(&body1_targets);
    auto child_sdf_path = body1_targets.front();
    auto child_body_prim = world_stage_->GetPrimAtPath(child_sdf_path);
    mjsBody* child_body;
    pxr::GfMatrix4d usd_world_pose_child_body;
    double body_mass;
    RETURN_IF_ERROR(InitializeMujocoBody(
        child_body_prim, usd_world_pose_parent_body, parent_body, &child_body,
        &usd_world_pose_child_body, &body_mass));

    // Add joint linking the child body to the parent.
    RETURN_IF_ERROR(InitializeMujocoJoint(
        usd_physics_joint, parent_body, child_body));


    RETURN_IF_ERROR(InitializeKinematicChain(
        child_sdf_path, all_nodes_map, usd_world_pose_child_body, child_body));
  }

  return absl::OkStatus();
}

absl::Status Circus::UpdateMujocoModelKeyframe() {
  // Save Keyframe
  mjmodel_->key_time[mjcmodel_keyframe_id_] = mjdata_->time;
  mju_copy(&mjmodel_->key_qpos[mjcmodel_keyframe_id_ * mjmodel_->nq],
           mjdata_->qpos, mjmodel_->nq);
  mju_copy(&mjmodel_->key_qvel[mjcmodel_keyframe_id_ * mjmodel_->nv],
           mjdata_->qvel, mjmodel_->nv);
  mju_copy(&mjmodel_->key_act[mjcmodel_keyframe_id_ * mjmodel_->na],
           mjdata_->act, mjmodel_->na);
  mju_copy(&mjmodel_->key_mpos[mjcmodel_keyframe_id_ * 3 * mjmodel_->nmocap],
           mjdata_->mocap_pos, 3 * mjmodel_->nmocap);
  mju_copy(&mjmodel_->key_mquat[mjcmodel_keyframe_id_ * 4 * mjmodel_->nmocap],
           mjdata_->mocap_quat, 4 * mjmodel_->nmocap);
  mju_copy(&mjmodel_->key_ctrl[mjcmodel_keyframe_id_ * mjmodel_->nu],
           mjdata_->ctrl, mjmodel_->nu);
  mjcmodel_->CopyBack(mjmodel_);
  return absl::OkStatus();
}

absl::Status Circus::InitializeMujocoFromWorldStage() {
  mjcmodel_ = std::make_unique<mjCModel>();

  // Start with an MJCF File if one is specified.
  if (!mjcf_file_to_merge_.empty()) {
    mjs_setString(
        mjcmodel_->spec.meshdir,
        std::string(file::Dirname(mjcf_file_to_merge_)).c_str());
    mjXReader mjx_reader;
    mjx_reader.SetModel(&mjcmodel_->spec);
    ASSIGN_OR_RETURN(
        auto mjcf_string,
        file::GetContents(mjcf_file_to_merge_, file::Defaults()));
    tinyxml2::XMLDocument mjcf_xml;
    auto xml_error = mjcf_xml.Parse(mjcf_string.c_str());
    if (xml_error != tinyxml2::XML_NO_ERROR) {
      return util::InvalidArgumentErrorBuilder()
            << "Invalid MJCF string error: " << xml_error;
    }
    mjx_reader.Parse(mjcf_xml.RootElement());
    if (mjcmodel_->spec.inertiafromgeom == 0 /* mjINERTIAFROMGEOM_FALSE*/) {
      return absl::InvalidArgumentError(
          "MJCF file sets mjINERTIAFROMGEOM_FALSE which is unsupported");
    }
  }

  // Populate the rigid bodies in the mjCModel.
  mjcmodel_->spec.strippath = false;
  mjcmodel_->spec.inertiafromgeom = 1;  // mjINERTIAFROMGEOM_TRUE;
  mjcmodel_->spec.inertiagrouprange[0] = 0;
  mjcmodel_->spec.inertiagrouprange[1] = 5;
  mjcmodel_->spec.option.timestep = 0.01;
  mjcmodel_keyframe_id_ = mjcmodel_->AddKey()->id;

  // Map of parent bodies to their joints in kinematic chain.
  std::map<pxr::SdfPath, std::vector<pxr::SdfPath>> all_nodes_map;

  // Map of all nodes that have no parents, this represents the base of all
  // multi-bodies.
  std::set<pxr::SdfPath> root_nodes;

  // First find all possible geoms that could be loaded. This is any prim that
  // supports the UsdPhysicsCollisionAPI.
  for (const auto& prim : world_stage_->Traverse()) {
    // The Collision API is the minimum necessary for physics to matter.
    if (!prim.HasAPI<pxr::UsdPhysicsCollisionAPI>()) {
      continue;
    }

    // Attempt to find a RigidBody which owns this prim. If one is found use it.
    // If one is not found then this prim will be its own RigidBody.
    auto body_prim = prim;
    auto candidate_prim = prim;
    do {
      if (candidate_prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>()){
        body_prim = candidate_prim;
        break;
      }
      candidate_prim = candidate_prim.GetParent();
    } while (candidate_prim.IsValid());

    all_nodes_map.insert({body_prim.GetPath(), {}});
    root_nodes.insert(body_prim.GetPath());
  }

  // Next add Joints to construct kinematic chains.
  for (const auto& prim : world_stage_->Traverse()) {
    if (!prim.IsA<pxr::UsdPhysicsJoint>()) {
      continue;
    }
    const auto& joint_prim =
        pxr::UsdPhysicsJoint::Get(world_stage_, prim.GetPath());
    pxr::SdfPathVector body0_targets;
    joint_prim.GetBody0Rel().GetTargets(&body0_targets);
    auto parent_sdf_path = body0_targets.front();
    auto it = all_nodes_map.find(parent_sdf_path);
    if (it == all_nodes_map.end()) {
      return absl::InvalidArgumentError("Unable to find parent node: " +
                                        parent_sdf_path.GetAsString());
    }
    it->second.push_back(joint_prim.GetPath());

    // Any child by definition is not a root node so remove
    pxr::SdfPathVector body1_targets;
    joint_prim.GetBody1Rel().GetTargets(&body1_targets);
    auto child_sdf_path = body1_targets.front();
    root_nodes.erase(child_sdf_path);
  }

  all_nodes_map.insert({pxr::SdfPath(kWorldNodeName), {}});

  for (const auto& root_node : root_nodes) {
    auto it = all_nodes_map.find(root_node);
    if (it == all_nodes_map.end()) {
      return absl::InternalError("Unable to find node: " +
                                 root_node.GetAsString());
    }

    auto root_body_prim = world_stage_->GetPrimAtPath(root_node);
    mjsBody* root_body;
    pxr::GfMatrix4d usd_identity_pose;
    usd_identity_pose.SetIdentity();
    pxr::GfMatrix4d world_pose_root_body;
    double body_mass;
    RETURN_IF_ERROR(InitializeMujocoBody(
        root_body_prim, usd_identity_pose, &mjcmodel_->GetWorld()->spec,
        &root_body, &world_pose_root_body, &body_mass));

    // Add Free Joint making the body independent if it has mass. If no mass
    // in any of the geoms assume it's a static collision object.
    if (body_mass > 0) {
      auto* free_joint = mjs_addJoint(root_body, 0);
      mjs_setString(
          free_joint->name,
          (root_node.GetAsString() + kFreeJointNameSuffix).c_str());
      free_joint->type = mjJNT_FREE;
    }

    // Follow any rigid body links.
    RETURN_IF_ERROR(InitializeKinematicChain(
        root_node, all_nodes_map, world_pose_root_body, root_body));
  }

  auto new_mjmodel = mjcmodel_->Compile();
  if (new_mjmodel == nullptr) {
    return util::InternalErrorBuilder()
           << "Unable to create mjModel: " << mjcmodel_->GetError().message;
  }

  // Configure Physics Options.
  mj_defaultOption(&new_mjmodel->opt);

  // make data and run once (to init rendering and check simulation)
  auto new_mjdata = mj_makeData(new_mjmodel);
  if (new_mjdata == nullptr) {
    return util::InternalErrorBuilder()
           << "Unable to create mjData: " << mjcmodel_->GetError().message;
  }

  // Confirm that the simulation can compute forward dynamics, but don't step.
  mj_forward(new_mjmodel, new_mjdata);

  std::swap(mjmodel_, new_mjmodel);
  std::swap(mjdata_, new_mjdata);

  // Delete old model and old data (after swap) if they were populated.
  if (new_mjdata != nullptr) {
    mj_deleteData(new_mjdata);
  }

  if (new_mjmodel != nullptr) {
    mj_deleteModel(new_mjmodel);
  }

  return absl::OkStatus();
}

absl::Status Circus::AddUsdLayer(const mjCircusLayer& circus_layer) {
  RET_CHECK(file::IsAbsolutePath(circus_layer.scene_path))
            .SetCode(absl::StatusCode::kInvalidArgument)
        << "scene_path must be absolute: "
      << DUMP_VARS(circus_layer.scene_path);

    pxr::UsdStageRefPtr layer_stage =
        pxr::UsdStage::Open(circus_layer.file_path);

    // Create a UsdGeomXForm to attach the new layer to, this will allow the
    // layer to be posed relative to its parent.
    auto circus_layer_usd_path = pxr::SdfPath(circus_layer.scene_path);
    pxr::UsdGeomXform circus_layer_usd_xform =
        pxr::UsdGeomXform::Define(world_stage_, circus_layer_usd_path);

    // Wrap the Prim in an XForm to allow posing it.
    auto container_prim = world_stage_->GetPrimAtPath(circus_layer_usd_path);

    // Attach the layer stage to the container prim.
    container_prim.GetReferences().AddReference(
        layer_stage->GetRootLayer()->GetIdentifier());

    return SetLocalPose(circus_layer.scene_path, circus_layer.pose);
}

absl::Status Circus::AddMjcfLayer(const mjCircusLayer& circus_layer) {
  if (!mjcf_file_to_merge_.empty()) {
    return absl::InvalidArgumentError("Only 1 MJCF layer is supported");
  }
  mjcf_file_to_merge_ = circus_layer.file_path;
  return absl::OkStatus();
}

absl::Status Circus::AddLayers(
    const std::vector<mjCircusLayer>& circus_layers) {
  for (const auto& circus_layer : circus_layers) {
    if (file::Extension(circus_layer.file_path) == "xml") {
      RETURN_IF_ERROR(AddMjcfLayer(circus_layer));
    } else {
      RETURN_IF_ERROR(AddUsdLayer(circus_layer));
    }
  }

  // TODO(b/316716516): Currently fully recompiles the mjCModel, which might be
  // needlessly expensive. When adding Scene Composition determine if
  // incremental model updates are possible.
  RETURN_IF_ERROR(InitializeMujocoFromWorldStage());

  return absl::OkStatus();
}

absl::Status Circus::SetLocalPose(const std::string& scene_path,
                                 const mjCircusPose& parent_pose_object) {
  RET_CHECK(file::IsAbsolutePath(scene_path))
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "scene_path must be absolute: " << DUMP_VARS(scene_path);

  auto usd_path = pxr::SdfPath(scene_path);

  pxr::GfMatrix4d mat = MjCircusPoseToTransform(parent_pose_object);

  RETURN_IF_ERROR(SetXformTransform(usd_path, mat, world_stage_));

  return absl::OkStatus();
}

absl::Status Circus::SetWorldPose(const std::string& scene_path,
                                 const mjCircusPose& world_pose_object) {
  RET_CHECK(file::IsAbsolutePath(scene_path))
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "scene_path must be absolute: " << DUMP_VARS(scene_path);

  auto scene_usd_path = pxr::SdfPath(scene_path);

  auto prim = world_stage_->GetPrimAtPath(scene_usd_path);
  RET_CHECK(prim).SetCode(absl::StatusCode::kNotFound)
      << "Unable to find prim at path: " << DUMP_VARS(scene_path);

  pxr::UsdGeomXformCache transform_cache;
  pxr::GfMatrix4d world_pose_parent =
      transform_cache.GetParentToWorldTransform(prim);
  pxr::GfMatrix4d world_pose_prim = MjCircusPoseToTransform(world_pose_object);
  pxr::GfMatrix4d parent_pose_prim =
      world_pose_parent.GetInverse() * world_pose_prim;

  RETURN_IF_ERROR(SetXformTransform(
      scene_usd_path, parent_pose_prim, world_stage_));

  return absl::OkStatus();
}

absl::StatusOr<pxr::GfMatrix4d> Circus::GetUsdLocalPose(
    const pxr::SdfPath& prim_path) {
  auto prim = world_stage_->GetPrimAtPath(prim_path);
  RET_CHECK(prim).SetCode(absl::StatusCode::kNotFound)
      << "Unable to find prim at path: " << DUMP_VARS(prim_path) << " in "
      << PrintAllRealPrims(world_stage_);
  auto xform = pxr::UsdGeomXform(prim);

  pxr::UsdGeomXformCache transform_cache;
  bool reset_xforms_stack;
  pxr::GfMatrix4d parent_pose_object = transform_cache.GetLocalTransformation(
      xform.GetPrim(), &reset_xforms_stack);
  RET_CHECK(!reset_xforms_stack) << "Unexpected reset_xforms_stack";
  return parent_pose_object;
}

absl::StatusOr<mjCircusPose> Circus::GetLocalPose(
    const std::string& scene_path) {
  RET_CHECK(file::IsAbsolutePath(scene_path))
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "scene_path must be absolute: " << DUMP_VARS(scene_path);

  auto usd_scene_path = pxr::SdfPath(scene_path);

  ASSIGN_OR_RETURN(pxr::GfMatrix4d parent_pose_object,
                   GetUsdLocalPose(usd_scene_path));

  return TransformToMjCircusPose(parent_pose_object);
}

absl::StatusOr<pxr::GfMatrix4d> Circus::GetUsdWorldPose(
    const pxr::SdfPath& prim_path) {
  auto prim = world_stage_->GetPrimAtPath(prim_path);
  RET_CHECK(prim).SetCode(absl::StatusCode::kNotFound)
      << "Unable to find prim at path: " << DUMP_VARS(prim_path) << " in "
      << PrintAllRealPrims(world_stage_);
  auto xform = pxr::UsdGeomXform(prim);

  pxr::UsdGeomXformCache transform_cache;
  return transform_cache.GetLocalToWorldTransform(xform.GetPrim());
}

absl::StatusOr<mjCircusPose> Circus::GetWorldPose(
    const std::string& scene_path) {
  RET_CHECK(file::IsAbsolutePath(scene_path))
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "scene_path must be absolute: " << DUMP_VARS(scene_path);

  auto usd_scene_path = pxr::SdfPath(scene_path);

  ASSIGN_OR_RETURN(pxr::GfMatrix4d world_pose_object,
                   GetUsdWorldPose(usd_scene_path));

  return TransformToMjCircusPose(world_pose_object);
}

absl::Status Circus::SyncMujocoToUsdStage() {
  for (size_t body_index = 1; body_index < mjmodel_->nbody; ++body_index) {
    auto body_name =
        std::string(&mjmodel_->names[mjmodel_->name_bodyadr[body_index]]);
    mjCircusPose world_pose_body;
    MujocoTranslationToMjCircusPose(
        mjdata_->xpos, body_index, &world_pose_body);
    MujocoQuaternionToMjCircusPose(
        mjdata_->xquat, body_index, &world_pose_body);
    RETURN_IF_ERROR(SetWorldPose(body_name, world_pose_body));
  }

  return absl::OkStatus();
}

absl::StatusOr<mjModel*> Circus::CreateMjModel() {
  RETURN_IF_ERROR(UpdateMujocoModelKeyframe());
  return mjcmodel_->Compile();
}

absl::StatusOr<googlex::proxy::simulation::FileBundle> Circus::ExportMjcf() {
  RETURN_IF_ERROR(UpdateMujocoModelKeyframe());
  std::array<char, 1024> error;
  std::string xml_string = mjWriteXML(
      &mjcmodel_->spec, error.data(), error.size());

  tinyxml2::XMLDocument mjcf_xml;
  auto xml_error = mjcf_xml.Parse(xml_string.c_str());
  if (xml_error != tinyxml2::XML_NO_ERROR) {
    return util::InvalidArgumentErrorBuilder()
           << "Invalid MJCF string error: " << xml_error;
  }

  // Extract assets and convert them to relative destinations.
  blue::ReferenceCollector reference_collector;
  auto* asset = mjcf_xml.RootElement()->FirstChildElement("asset");
  if (asset) {
    auto* asset_child = asset->FirstChildElement();
    while (asset_child != nullptr) {
      if (asset_child->Attribute("file") == nullptr) {
        asset_child = asset_child->NextSiblingElement();
        continue;
      }
      std::string file_attribute = asset_child->Attribute("file");
      if (file_attribute.empty()) {
        asset_child = asset_child->NextSiblingElement();
        continue;
      }
      auto relative_path = std::string(absl::StripPrefix(
          file_attribute, *(std::string*)mjcmodel_->spec.meshdir));
      relative_path = std::string(absl::StripPrefix(relative_path, "/"));
      RETURN_IF_ERROR(reference_collector.AddReferencedAssets(blue::PathPair{
          .relative_path = relative_path,
          .resolved_path = file::JoinPath(
              *(std::string*)mjcmodel_->spec.meshdir, relative_path)}));
      asset_child->SetAttribute("file", relative_path.c_str());
      asset_child = asset_child->NextSiblingElement();
    }
  }

  tinyxml2::XMLPrinter printer;
  mjcf_xml.Accept(&printer);

  // Add MJCF to Bundle
  googlex::proxy::simulation::FileBundle file_bundle;
  (*file_bundle.mutable_file_data())["circus_extract.xml"] =
      std::string(printer.CStr());

  // Add Assets to Bundle
  for (const auto& entry : reference_collector.GetCollectedPaths()) {
    RETURN_IF_ERROR(googlex::proxy::simulation::AddFileToBundleAs(
        entry.resolved_path, entry.relative_path, &file_bundle));
  }

  return file_bundle;
}

absl::StatusOr<std::string> Circus::ExportUsd() {
  std::string output;
  world_stage_->ExportToString(&output);
  return output;
}

}  // namespace mujoco
