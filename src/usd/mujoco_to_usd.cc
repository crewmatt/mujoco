#include "third_party/mujoco/src/usd/importers/mjcf/mujoco_to_usd.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/mujoco/src/usd/circus_util.h"
#include "third_party/robotics/lib/math/geometry_typedefs.h"
#include "third_party/robotics/lib/math/quaternion.h"
#include "third_party/usd/v23/pxr/base/gf/quatf.h"
#include "third_party/usd/v23/pxr/base/gf/vec3f.h"
#include "third_party/usd/v23/pxr/base/tf/token.h"
#include "third_party/usd/v23/pxr/base/vt/array.h"
#include "third_party/usd/v23/pxr/usd/sdf/path.h"
#include "third_party/usd/v23/pxr/usd/sdf/types.h"
#include "third_party/usd/v23/pxr/usd/usd/common.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/capsule.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cube.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/cylinder.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/mesh.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/metrics.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/points.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/primvarsAPI.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/scope.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/sphere.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xform.h"
#include "third_party/usd/v23/pxr/usd/usdGeom/xformable.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/collisionAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/joint.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/limitAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/prismaticJoint.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/revoluteJoint.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/sphericalJoint.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/massAPI.h"
#include "third_party/usd/v23/pxr/usd/usdPhysics/rigidBodyAPI.h"
#include "util/geometry/s1angle.h"
#include "util/task/status_macros.h"

namespace mujoco {
namespace {

// The ID of the World in mjModel and mjData.
static constexpr int kWorldIndex = 0;

static constexpr char kRootPath[] = "/root";

static constexpr char kBody[] = "body";
static constexpr char kJoint[] = "joint";
static constexpr char kGeom[] = "geom";

// USD Scope used to contain Joints.
static constexpr char kJointScope[] = "Joints";

}  // namespace

std::string MujocoToUsd::GetUsdName(
    const size_t name_index, const std::string& type, const size_t type_index) {
  std::string result = &mjmodel_->names[name_index];
  if (result.empty()) {
    result = type + "_" + std::to_string(type_index);
  }
  result = absl::StrReplaceAll(result, {{".", "_"}});
  return result;
}

absl::Status MujocoToUsd::InitializeUsdGeomXformable(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path,
    pxr::UsdGeomXformable* usd_geom_xformable) {
  RETURN_IF_ERROR(SetXformTransform(
      MujocoPosQuatToTransform(&mjmodel_->geom_pos[3 * mujoco_geom_index],
                               &mjmodel_->geom_quat[4 * mujoco_geom_index]),
      *usd_geom_xformable));

  // Add Color to the geom.
  auto api = pxr::UsdGeomPrimvarsAPI(usd_geom_xformable->GetPrim());
  auto color_primvar = api.CreatePrimvar(pxr::TfToken(kDisplayColorPrimvar),
                                         pxr::SdfValueTypeNames->Color3fArray);

  // MuJoCo is RGBA not RGB.
  pxr::VtArray<pxr::GfVec3f> usd_rgb;
  if (mjmodel_->geom_matid[mujoco_geom_index] == -1) {
    usd_rgb = {{
        mjmodel_->geom_rgba[4 * mujoco_geom_index],
        mjmodel_->geom_rgba[4 * mujoco_geom_index + 1],
        mjmodel_->geom_rgba[4 * mujoco_geom_index + 2],
    }};
  } else {
    usd_rgb = {{
        mjmodel_->mat_rgba[mjmodel_->geom_matid[mujoco_geom_index] * 4],
        mjmodel_->mat_rgba[mjmodel_->geom_matid[mujoco_geom_index] * 4 + 1],
        mjmodel_->mat_rgba[mjmodel_->geom_matid[mujoco_geom_index] * 4 + 2],
    }};
  }

  color_primvar.Set(usd_rgb);

  if (mjmodel_->geom_contype[mujoco_geom_index] == 1) {
    pxr::UsdPhysicsCollisionAPI::Apply(usd_geom_xformable->GetPrim());
  }

  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeomMesh(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path) {
  auto usd_geom_mesh = pxr::UsdGeomMesh::Define(stage_, geom_sdf_path);
  RETURN_IF_ERROR(InitializeUsdGeomXformable(
      mujoco_body_index, mujoco_geom_index, geom_sdf_path, &usd_geom_mesh));

  size_t mujoco_data_id = mjmodel_->geom_dataid[mujoco_geom_index];
  size_t mujoco_vertex_start_index = mjmodel_->mesh_vertadr[mujoco_data_id] * 3;
  size_t vertex_number = mjmodel_->mesh_vertnum[mujoco_data_id];
  pxr::VtArray<pxr::GfVec3f> mujoco_points;
  mujoco_points.reserve(vertex_number);
  for (size_t mujoco_vertex_index = mujoco_vertex_start_index;
       mujoco_vertex_index < mujoco_vertex_start_index + (3 * vertex_number);
       mujoco_vertex_index += 3) {
    mujoco_points.push_back({
        mjmodel_->mesh_vert[mujoco_vertex_index],
        mjmodel_->mesh_vert[mujoco_vertex_index + 1],
        mjmodel_->mesh_vert[mujoco_vertex_index + 2],
    });
  }
  usd_geom_mesh.GetPointsAttr().Set<pxr::VtArray<pxr::GfVec3f>>(mujoco_points);

  size_t mujoco_face_start_index = mjmodel_->mesh_faceadr[mujoco_data_id] * 3;
  size_t face_number = mjmodel_->mesh_facenum[mujoco_data_id];
  pxr::VtArray<int> mujoco_faces;
  mujoco_faces.reserve(3 * face_number);
  for (size_t mujoco_face_index = mujoco_face_start_index;
       mujoco_face_index < mujoco_face_start_index + (3 * face_number);
       mujoco_face_index += 3) {
    mujoco_faces.push_back(mjmodel_->mesh_face[mujoco_face_index]);
    mujoco_faces.push_back(mjmodel_->mesh_face[mujoco_face_index + 1]);
    mujoco_faces.push_back(mjmodel_->mesh_face[mujoco_face_index + 2]);
  }
  usd_geom_mesh.GetFaceVertexIndicesAttr().Set<pxr::VtArray<int>>(mujoco_faces);

  pxr::VtArray<int> vertex_counts;
  for (int i = 0; i < face_number; ++i) {
    vertex_counts.push_back(3);
  }
  usd_geom_mesh.GetFaceVertexCountsAttr().Set<pxr::VtArray<int>>(vertex_counts);

  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeomCapsule(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path) {
  auto usd_geom_capsule = pxr::UsdGeomCapsule::Define(stage_, geom_sdf_path);
  RETURN_IF_ERROR(InitializeUsdGeomXformable(
      mujoco_body_index, mujoco_geom_index, geom_sdf_path, &usd_geom_capsule));
  // MuJoCo uses half sizes.
  usd_geom_capsule.GetRadiusAttr().Set<double>(
      mjmodel_->geom_size[3 * mujoco_geom_index]);
  usd_geom_capsule.GetHeightAttr().Set<double>(
      mjmodel_->geom_size[3 * mujoco_geom_index + 1] * 2);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeomCube(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path) {
  auto usd_geom_cube = pxr::UsdGeomCube::Define(stage_, geom_sdf_path);
  RETURN_IF_ERROR(InitializeUsdGeomXformable(
      mujoco_body_index, mujoco_geom_index, geom_sdf_path, &usd_geom_cube));
  // MuJoCo uses half sizes.
  pxr::GfVec3f scale = {
      static_cast<float>(mjmodel_->geom_size[3 * mujoco_geom_index] * 2),
      static_cast<float>(mjmodel_->geom_size[3 * mujoco_geom_index + 1] * 2),
      static_cast<float>(mjmodel_->geom_size[3 * mujoco_geom_index + 2] * 2)};
  float size = std::max({scale[0], scale[1], scale[2]});
  for (int i = 0; i < 3; ++i) {
    scale[i] = scale[i] / size;
  }
  usd_geom_cube.GetSizeAttr().Set<double>(size);
  pxr::UsdGeomXformable(usd_geom_cube.GetPrim())
      .AddScaleOp()
      .Set<pxr::GfVec3f>(scale);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeomCylinder(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path) {
  auto usd_geom_cylinder = pxr::UsdGeomCylinder::Define(stage_, geom_sdf_path);
  RETURN_IF_ERROR(InitializeUsdGeomXformable(
      mujoco_body_index, mujoco_geom_index, geom_sdf_path, &usd_geom_cylinder));
  // MuJoCo uses half sizes.
  usd_geom_cylinder.GetRadiusAttr().Set<double>(
      mjmodel_->geom_size[3 * mujoco_geom_index] * 2);
  usd_geom_cylinder.GetHeightAttr().Set<double>(
      mjmodel_->geom_size[3 * mujoco_geom_index + 1] * 2);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeomSphere(
    const size_t mujoco_body_index, const size_t mujoco_geom_index,
    const pxr::SdfPath& geom_sdf_path) {
  auto usd_geom_sphere = pxr::UsdGeomSphere::Define(stage_, geom_sdf_path);
  RETURN_IF_ERROR(InitializeUsdGeomXformable(
      mujoco_body_index, mujoco_geom_index, geom_sdf_path, &usd_geom_sphere));
  // MuJoCo uses half sizes.
  usd_geom_sphere.GetRadiusAttr().Set<double>(
      mjmodel_->geom_size[3 * mujoco_geom_index]);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdGeoms(
    const size_t mujoco_body_index, const pxr::SdfPath& body_sdf_path) {
  for (int mujoco_geom_index = mjmodel_->body_geomadr[mujoco_body_index];
       mujoco_geom_index < mjmodel_->body_geomadr[mujoco_body_index] +
                               mjmodel_->body_geomnum[mujoco_body_index];
       ++mujoco_geom_index) {
    auto geom_sdf_path = body_sdf_path.AppendPath(
        pxr::SdfPath(GetUsdName(
            mjmodel_->name_geomadr[mujoco_geom_index], kGeom,
            mujoco_geom_index)));
    if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_MESH) {
      RETURN_IF_ERROR(InitializeUsdGeomMesh(mujoco_body_index,
                                            mujoco_geom_index, geom_sdf_path));
    } else if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_BOX) {
      RETURN_IF_ERROR(InitializeUsdGeomCube(mujoco_body_index,
                                            mujoco_geom_index, geom_sdf_path));
    } else if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_CAPSULE) {
      RETURN_IF_ERROR(InitializeUsdGeomCapsule(
          mujoco_body_index, mujoco_geom_index, geom_sdf_path));
    } else if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_CYLINDER) {
      RETURN_IF_ERROR(InitializeUsdGeomCylinder(
          mujoco_body_index, mujoco_geom_index, geom_sdf_path));
    } else if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_ELLIPSOID) {
      // Initialize Ellipsoid as Sphere for now.
      RETURN_IF_ERROR(InitializeUsdGeomSphere(
          mujoco_body_index, mujoco_geom_index, geom_sdf_path));
    } else if (mjmodel_->geom_type[mujoco_geom_index] == mjGEOM_SPHERE) {
      RETURN_IF_ERROR(InitializeUsdGeomSphere(
          mujoco_body_index, mujoco_geom_index, geom_sdf_path));
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unimplemented geom type: ", mjmodel_->geom_type[mujoco_geom_index]));
    }
  }
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdPhysicsJoint(
    const size_t mujoco_parent_body_index,
    const size-t mujoco_child_body_index,
    const size_t mujoco_joint_index,
    const pxr::SdfPath& joint_sdf_path,
    const pxr::SdfPath& parent_body_sdf_path,
    const pxr::SdfPath& child_body_sdf_path) {
  auto usd_joint = pxr::UsdPhysicsJoint::Define(stage_, joint_sdf_path);
  usd_joint.GetBody0Rel().AddTarget(parent_body_sdf_path);
  usd_joint.GetBody1Rel().AddTarget(child_body_sdf_path);

  // Pose the joint on the child body.
  pxr:Vec3f parent_body_translate_joint = {
    mjmodel_->jnt_pos[3 * mujoco_joint_index],
    mjmodel_->jnt_pos[3 * mujoco_joint_index + 1],
    mjmodel_->jnt_pos[3 * mujoco_joint_index + 2],
  }
  usd_joint.GetLocalPos1Attr().Set<pxr::Vec3f>(body_pose_joint);
  
  // MuJoCo specifies Axis not Rotation, but USD only allows X, Y, Z axes.
  // Here rotate all USD Joints so that X lines up with the MuJoCo Axis
  auto child_body_rot_joint = robotics::math::QuaternionFromAxisAngle<float>(
      robotics::math::Vector3d({
        mjmodel_->jnt_axis[3 * mujoco_joint_index],
        mjmodel_->jnt_axis[3 * mujoco_joint_index + 1],
        mjmodel_->jnt_axis[3 * mujoco_joint_index + 2],
      }),
      S1Angle::Degrees(0.0));

  pxr::GfQuatf pxr_quaternion;
  pxr_quaternion.SetReal(child_body_rot_joint.w());
  pxr_quaternion.SetImaginary(
      child_body_rot_joint.x(), child_body_rot_joint.y(), child_body_rot_joint.z());
  usd_joint.GetLocalRot1Attr().Set<pxr::GfQuatf>(pxr_quaternion);

  // The Parent Pose of the joint is redundant and is just calculated from the child pose.
  // parent_pose_joint = parent_pose_root * root_pose_child * child_pose_joint
  
    switch (mjmodel_->jnt_type[mujoco_joint_index]) {
      case mjJNT_FREE: {
        // Free Joints should be handled as new root bodies.
        return absl::InvalidArgumentError(absl::StrCat(
            "Unimplemented joint type: ",
            mjmodel_->jnt_type[mujoco_joint_index]));
      }
      case mjJNT_BALL: {
        // Ball joints create all 3 degrees of rotational freedom
	      for (const auto& limit_type : {"RotX", "RotY", "RotZ"}) {
          auto joint_property =
              pxr::TfToken(absl::StrCat("limit:", limit_type));

          auto usd_physics_limit_api = pxr::UsdPhysicsLimitAPI::Define(
              world_stage_,
              usd_physics_joint.GetPath().AppendProperty(joint_property));
          usd_physics_limit_api.GetLowAttr().Set<float>(
              mjmodel_->jnt_range[2 * mujoco_joint_index]);
          usd_physics_limit_api.GetHighAttr().Set<float>(
              mjmodel_->jnt_range[2 * mujoco_joint_index + 1]);
        }
        break;
      }
      case mjJNT_HINGE: {
        auto joint_property =
            pxr::TfToken(absl::StrCat("limit:RotX"));

        auto usd_physics_limit_api = pxr::UsdPhysicsLimitAPI::Define(
            world_stage_,
            usd_physics_joint.GetPath().AppendProperty(joint_property));
        usd_physics_limit_api.GetLowAttr().Set<float>(
            mjmodel_->jnt_range[2 * mujoco_joint_index]);
        usd_physics_limit_api.GetHighAttr().Set<float>(
            mjmodel_->jnt_range[2 * mujoco_joint_index + 1]);
        break;
      }
      case mjJNT_SLIDE: {
        auto joint_property =
            pxr::TfToken(absl::StrCat("limit:TransX"));

        auto usd_physics_limit_api = pxr::UsdPhysicsLimitAPI::Define(
            world_stage_,
            usd_physics_joint.GetPath().AppendProperty(joint_property));
        usd_physics_limit_api.GetLowAttr().Set<float>(
            mjmodel_->jnt_range[2 * mujoco_joint_index]);
        usd_physics_limit_api.GetHighAttr().Set<float>(
            mjmodel_->jnt_range[2 * mujoco_joint_index + 1]);
        break;
      }
      default: {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unimplemented joint type: ",
            mjmodel_->jnt_type[mujoco_joint_index]));
      }
    }
  
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdPhysicsRevoluteJoint(
    const size_t mujoco_joint_index,
    const pxr::SdfPath& joint_sdf_path) {
  auto usd_revolute_joint =
      pxr::UsdPhysicsRevoluteJoint::Define(stage_, joint_sdf_path);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdPhysicsSphericalJoint(
    const size_t mujoco_joint_index,
    const pxr::SdfPath& joint_sdf_path) {
  auto usd_spherical_joint =
      pxr::UsdPhysicsSphericalJoint::Define(stage_, joint_sdf_path);
  return absl::OkStatus();
}

absl::Status MujocoToUsd::InitializeUsdJoints(
    const size_t mujoco_body_index,
    const pxr::SdfPath& parent_sdf_path) {
  for (int mujoco_joint_index = mjmodel_->body_jntadr[mujoco_body_index];
       mujoco_joint_index < (mjmodel_->body_jntadr[mujoco_body_index] +
                             mjmodel_->body_jntnum[mujoco_body_index]);
       ++mujoco_joint_index) {
    // Create the joint Scope to contain joints. If one exists it will reuse
    // it.
    auto joint_scope_path = parent_sdf_path.AppendPath(
        pxr::SdfPath(kJointScope));
    pxr::UsdGeomScope::Define(stage_, joint_scope_path);

    // Create the joint.
    auto mujoco_joint_name = GetUsdName(
        mjmodel_->name_jntadr[mujoco_joint_index], kJoint,
        mujoco_joint_index);
    auto joint_sdf_path =
        parent_sdf_path.AppendPath(pxr::SdfPath(mujoco_joint_name));

    // Define the joint.
    switch (mjmodel_->jnt_type[mujoco_joint_index]) {
      case mjJNT_FREE: {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unimplemented joint type: ",
            mjmodel_->jnt_type[mujoco_joint_index]));
      }
      case mjJNT_BALL: {
        RETURN_IF_ERROR(InitializeUsdPhysicsSphericalJoint(
            mujoco_joint_index, joint_sdf_path));
        break;
      }
      case mjJNT_HINGE: {
        RETURN_IF_ERROR(InitializeUsdPhysicsRevoluteJoint(
            mujoco_joint_index, joint_sdf_path));
        break;
      }
      case mjJNT_SLIDE: {
        RETURN_IF_ERROR(InitializeUsdPhysicsPrismaticJoint(
            mujoco_joint_index, joint_sdf_path));
        break;
      }
      default: {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unimplemented joint type: ",
            mjmodel_->jnt_type[mujoco_joint_index]));
      }
    }
  }
}

absl::Status MujocoToUsd::InitializeUsdRigidBodyRecursive(
    const size_t mujoco_parent_body_index,
    const pxr::SdfPath& input_parent_sdf_path,
    const pxr::GfMatrix4d root_pose_parent_transform) {
  for (int mujoco_body_index = 0; mujoco_body_index < mjmodel_->nbody;
       ++mujoco_body_index) {
    if (mujoco_body_index == kWorldIndex) {
      continue;
    }

    if (mjmodel_->body_parentid[mujoco_body_index] !=
        mujoco_parent_body_index) {
      continue;
    }

    // Add the body.
    auto mujoco_body_name = GetUsdName(
        mjmodel_->name_bodyadr[mujoco_body_index], kBody, mujoco_body_index);

    pxr::GfMatrix4d root_pose_body_transform =
        root_pose_parent_transform *
        MujocoPosQuatToTransform(&mjmodel_->body_pos[3 * mujoco_body_index],
                                 &mjmodel_->body_quat[4 * mujoco_body_index]);

    auto parent_sdf_path = input_parent_sdf_path;
    auto body_sdf_path =
        parent_sdf_path.AppendPath(pxr::SdfPath(mujoco_body_name));

    // If the body's parent is the world, create another Xform to parent all
    // rigid bodies to allow posing the whole body.
    if (mjmodel_->body_parentid[mujoco_body_index] == kWorldIndex) {
      auto root_body_xform = pxr::UsdGeomXform::Define(stage_, body_sdf_path);
      RETURN_IF_ERROR(
          SetXformTransform(root_pose_body_transform, root_body_xform));
      parent_sdf_path = body_sdf_path;
      body_sdf_path = body_sdf_path.AppendPath(pxr::SdfPath(mujoco_body_name));
      root_pose_body_transform.SetIdentity();
    }

    auto body_xform = pxr::UsdGeomXform::Define(stage_, body_sdf_path);
    RETURN_IF_ERROR(SetXformTransform(root_pose_body_transform, body_xform));
    pxr::UsdPhysicsRigidBodyAPI::Apply(body_xform.GetPrim());

    if (mjmodel_->body_mass[mujoco_body_index] > 0) {
      auto mass_api = pxr::UsdPhysicsMassAPI::Apply(body_xform.GetPrim());
      mass_api.GetMassAttr().Set<float>(
          mjmodel_->body_mass[mujoco_body_index]);
    }

    // Add the geoms to the body.
    RETURN_IF_ERROR(InitializeUsdGeoms(mujoco_body_index, body_sdf_path));

    if (mujoco_parent_body_index != kWorldIndex) {
      // It's a multibody, so add the appropriate joint linking this body to the
      // parent.
      RETURN_IF_ERROR(InitializeUsdJoints(mujoco_body_index, parent_sdf_path));
    }

    // Rigid Bodies can not be nested, so continue using the root parent SDF
    // path. The base links in USD are defined as rigid bodies that do not have
    // a joint connecting them to a parent.
    RETURN_IF_ERROR(InitializeUsdRigidBodyRecursive(
        mujoco_body_index, parent_sdf_path, root_pose_body_transform));
  }
  return absl::OkStatus();
}

absl::Status MujocoToUsd::AddToUsdStageInternal(
    const pxr::SdfPath& path) {
  pxr::SdfPath root_sdf_path = pxr::SdfPath(kRootPath);
  pxr::GfMatrix4d identity;
  identity.SetIdentity();
  RETURN_IF_ERROR(
      InitializeUsdRigidBodyRecursive(kWorldIndex, root_sdf_path, identity));
  return absl::OkStatus();
}

absl::Status MujocoToUsd::AddToUsdStage(
    const mjModel* mjmodel, const mjData* mjdata, const pxr::SdfPath& path,
    pxr::UsdStageRefPtr* output_stage) {
  MujocoToUsd mujoco_to_usd(mjmodel, mjdata, *output_stage);
  return mujoco_to_usd.AddToUsdStageInternal(path);
}

absl::Status MujocoToUsd::AddToSdfLayer(
    const mjModel* mjmodel, const mjData* mjdata, pxr::SdfLayer* output_layer) {
  // Create an internal UsdStage with a root layer.
  auto temp_layer = pxr::SdfLayer::CreateAnonymous("unused.usda");
  auto stage = pxr::UsdStage::Open(temp_layer);
  auto root_sdf_path = pxr::SdfPath("/root");
  auto root_xform = pxr::UsdGeomXform::Define(stage, root_sdf_path);
  stage->SetDefaultPrim(root_xform.GetPrim());

  RETURN_IF_ERROR(
      AddToUsdStage(mjmodel, mjdata, root_sdf_path, &stage));

  output_layer->TransferContent(temp_layer);

  return absl::OkStatus();
}

}  // namespace mujoco
