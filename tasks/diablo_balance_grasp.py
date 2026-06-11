"""
DiabloBalanceGrasp
==================
Simultaneous wheeled-balance and arm-grasping task for the DIABLO robot.

Flow:
  Phase 0  BALANCE_APPROACH  –  robot stabilises its inverted-pendulum body and
                                 drives toward the table object.
  Phase 1  GRASP             –  robot keeps balance while the arm reaches,
                                 grasps and lifts the object.

The transition from Phase 0 → Phase 1 is latched once the robot has maintained
|pitch| < balanced_pitch_threshold for balance_min_steps consecutive steps AND
is within grasp_approach_dist of the object handle.

Action space (9):
  [0]   height command   → IK leg height  [h_min, h_max] m
  [1]   pitch  command   → IK leg pitch   [-p_max, +p_max] rad  (balance lean)
  [2]   left  wheel vel  → [-wheel_vel_max, +wheel_vel_max] rad/s
  [3]   right wheel vel  → [-wheel_vel_max, +wheel_vel_max] rad/s
  [4‥7] right arm joints (4, delta from default)
  [8]   gripper open / close

Observation space (72):
  progress(1), base_lin_vel_body(3), base_ang_vel(3), base_quat(4),
  base_height_norm(1), leg_pos_norm(4), leg_vel(4), wheel_vel(2),
  arm_pos_norm(4), arm_vel(4), eef_pos(3), eef_rot(4),
  object_pos(3), object_rot(4), handle_pos(3),
  eef→handle(3), robot→handle(3), phase(1), prev_actions(9),
  platform_pos(3), rel_bottom_to_plat(3), object_one_hot(3)

Multi-object generalization:
  Three object types are loaded at startup. Envs are cyclically assigned to one
  object type and keep it for the entire training run. The one-hot vector in the
  observation tells the policy which object it is working with.
  Actor layout per env: diablo(0) table(1) object(2) platform(3)
"""

import math
import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate_inverse,
    axisangle2quat, to_torch, tensor_clamp,
)


# ── helper ────────────────────────────────────────────────────────────────────

def _quat_to_pitch_roll(q: torch.Tensor):
    """Return (pitch, roll) scalars per env from quaternion (x,y,z,w)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp  = torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0)
    pitch = torch.asin(sinp)
    return pitch, roll


# ══════════════════════════════════════════════════════════════════════════════

class DiabloBalanceGrasp(VecTask):

    PHASE_APPROACH = 0
    PHASE_GRASP    = 1
    PHASE_PLACE    = 2
    PHASE_RELEASE  = 3

    # Physical constants
    R_WHEEL = 0.08
    L1 = L2 = 0.14
    L_MAX   = 0.28
    BODY_OFFSET = 0.16

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        env_cfg  = cfg["env"]

        self.max_episode_length    = env_cfg["episodeLength"]
        self.action_scale          = env_cfg.get("actionScale",  1.5)
        self.wheel_vel_max         = env_cfg.get("wheelVelMax",  5.0)

        # Leg IK ranges
        self.h_mid   = 0.29;  self.h_range = 0.06
        self.h_min   = 0.23;  self.h_max   = 0.35
        self.h_grasp_mid = 0.48   # 0.48 m, slightly below max for better balance
        self.p_max   = 0.20

        # Phase thresholds
        self.fall_pitch_thr      = env_cfg.get("fallPitchThreshold",     0.50)
        self.balanced_pitch_thr  = env_cfg.get("balancedPitchThreshold", 0.18)
        self.balance_min_steps   = env_cfg.get("balanceMinSteps",        1)
        self.grasp_approach_dist = env_cfg.get("graspApproachDist",      0.25)

        # Reward scales
        rwd = env_cfg["rewards"]
        self.balance_scale         = rwd.get("balanceScale",       3.0)
        self.alive_bonus           = rwd.get("aliveBonus",         2.0)
        self.height_scale          = rwd.get("heightScale",        2.0)
        self.approach_scale        = rwd.get("approachScale",      5.0)
        self.dist_scale            = rwd.get("distRewardScale",    5.0)
        self.rot_scale             = rwd.get("rotRewardScale",     2.0)
        self.grasp_scale           = rwd.get("graspRewardScale",   5.0)
        self.lift_scale            = rwd.get("liftRewardScale",    15.0)
        self.success_bonus         = rwd.get("successBonus",       500.0)
        self.fall_penalty          = rwd.get("fallPenalty",        100.0)
        self.action_penalty_scale  = rwd.get("actionPenaltyScale", 0.002)

        # Multi-object config
        self.object_configs = env_cfg["objects"]
        self.object_assets  = []   # filled in _create_envs

        # DOF name lists
        self.right_arm_names = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
        self.right_gripper_names = [
            "r_index_base", "r_index_middle", "r_index_tip",
            "r_mid_base",   "r_mid_middle",   "r_mid_tip",
            "r_thumb_base", "r_thumb_middle",  "r_thumb_tip",
        ]
        self.dof_indices = {}

        n   = env_cfg["numEnvs"]
        dev = sim_device

        # Per-env buffers
        self.phase_buf       = torch.zeros(n, dtype=torch.long,    device=dev)
        self.balance_timer   = torch.zeros(n, dtype=torch.float32, device=dev)
        self.actions         = torch.zeros((n, 9), dtype=torch.float32, device=dev)
        self.prev_actions    = torch.zeros((n, 9), dtype=torch.float32, device=dev)
        self.episode_success        = torch.zeros(n, dtype=torch.bool, device=dev)
        self.partial_success_buf    = torch.zeros(n, dtype=torch.bool, device=dev)
        self.phase2_achieved_buf    = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tight_contact_buf      = torch.zeros(n, dtype=torch.bool, device=dev)

        # Per-env object metadata (set at _create_envs, fixed for each env)
        self.object_id           = torch.zeros(n, dtype=torch.long,    device=dev)
        self.object_heights      = torch.zeros(n, dtype=torch.float32, device=dev)
        self.object_half_heights = torch.zeros(n, dtype=torch.float32, device=dev)

        # Platform
        self.platform_pos_tensor = torch.zeros((n, 3), dtype=torch.float32, device=dev)

        # Statistics (first round of num_envs episodes is discarded as warmup)
        self.total_attempts          = 0
        self.total_successes         = 0
        self.total_partial_successes = 0
        self.total_phase2_achieved   = 0
        self.total_tight_contacts    = 0
        self._stat_warmup_done       = False

        self.up_axis     = "z"
        self.up_axis_idx = 2
        self.states      = {}

        self.cfg["env"]["numActions"]      = 9
        self.cfg["env"]["numObservations"] = 72

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if not self.headless:
            self.gym.viewer_camera_look_at(
                self.viewer, None,
                gymapi.Vec3(0.5, -2.5, 1.5),
                gymapi.Vec3(0.0,  0.0, 0.35),
            )

        self._acquire_tensors()
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        self.gripper_upper_limits = self.dof_upper_limits[self.dof_indices["right_gripper"]]
        self.gripper_lower_limits = self.dof_lower_limits[self.dof_indices["right_gripper"]]

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_tensors()

    # ── sim creation ──────────────────────────────────────────────────────────

    def create_sim(self):
        self.sim_params.up_axis   = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0.0
        self.sim_params.gravity.y = 0.0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, pp)

    # ── env / asset creation ──────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3( spacing,  spacing, spacing)

        asset_root  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        diablo_file = self.cfg["env"]["asset"]["assetFileNamediablo"]

        # ── Robot ─────────────────────────────────────────────────────────
        ao = gymapi.AssetOptions()
        ao.fix_base_link           = False
        ao.collapse_fixed_joints   = False
        ao.disable_gravity         = False
        ao.armature                = 0.01
        ao.default_dof_drive_mode  = gymapi.DOF_MODE_POS
        ao.use_mesh_materials      = False
        ao.flip_visual_attachments = False
        print(f"Loading {diablo_file} …")
        diablo_asset = self.gym.load_asset(self.sim, asset_root, diablo_file, ao)

        self.num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        self.num_diablo_shapes = self.gym.get_asset_rigid_shape_count(diablo_asset)
        self.num_dofs          = self.gym.get_asset_dof_count(diablo_asset)
        print(f"Total DOFs: {self.num_dofs}")

        dof_dict = self.gym.get_asset_dof_dict(diablo_asset)
        self.dof_indices["right_arm"]     = [dof_dict[n] for n in self.right_arm_names]
        self.dof_indices["right_gripper"] = [dof_dict[n] for n in self.right_gripper_names]
        self.dof_indices["legs"]          = [
            dof_dict["left_fake_hip_joint"],  dof_dict["left_fake_knee_joint"],
            dof_dict["right_fake_hip_joint"], dof_dict["right_fake_knee_joint"],
        ]
        self.dof_indices["wheel_l"] = dof_dict["left_wheel_joint"]
        self.dof_indices["wheel_r"] = dof_dict["right_wheel_joint"]

        # DOF properties
        dof_props = self.gym.get_asset_dof_properties(diablo_asset)
        stiffness_map = {
            "head_pan":  1e6, "head_tilt": 1e6,
            "l_sho_pitch": 1e6, "l_sho_roll": 1e6, "l_el": 1e6, "l_wrist": 1e6,
            "r_sho_pitch": 15.0, "r_sho_roll": 20.0, "r_el": 13.82, "r_wrist": 4.55,
            "left_fake_hip_joint":  400.0, "left_fake_knee_joint":  400.0,
            "right_fake_hip_joint": 400.0, "right_fake_knee_joint": 400.0,
            "left_wheel_joint":  0.0, "right_wheel_joint": 0.0,
        }
        damping_map = {
            "head_pan": 100., "head_tilt": 100.,
            "l_sho_pitch": 100., "l_sho_roll": 100., "l_el": 100., "l_wrist": 100.,
            "r_sho_pitch": 0.3, "r_sho_roll": 0.4, "r_el": 0.1, "r_wrist": 0.002,
            "left_fake_hip_joint":  50., "left_fake_knee_joint":  50.,
            "right_fake_hip_joint": 50., "right_fake_knee_joint": 50.,
            "left_wheel_joint":  20., "right_wheel_joint": 20.,
        }

        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            name = self.gym.get_asset_dof_name(diablo_asset, i)
            dof_props["driveMode"][i] = (gymapi.DOF_MODE_VEL
                                         if name in ("left_wheel_joint", "right_wheel_joint")
                                         else gymapi.DOF_MODE_POS)
            dof_props["stiffness"][i] = stiffness_map.get(name, 700.0)
            dof_props["damping"][i]   = damping_map.get(name, 50.0)
            if name in self.right_gripper_names:
                dof_props["stiffness"][i] = 20.0
                dof_props["damping"][i]   = 0.5
                dof_props["effort"][i]    = 0.2
            self.dof_lower_limits.append(float(dof_props["lower"][i]))
            self.dof_upper_limits.append(float(dof_props["upper"][i]))

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)

        # ── Table ─────────────────────────────────────────────────────────
        tao = gymapi.AssetOptions(); tao.fix_base_link = True
        table_asset  = self.gym.create_box(self.sim, 0.3, 0.5, 0.01, tao)
        n_tab_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        n_tab_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

        TABLE_POS         = gymapi.Vec3(0.25, 0.0, 0.55)
        self.table_surface_z = TABLE_POS.z + 0.005  # 0.555

        # ── Load all object assets ─────────────────────────────────────────
        oao = gymapi.AssetOptions()
        oao.fix_base_link      = False
        oao.use_mesh_materials = True
        oao.mesh_normal_mode   = gymapi.COMPUTE_PER_VERTEX
        oao.override_com       = True
        oao.override_inertia   = True
        oao.vhacd_enabled      = True
        oao.vhacd_params       = gymapi.VhacdParams()
        oao.vhacd_params.resolution = 1000

        for obj_cfg in self.object_configs:
            asset = self.gym.load_asset(self.sim, asset_root, obj_cfg["objectRoot"], oao)
            self.object_assets.append(asset)
            print(f"Loaded object: {obj_cfg['name']} ({obj_cfg['objectRoot']})")

        # Use max body/shape count across object types for aggregate sizing
        n_obj_bodies = max(self.gym.get_asset_rigid_body_count(a) for a in self.object_assets)
        n_obj_shapes = max(self.gym.get_asset_rigid_shape_count(a) for a in self.object_assets)

        self.initial_object_z = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # ── Platform ──────────────────────────────────────────────────────
        pao = gymapi.AssetOptions(); pao.fix_base_link = True
        platform_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.01, pao)
        n_plat_bodies  = self.gym.get_asset_rigid_body_count(platform_asset)
        n_plat_shapes  = self.gym.get_asset_rigid_shape_count(platform_asset)

        total_bodies = self.num_diablo_bodies + n_tab_bodies + n_obj_bodies + n_plat_bodies
        total_shapes = self.num_diablo_shapes + n_tab_shapes + n_obj_shapes + n_plat_shapes

        # ── Static poses ──────────────────────────────────────────────────
        table_pose        = gymapi.Transform()
        table_pose.p      = TABLE_POS
        table_pose.r      = gymapi.Quat(0, 0, 0, 1)

        robot_init_pose   = gymapi.Transform()
        robot_init_pose.p = gymapi.Vec3(-0.55, 0.0, 0.29)
        robot_init_pose.r = gymapi.Quat(0, 0, 0, 1)

        obj_init_pose     = gymapi.Transform()
        obj_init_pose.r   = gymapi.Quat(0, 0, 1, 0)

        platform_init_pose   = gymapi.Transform()
        platform_init_pose.p = gymapi.Vec3(0.0, 0.0, -1.0)
        platform_init_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.envs          = []
        self.actor_handles = []
        self.table_handles = []
        self.obj_handles   = []
        self.plat_handles  = []

        # Per-env global rigid-body indices (accounts for mixed body counts per env)
        self.eef_global_indices           = []
        self.handle_target_global_indices = []
        _body_offset = 0

        _num_obj_types = len(self.object_assets)
        _start_offset  = np.random.randint(_num_obj_types)

        # Eval mode: force a single object type across all envs
        _forced_obj = None
        _eval_target = self.cfg["env"].get("eval_object_name", "")
        if _eval_target:
            for k, oc in enumerate(self.object_configs):
                if oc.get("name", "") == _eval_target:
                    _forced_obj = k
                    print(f"[Eval] Forced object type: {_eval_target} (idx {k})")
                    break

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env, total_bodies, total_shapes, True)

            # Robot — actor index 0
            diablo = self.gym.create_actor(env, diablo_asset, robot_init_pose, "diablo", i, 0, 0)
            self.gym.set_actor_dof_properties(env, diablo, dof_props)
            self.actor_handles.append(diablo)

            # Table — actor index 1
            table = self.gym.create_actor(env, table_asset, table_pose, "table", i, 1, 0)
            self.table_handles.append(table)

            # Object — actor index 2, cyclic type assignment
            obj_idx = _forced_obj if _forced_obj is not None else (i + _start_offset) % _num_obj_types
            obj_cfg = self.object_configs[obj_idx]
            self.object_id[i]           = obj_idx
            self.object_heights[i]      = obj_cfg["objectHeight"]
            self.object_half_heights[i] = obj_cfg["objectHalfHeight"]
            self.initial_object_z[i]    = self.table_surface_z + obj_cfg["objectHalfHeight"] + 0.005

            obj_init_pose.p = gymapi.Vec3(
                TABLE_POS.x,
                float(np.random.uniform(-0.15, 0.15)),
                float(self.initial_object_z[i]))
            obj_actor = self.gym.create_actor(
                env, self.object_assets[obj_idx], obj_init_pose, "object", i, 0, 0)
            self.obj_handles.append(obj_actor)

            # Platform — actor index 3
            plat_actor = self.gym.create_actor(
                env, platform_asset, platform_init_pose, "platform", i, 0, 0)
            self.gym.set_rigid_body_color(
                env, plat_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.1))
            self.plat_handles.append(plat_actor)

            self.gym.end_aggregate(env)
            self.envs.append(env)

            # Record global rigid-body indices for this env
            _env_bodies = self.gym.get_env_rigid_body_count(env)
            _eef_local  = self.gym.find_actor_rigid_body_index(
                env, diablo,     "panda_grip_site", gymapi.DOMAIN_ENV)
            _hdl_local  = self.gym.find_actor_rigid_body_index(
                env, obj_actor,  "handle_target",   gymapi.DOMAIN_ENV)
            self.eef_global_indices.append(_body_offset + _eef_local)
            self.handle_target_global_indices.append(_body_offset + _hdl_local)
            _body_offset += _env_bodies

        # Actor strides: 4 actors per env
        self.diablo_actor_ids = torch.arange(0, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.table_actor_ids  = torch.arange(1, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.obj_actor_ids    = torch.arange(2, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.plat_actor_ids   = torch.arange(3, num_envs * 4, 4, dtype=torch.int32, device=self.device)

        self.eef_global_indices           = to_torch(self.eef_global_indices,
                                                      dtype=torch.long, device=self.device)
        self.handle_target_global_indices = to_torch(self.handle_target_global_indices,
                                                      dtype=torch.long, device=self.device)
        print(f"EEF global[0]: {self.eef_global_indices[0]}  |  "
              f"Handle global[0]: {self.handle_target_global_indices[0]}")

        # Print object type distribution
        for k, oc in enumerate(self.object_configs):
            count = (self.object_id == k).sum().item()
            print(f"  Object '{oc['name']}': {count} envs")

        self.init_data()

    # ── data initialisation ───────────────────────────────────────────────────

    def init_data(self):
        self._acquire_tensors()

        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        arm_def = np.radians([0, -60, -80, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_def):
            self.default_dof_pos[idx] = float(val)
        self.default_dof_pos[self.dof_indices["right_gripper"]] = \
            self.dof_lower_limits[self.dof_indices["right_gripper"]]

        self.arm_default_grasp = self.default_dof_pos.clone()
        arm_grasp_def = np.radians([0, -10, -55, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_grasp_def):
            self.arm_default_grasp[idx] = float(val)

    def _acquire_tensors(self):
        self.root_state_tensor       = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor        = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.root_state       = gymtorch.wrap_tensor(self.root_state_tensor)
        self.dof_state        = gymtorch.wrap_tensor(self.dof_state_tensor)
        # flat rigid-body state [total_bodies_across_all_envs, 13]
        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state_tensor)

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_states()

    def _update_states(self):
        base_quat     = self.root_state[self.diablo_actor_ids, 3:7]
        world_lin_vel = self.root_state[self.diablo_actor_ids, 7:10]
        world_ang_vel = self.root_state[self.diablo_actor_ids, 10:13]
        body_lin_vel  = quat_rotate_inverse(base_quat, world_lin_vel)

        # Use pre-computed global indices — safe for mixed body-count object types
        self.states.update({
            "base_pos":          self.root_state[self.diablo_actor_ids, :3],
            "base_quat":         base_quat,
            "base_lin_vel":      body_lin_vel,
            "base_ang_vel":      world_ang_vel,
            "eef_pos":           self.rigid_body_state[self.eef_global_indices, :3],
            "eef_rot":           self.rigid_body_state[self.eef_global_indices, 3:7],
            "object_pos":        self.root_state[self.obj_actor_ids, :3],
            "object_rot":        self.root_state[self.obj_actor_ids, 3:7],
            "handle_target_pos": self.rigid_body_state[self.handle_target_global_indices, :3],
            "handle_target_rot": self.rigid_body_state[self.handle_target_global_indices, 3:7],
            "dof_pos":           self.dof_pos,
            "dof_vel":           self.dof_vel,
        })

    # ── observations ──────────────────────────────────────────────────────────

    def compute_observations(self):
        self._refresh_tensors()

        base_pos   = self.states["base_pos"]
        base_quat  = self.states["base_quat"]
        base_lv    = self.states["base_lin_vel"]
        base_av    = self.states["base_ang_vel"]
        eef_pos    = self.states["eef_pos"]
        eef_rot    = self.states["eef_rot"]
        obj_pos    = self.states["object_pos"]
        obj_rot    = self.states["object_rot"]
        handle_pos = self.states["handle_target_pos"]
        dof_pos    = self.states["dof_pos"]
        dof_vel    = self.states["dof_vel"]

        delta        = self.dof_upper_limits - self.dof_lower_limits + 1e-6
        dof_pos_norm = (dof_pos - self.dof_lower_limits) / delta

        leg_ids = self.dof_indices["legs"]
        arm_ids = self.dof_indices["right_arm"]
        wl_id   = self.dof_indices["wheel_l"]
        wr_id   = self.dof_indices["wheel_r"]

        leg_pos_norm = dof_pos_norm[:, leg_ids]
        leg_vel      = dof_vel[:, leg_ids]
        wheel_vel    = torch.stack([dof_vel[:, wl_id], dof_vel[:, wr_id]], dim=-1)
        arm_pos_norm = dof_pos_norm[:, arm_ids]
        arm_vel      = dof_vel[:, arm_ids]

        height_norm     = ((base_pos[:, 2] - (self.h_mid + self.BODY_OFFSET)) / self.h_range).unsqueeze(-1)
        eef_to_handle   = handle_pos - eef_pos
        robot_to_handle = handle_pos - base_pos
        phase_obs       = self.phase_buf.float().unsqueeze(-1)

        obj_up_local   = torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1)
        world_obj_up   = quat_apply(obj_rot, obj_up_local)
        obj_bottom_pos = obj_pos - self.object_half_heights.unsqueeze(-1) * world_obj_up
        rel_bottom_to_plat = self.platform_pos_tensor - obj_bottom_pos

        # One-hot encoding of object type
        one_hot = torch.zeros(self.num_envs, 3, device=self.device)
        one_hot.scatter_(1, self.object_id.unsqueeze(1), 1.0)

        o = self.obs_buf
        idx = 0
        def w(t, n):
            nonlocal idx
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            o[:, idx:idx + n] = t
            idx += n

        w(self.progress_buf.float() / self.max_episode_length, 1)  # 1
        w(base_lv,         3)   # 4
        w(base_av,         3)   # 7
        w(base_quat,       4)   # 11
        w(height_norm,     1)   # 12
        w(leg_pos_norm,    4)   # 16
        w(leg_vel,         4)   # 20
        w(wheel_vel,       2)   # 22
        w(arm_pos_norm,    4)   # 26
        w(arm_vel,         4)   # 30
        w(eef_pos,         3)   # 33
        w(eef_rot,         4)   # 37
        w(obj_pos,         3)   # 40
        w(obj_rot,         4)   # 44
        w(handle_pos,      3)   # 47
        w(eef_to_handle,   3)   # 50
        w(robot_to_handle, 3)   # 53
        w(phase_obs,       1)   # 54
        w(self.prev_actions, 9) # 63
        w(self.platform_pos_tensor, 3) # 66
        w(rel_bottom_to_plat, 3)       # 69
        w(one_hot,            3)       # 72

        assert idx == 72, f"obs mismatch {idx}"
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.obs_buf

    # ── reward ────────────────────────────────────────────────────────────────

    def compute_reward(self):
        (
            self.rew_buf[:], self.reset_buf[:],
            bal_rew, alive_rew, height_rew, approach_rew,
            dist_rew, rot_rew, grasp_rew, lift_rew, trans_rew, place_rew,
            release_rew, retreat_rew, orient_rew,
            grasp_bal_bonus, success_rew, total_pen,
            smooth_pen, energy_pen, arm_motion_pen, fall_pen, braking_pen, plat_collision_pen,
            is_success, is_fallen, is_balanced, is_grasping, is_lifted,
        ) = compute_balance_grasp_reward(
            reset_buf          = self.reset_buf,
            progress_buf       = self.progress_buf,
            actions            = self.actions,
            prev_actions       = self.prev_actions,
            base_pos           = self.states["base_pos"],
            base_quat          = self.states["base_quat"],
            base_ang_vel       = self.states["base_ang_vel"],
            eef_pos            = self.states["eef_pos"],
            eef_rot            = self.states["eef_rot"],
            handle_pos         = self.states["handle_target_pos"],
            handle_rot         = self.states["handle_target_rot"],
            object_pos         = self.states["object_pos"],
            object_rot         = self.states["object_rot"],
            initial_object_z   = self.initial_object_z,
            platform_pos       = self.platform_pos_tensor,
            phase_buf          = self.phase_buf,
            num_envs           = self.num_envs,
            max_episode_length = self.max_episode_length,
            h_mid              = self.h_mid + self.BODY_OFFSET,
            h_grasp_mid        = self.h_grasp_mid,
            fall_pitch_thr     = self.fall_pitch_thr,
            object_half_height = self.object_half_heights,   # per-env tensor
            balance_scale      = self.balance_scale,
            alive_bonus        = self.alive_bonus,
            height_scale       = self.height_scale,
            approach_scale     = self.approach_scale,
            dist_scale         = self.dist_scale,
            rot_scale          = self.rot_scale,
            grasp_scale        = self.grasp_scale,
            lift_scale         = self.lift_scale,
            success_bonus      = self.success_bonus,
            fall_penalty       = self.fall_penalty,
            action_penalty_scale = self.action_penalty_scale,
        )

        self.extras["rewards/balance"]       = bal_rew.mean()
        self.extras["rewards/alive"]         = alive_rew.mean()
        self.extras["rewards/height"]        = height_rew.mean()
        self.extras["rewards/approach"]      = approach_rew.mean()
        self.extras["rewards/dist"]          = dist_rew.mean()
        self.extras["rewards/rot"]           = rot_rew.mean()
        self.extras["rewards/grasp"]         = grasp_rew.mean()
        self.extras["rewards/lift"]          = lift_rew.mean()
        self.extras["rewards/transport"]     = trans_rew.mean()
        self.extras["rewards/placement"]     = place_rew.mean()
        self.extras["rewards/release"]       = release_rew.mean()
        self.extras["rewards/retreat"]       = retreat_rew.mean()
        self.extras["rewards/orient"]        = orient_rew.mean()
        self.extras["rewards/grasp_balance"] = grasp_bal_bonus.mean()
        self.extras["rewards/success"]       = success_rew.mean()
        self.extras["rewards/pen_total"]     = total_pen.mean()
        
        # ── Diagnostic Metrics ─────────────────────────────────────────────
        # 1. Object relative height to platform surface (Phase 3 focus)
        plat_surface_z = self.platform_pos_tensor[:, 2] + 0.005
        obj_up = quat_apply(self.states["object_rot"], torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1))
        obj_bottom_z = self.states["object_pos"][:, 2] - self.object_half_heights * obj_up[:, 2]
        dist_z_to_plat = obj_bottom_z - plat_surface_z
        self.extras["metrics/obj_to_plat_z"] = dist_z_to_plat.mean()
        
        # 2. EEF to Handle distance (Retreat focus)
        eef_to_handle_dist = torch.norm(self.states["eef_pos"] - self.states["handle_target_pos"], dim=-1)
        self.extras["metrics/eef_to_handle_dist"] = eef_to_handle_dist.mean()
        
        self.extras["rewards/pen_plat_collision"] = plat_collision_pen.mean()
        
        # 3. Horizontal alignment with platform
        dist_xy_to_plat = torch.norm(self.states["object_pos"][:, :2] - self.platform_pos_tensor[:, :2], dim=-1)
        self.extras["metrics/obj_to_plat_xy"] = dist_xy_to_plat.mean()

        # ── Partial success latch ──────────────────────────────────────────
        # 定義：Phase 2/3 + XY ≤ 5cm + 物體底部距平台面 ≤ 1cm（含輕微接觸）
        partial_cond = (self.phase_buf >= self.PHASE_PLACE) & \
                       (dist_xy_to_plat < 0.05) & \
                       (dist_z_to_plat > -0.02) & (dist_z_to_plat < 0.01)
        self.partial_success_buf |= partial_cond

        # ── Phase 2 entry latch ───────────────────────────────────────────
        self.phase2_achieved_buf |= (self.phase_buf >= self.PHASE_PLACE)

        # ── Tight contact latch (xy<5cm & |dz|<2cm) ──────────────────────
        tight_cond = (dist_xy_to_plat < 0.05) & (torch.abs(dist_z_to_plat) < 0.02)
        self.tight_contact_buf |= tight_cond

        # 4. Phase 3 specific metrics (Release & Retreat)
        phase3_mask = (self.phase_buf == self.PHASE_RELEASE)
        if phase3_mask.any():
            self.extras["metrics/phase3_obj_to_plat_z_gap"] = dist_z_to_plat[phase3_mask].mean()
            self.extras["metrics/phase3_eef_to_handle_dist"] = eef_to_handle_dist[phase3_mask].mean()
            self.extras["metrics/phase3_obj_to_plat_xy"] = dist_xy_to_plat[phase3_mask].mean()
        else:
            self.extras["metrics/phase3_obj_to_plat_z_gap"] = torch.tensor(0.0, device=self.device)
            self.extras["metrics/phase3_eef_to_handle_dist"] = torch.tensor(0.0, device=self.device)
            self.extras["metrics/phase3_obj_to_plat_xy"] = torch.tensor(0.0, device=self.device)

        # Phase 2 metrics (Placement preparation)
        phase2_mask = (self.phase_buf == self.PHASE_PLACE)
        if phase2_mask.any():
            self.extras["metrics/phase2_obj_to_plat_xy"] = dist_xy_to_plat[phase2_mask].mean()
            self.extras["metrics/phase2_obj_to_plat_z_gap"] = dist_z_to_plat[phase2_mask].mean()
        else:
            self.extras["metrics/phase2_obj_to_plat_xy"] = torch.tensor(0.0, device=self.device)
            self.extras["metrics/phase2_obj_to_plat_z_gap"] = torch.tensor(0.0, device=self.device)

        # 5. Phase percentages
        for p in range(4):
            self.extras[f"metrics/phase_{p}_rate"] = (self.phase_buf == p).float().mean()

        self.extras["metrics/fall_rate"]     = is_fallen.float().mean()
        self.extras["metrics/balanced_rate"] = is_balanced.float().mean()
        self.extras["metrics/grasping_rate"] = is_grasping.float().mean()
        self.extras["metrics/lifted_rate"]   = is_lifted.float().mean()
        self.extras["metrics/phase"]         = self.phase_buf.float().mean()
        self.extras["metrics/balance_timer"] = self.balance_timer.mean()

        # Per-object type success diagnostics
        for k, oc in enumerate(self.object_configs):
            mask = (self.object_id == k)
            if mask.any():
                self.extras[f"metrics/success_{oc['name']}"] = is_success[mask].float().mean()

        self.episode_success |= is_success

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset_idx(self, env_ids):
        num = len(env_ids)
        R, l1, l2 = self.R_WHEEL, self.L1, self.L2

        # Random initial height and small pitch perturbation
        h_tgt = 0.29 + (torch.rand(num, device=self.device) - 0.5) * 0.04
        p_tgt = (torch.rand(num, device=self.device) - 0.5) * 0.08
        y_tgt = (torch.rand(num, device=self.device) - 0.5) * 0.60

        # IK → leg angles
        L0        = torch.clamp((h_tgt - R) / torch.cos(p_tgt), 0.01, l1 + l2)
        cos_alpha = torch.clamp((l1**2 + l2**2 - L0**2) / (2*l1*l2), -1.0, 1.0)
        tgt_knee  = math.pi - torch.acos(cos_alpha)
        cos_beta  = torch.clamp(L0 / (2*l1), -1.0, 1.0)
        tgt_hip   = -p_tgt - torch.acos(cos_beta)

        leg_ids = self.dof_indices["legs"]
        self.dof_pos[env_ids, leg_ids[0]] = tgt_hip
        self.dof_pos[env_ids, leg_ids[1]] = tgt_knee
        self.dof_pos[env_ids, leg_ids[2]] = tgt_hip
        self.dof_pos[env_ids, leg_ids[3]] = tgt_knee

        arm_grip = self.dof_indices["right_arm"] + self.dof_indices["right_gripper"]
        self.dof_pos[env_ids[:, None], arm_grip] = self.default_dof_pos[arm_grip]
        self.dof_vel[env_ids, :] = 0.0

        # Robot start position
        start_x = -(0.50 + torch.rand(num, device=self.device) * 0.20)
        start_y =  (torch.rand(num, device=self.device) - 0.5) * 0.20
        bx = L0 * torch.sin(p_tgt) * torch.cos(y_tgt) + start_x
        by = L0 * torch.sin(p_tgt) * torch.sin(y_tgt) + start_y

        self.root_state[self.diablo_actor_ids[env_ids], 0] = bx
        self.root_state[self.diablo_actor_ids[env_ids], 1] = by
        self.root_state[self.diablo_actor_ids[env_ids], 2] = h_tgt + self.BODY_OFFSET
        robot_quat = quat_from_euler_xyz(torch.zeros_like(p_tgt), p_tgt, y_tgt)
        self.root_state[self.diablo_actor_ids[env_ids], 3:7]  = robot_quat
        self.root_state[self.diablo_actor_ids[env_ids], 7:13] = 0.0

        # Object: reset to its fixed table position (type is fixed per env)
        obj_x = torch.clamp(0.25 + (torch.rand(num, device=self.device) - 0.5) * 0.14, 0.18, 0.32)
        obj_y = torch.clamp((torch.rand(num, device=self.device) - 0.5) * 0.20, -0.10, 0.10)
        self.root_state[self.obj_actor_ids[env_ids], 0] = obj_x
        self.root_state[self.obj_actor_ids[env_ids], 1] = obj_y
        self.root_state[self.obj_actor_ids[env_ids], 2] = self.initial_object_z[env_ids]
        aa = torch.zeros(num, 3, device=self.device)
        aa[:, 2] = math.pi + (torch.rand(num, device=self.device) - 0.5) * 0.5
        init_rot = torch.tensor([0., 0., 0., 1.], device=self.device).unsqueeze(0).repeat(num, 1)
        self.root_state[self.obj_actor_ids[env_ids], 3:7]  = quat_mul(axisangle2quat(aa), init_rot)
        self.root_state[self.obj_actor_ids[env_ids], 7:13] = 0.0

        # Apply DOF state
        d_ids = self.diablo_actor_ids[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(d_ids), len(d_ids))

        # Platform: hidden initially, placed relative to robot
        lp_x = 0.25 + (torch.rand(num, device=self.device) - 0.5) * 0.05
        lp_y = -0.22 + (torch.rand(num, device=self.device) - 0.5) * 0.05
        lp_z = torch.zeros(num, device=self.device)
        lp_pos = torch.stack([lp_x, lp_y, lp_z], dim=-1)
        wp_off = quat_apply(robot_quat, lp_pos)
        self.platform_pos_tensor[env_ids] = \
            torch.stack([bx, by, torch.zeros_like(bx)], dim=-1) + wp_off
        self.platform_pos_tensor[env_ids, 2] = -1.0

        self.root_state[self.plat_actor_ids[env_ids], :3]  = self.platform_pos_tensor[env_ids]
        self.root_state[self.plat_actor_ids[env_ids], 3:7] = \
            torch.tensor([0., 0., 0., 1.], device=self.device).repeat(num, 1)

        all_ids = torch.cat([
            self.diablo_actor_ids[env_ids],
            self.obj_actor_ids[env_ids],
            self.plat_actor_ids[env_ids],
        ]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(all_ids), len(all_ids))

        # Reset counters
        self.progress_buf[env_ids]       = 0
        self.reset_buf[env_ids]          = 0
        self.phase_buf[env_ids]          = self.PHASE_APPROACH
        self.balance_timer[env_ids]      = 0.0
        self.actions[env_ids]            = 0.0
        self.prev_actions[env_ids]       = 0.0
        self.episode_success[env_ids]    = False
        self.partial_success_buf[env_ids] = False
        self.phase2_achieved_buf[env_ids] = False
        self.tight_contact_buf[env_ids]   = False

    # ── physics step ──────────────────────────────────────────────────────────

    def pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        N  = self.num_envs
        pos_targets = self.default_dof_pos.unsqueeze(0).repeat(N, 1).clone()
        vel_targets = torch.zeros(N, self.num_dofs, device=self.device)

        # ── Leg IK ────────────────────────────────────────────────────────
        h_cmd = self.actions[:, 0]
        p_cmd = self.actions[:, 1]
        h_tgt = torch.clamp(self.h_mid + h_cmd * self.h_range, self.h_min, self.h_max)
        p_tgt = torch.clamp(p_cmd * self.p_max, -self.p_max, self.p_max)
        L0    = torch.clamp((h_tgt - self.R_WHEEL) / torch.cos(p_tgt), 0.01, self.L_MAX)
        c_a   = torch.clamp((self.L1**2 + self.L2**2 - L0**2) / (2*self.L1*self.L2), -1.0, 1.0)
        knee  = math.pi - torch.acos(c_a)
        c_b   = torch.clamp(L0 / (2*self.L1), -1.0, 1.0)
        hip   = -p_tgt - torch.acos(c_b)

        leg_ids = self.dof_indices["legs"]
        pos_targets[:, leg_ids[0]] = hip
        pos_targets[:, leg_ids[1]] = knee
        pos_targets[:, leg_ids[2]] = hip
        pos_targets[:, leg_ids[3]] = knee

        # ── Wheels ────────────────────────────────────────────────────────
        vel_targets[:, self.dof_indices["wheel_l"]] = self.actions[:, 2] * self.wheel_vel_max
        vel_targets[:, self.dof_indices["wheel_r"]] = self.actions[:, 3] * self.wheel_vel_max

        # ── Arm ───────────────────────────────────────────────────────────
        arm_ids = self.dof_indices["right_arm"]
        # Use arm_default_grasp for Phase 1, 2, 3
        phase_active_mask = (self.phase_buf >= self.PHASE_GRASP).unsqueeze(1).expand(-1, 4)
        arm_base = torch.where(
            phase_active_mask,
            self.arm_default_grasp[arm_ids].unsqueeze(0).expand(N, -1),
            self.default_dof_pos[arm_ids].unsqueeze(0).expand(N, -1),
        )
        arm_tgt = self.action_scale * self.actions[:, 4:8] + arm_base
        pos_targets[:, arm_ids] = arm_tgt

        phase0_mask = (self.phase_buf == self.PHASE_APPROACH).unsqueeze(1).expand(-1, 4)
        pos_targets[:, arm_ids] = torch.where(
            phase0_mask,
            0.6 * arm_tgt + 0.4 * self.default_dof_pos[arm_ids],
            arm_tgt,
        )

        # ── Gripper ───────────────────────────────────────────────────────
        eef_pos    = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        dist       = torch.linalg.norm(eef_pos - handle_pos, dim=-1)
        
        # Gripper logic:
        # Phase 1: Close if close to handle and action 8 >= 0
        # Phase 2: Stay closed
        # Phase 3: Open if action 8 < 0
        
        grasping_phase = (self.phase_buf == self.PHASE_GRASP)
        holding_phase  = (self.phase_buf == self.PHASE_PLACE)
        release_phase  = (self.phase_buf == self.PHASE_RELEASE)
        
        should_close = torch.zeros(N, dtype=torch.bool, device=self.device)
        # Phase 1: close if close to handle
        should_close = torch.where(grasping_phase, (dist <= 0.035) & (self.actions[:, 8] >= 0.0), should_close)
        # Phase 2: keep closed
        should_close = torch.where(holding_phase, torch.ones_like(should_close), should_close)
        # Phase 3: follow action 8
        should_close = torch.where(release_phase, self.actions[:, 8] >= 0.0, should_close)
        
        g_ids = self.dof_indices["right_gripper"]
        fing_tgt = torch.where(
            should_close.unsqueeze(1).expand(-1, 9),
            self.gripper_upper_limits.expand(N, 9),
            self.gripper_lower_limits.expand(N, 9),
        )
        pos_targets[:, g_ids] = fing_tgt

        pos_targets = tensor_clamp(pos_targets, self.dof_lower_limits, self.dof_upper_limits)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh_tensors()

        # Reveal platform when object is lifted
        obj_heights  = self.states["object_pos"][:, 2] - self.initial_object_z
        reveal_mask  = (obj_heights > 0.02) & (self.platform_pos_tensor[:, 2] < 0)
        target_z     = 0.55
        if reveal_mask.any():
            self.platform_pos_tensor[reveal_mask, 2] = target_z
            self.root_state[self.plat_actor_ids[reveal_mask], 2] = target_z
            p_ids = self.plat_actor_ids[reveal_mask].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(p_ids), len(p_ids))

        # ── Phase transition ───────────────────────────────────────────────
        pitch, roll = _quat_to_pitch_roll(self.states["base_quat"])
        is_bal = (torch.abs(pitch) < self.balanced_pitch_thr) & \
                 (torch.abs(roll)  < self.balanced_pitch_thr * 1.5)
        self.balance_timer = torch.where(is_bal, self.balance_timer + 1,
                                          torch.zeros_like(self.balance_timer))

        base_pos   = self.states["base_pos"]
        base_quat  = self.states["base_quat"]
        handle_pos = self.states["handle_target_pos"]
        object_pos = self.states["object_pos"]
        platform_pos = self.platform_pos_tensor
        plat_surface_z = platform_pos[:, 2] + 0.005
        obj_up = quat_apply(self.states["object_rot"], torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1))
        obj_bottom_z = object_pos[:, 2] - self.object_half_heights * obj_up[:, 2]

        fwd_w = quat_apply(base_quat,
                           torch.tensor([1., 0., 0.], device=self.device)
                           .unsqueeze(0).repeat(self.num_envs, 1))
        rt_w  = quat_apply(base_quat,
                           torch.tensor([0.,-1., 0.], device=self.device)
                           .unsqueeze(0).repeat(self.num_envs, 1))
        pregrasp_xy      = handle_pos[:, :2] - 0.18 * fwd_w[:, :2] - 0.15 * rt_w[:, :2]
        dist_to_pregrasp = torch.norm(pregrasp_xy - base_pos[:, :2], dim=-1)
        
        # Phase 0 -> 1: Approach -> Grasp
        in_range = dist_to_pregrasp < self.grasp_approach_dist
        ready_grasp = (self.phase_buf == self.PHASE_APPROACH) & in_range & (self.balance_timer >= self.balance_min_steps)
        self.phase_buf = torch.where(ready_grasp, torch.ones_like(self.phase_buf) * self.PHASE_GRASP, self.phase_buf)

        # Phase 1 -> 2: Grasp -> Place (XY alignment)
        # Condition: Object is lifted significantly
        obj_lifted = (object_pos[:, 2] - self.initial_object_z > 0.02)
        ready_place = (self.phase_buf == self.PHASE_GRASP) & obj_lifted
        self.phase_buf = torch.where(ready_place, torch.ones_like(self.phase_buf) * self.PHASE_PLACE, self.phase_buf)

        # Phase 2 -> 3: Place -> Release
        # Condition: Object is centered above platform within 2cm horizontal and 2cm vertical gap
        dist_xy_to_plat = torch.norm(object_pos[:, :2] - platform_pos[:, :2], dim=-1)
        dist_z_to_plat  = obj_bottom_z - plat_surface_z
        ready_release = (self.phase_buf == self.PHASE_PLACE) & (dist_xy_to_plat < 0.02) & (dist_z_to_plat < 0.02)
        self.phase_buf = torch.where(ready_release, torch.ones_like(self.phase_buf) * self.PHASE_RELEASE, self.phase_buf)

        # ── Episode resets ─────────────────────────────────────────────────
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            # ── 訓練時關閉，評估時改回 True ──────────────────────────────────
            _EVAL_MODE = False

            if _EVAL_MODE:
                self.total_attempts          += len(env_ids)
                self.total_successes         += self.episode_success[env_ids].sum().item()
                self.total_partial_successes += self.partial_success_buf[env_ids].sum().item()
                self.total_phase2_achieved   += self.phase2_achieved_buf[env_ids].sum().item()
                self.total_tight_contacts    += self.tight_contact_buf[env_ids].sum().item()

                # Skip stats print during warmup (first num_envs episodes)
                if not self._stat_warmup_done:
                    if self.total_attempts >= self.num_envs:
                        self.total_attempts = 0; self.total_successes = 0
                        self.total_partial_successes = 0; self.total_phase2_achieved = 0
                        self.total_tight_contacts = 0; self._stat_warmup_done = True
                    # Fall through to reset_idx so envs are not stuck
                else:
                    sr  = self.total_successes         / max(self.total_attempts, 1)
                    psr = self.total_partial_successes / max(self.total_attempts, 1)
                    p2r = self.total_phase2_achieved   / max(self.total_attempts, 1)
                    tcr = self.total_tight_contacts    / max(self.total_attempts, 1)
                    self.extras["metrics/success_rate"]         = sr
                    self.extras["metrics/partial_success_rate"] = psr
                    self.extras["metrics/phase2_achieved_rate"] = p2r
                    self.extras["metrics/tight_contact_rate"]   = tcr
                    print(f"final success_rate: {sr:.4f}  "
                          f"final partial_success_rate: {psr:.4f}  "
                          f"final phase2_rate: {p2r:.4f}  "
                          f"final tight_contact_rate: {tcr:.4f}  "
                          f"({self.total_attempts} eps)")

            self.reset_idx(env_ids)
            self.episode_success[env_ids]     = False
            self.partial_success_buf[env_ids] = False
            self.phase2_achieved_buf[env_ids] = False
            self.tight_contact_buf[env_ids]   = False

        self.compute_observations()
        self.compute_reward()


# ══════════════════════════════════════════════════════════════════════════════
# Master reward function
# ══════════════════════════════════════════════════════════════════════════════

def compute_balance_grasp_reward(
    reset_buf, progress_buf, actions, prev_actions,
    base_pos, base_quat, base_ang_vel,
    eef_pos, eef_rot, handle_pos, handle_rot,
    object_pos, object_rot, initial_object_z,
    platform_pos, phase_buf,
    num_envs: int, max_episode_length: float,
    h_mid: float, h_grasp_mid: float, fall_pitch_thr: float,
    object_half_height,   # per-env tensor [num_envs]
    balance_scale: float, alive_bonus: float, height_scale: float,
    approach_scale: float, dist_scale: float, rot_scale: float,
    grasp_scale: float, lift_scale: float, success_bonus: float,
    fall_penalty: float, action_penalty_scale: float,
):
    dev = base_pos.device

    # ── Pitch / roll ──────────────────────────────────────────────────────
    qx, qy, qz, qw = base_quat[:,0], base_quat[:,1], base_quat[:,2], base_quat[:,3]
    roll  = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp  = torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0)
    pitch = torch.asin(sinp)
    pitch_abs = torch.abs(pitch)
    roll_abs  = torch.abs(roll)

    # ── 1. Balance ────────────────────────────────────────────────────────
    bal_rew   = torch.exp(-15.0 * pitch**2) + torch.exp(-15.0 * roll**2)
    ang_speed = torch.norm(base_ang_vel, dim=-1)
    bal_rew   = (bal_rew - 0.3 * torch.clamp(ang_speed, 0.0, 5.0)) * balance_scale

    # ── 2. Alive bonus ────────────────────────────────────────────────────
    is_balanced = (pitch_abs < fall_pitch_thr) & (roll_abs < fall_pitch_thr)
    alive_rew   = torch.where(is_balanced,
                              torch.full_like(bal_rew, alive_bonus),
                              torch.zeros_like(bal_rew))

    # ── Platform & Object States (Early calc for use in multiple sections) ──
    ohh        = object_half_height.unsqueeze(-1)   # [N,1] for broadcast
    w_obj_up   = quat_apply(object_rot, torch.tensor([0.,0.,1.], device=dev).repeat(num_envs, 1))
    obj_bottom = object_pos - ohh * w_obj_up
    plat_surface_z = platform_pos[:, 2] + 0.005
    dist_xy_to_plat = torch.norm(obj_bottom[:, :2] - platform_pos[:, :2], p=2, dim=-1)
    dist_z_to_plat_surface = obj_bottom[:, 2] - plat_surface_z
    is_over_plat = (dist_xy_to_plat < 0.04)
    is_on_plat   = is_over_plat & (torch.abs(dist_z_to_plat_surface) < 0.015)

    # ── 3. Height regulation ──────────────────────────────────────────────
    # Height logic based on phases
    # Phase 0: h_mid (0.45)
    # Phase 1 & 2: h_grasp_low (0.42) - lowered to help reach mug
    # Phase 3: h_mid (0.45) - lowered to ensure arm can press object to platform
    
    is_approaching = (phase_buf == 0)
    is_manipulating = (phase_buf == 1) | (phase_buf == 2)
    is_releasing = (phase_buf == 3)
    
    h_grasp_low = 0.42
    height_target = torch.where(is_approaching, torch.full_like(base_pos[:, 2], h_mid), 
                    torch.where(is_manipulating, torch.full_like(base_pos[:, 2], h_grasp_low),
                    torch.full_like(base_pos[:, 2], h_mid)))
    
    height_err = torch.abs(base_pos[:, 2] - height_target)
    height_rew = torch.exp(-20.0 * height_err**2) * height_scale

    # ── 4. Approach (Phase 0) ─────────────────────────────────────────────
    fwd_w = quat_apply(base_quat,
                       torch.tensor([1.,0.,0.], device=dev).unsqueeze(0).expand(num_envs, -1))
    rt_w  = quat_apply(base_quat,
                       torch.tensor([0.,-1.,0.], device=dev).unsqueeze(0).expand(num_envs, -1))
    pregrasp_xy      = handle_pos[:, :2] - 0.18 * fwd_w[:, :2] - 0.15 * rt_w[:, :2]
    dist_to_pregrasp = torch.norm(pregrasp_xy - base_pos[:, :2], dim=-1)
    approach_rew     = (1.0 / (1.0 + 2.0 * dist_to_pregrasp)) * approach_scale
    approach_rew     = torch.where(is_approaching, approach_rew, torch.zeros_like(approach_rew))

    arm_actions    = actions[:, 4:8]
    arm_motion_pen = torch.sum(arm_actions**2, dim=-1) * 0.1
    arm_motion_pen = torch.where(is_approaching, arm_motion_pen, torch.zeros_like(arm_motion_pen))

    # ── 5. EEF reaching (Phase 1) ─────────────────────────────────────────
    d_eef    = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    dist_rew = 1.0 / (1.0 + 40.0 * d_eef**2) * dist_scale
    dist_rew = torch.where(phase_buf >= 1, dist_rew, dist_rew * 0.1)

    # ── 6. Orientation alignment (Phase 1 & 2) ───────────────────────────────
    ax1 = quat_apply(eef_rot, torch.tensor([0.,0.,-1.], device=dev).repeat(num_envs,1))
    ax2 = quat_apply(handle_rot, torch.tensor([1.,0.,0.], device=dev).repeat(num_envs,1))
    ax3 = quat_apply(eef_rot, torch.tensor([1.,0.,0.], device=dev).repeat(num_envs,1))
    ax4 = quat_apply(handle_rot, torch.tensor([0.,0.,1.], device=dev).repeat(num_envs,1))
    dot1 = (ax1 * ax2).sum(-1)
    dot2 = (ax3 * ax4).sum(-1)
    rot_rew = 0.5 * (torch.clamp(dot1, max=0.0) + torch.clamp(dot2, min=0.0)) * rot_scale
    rot_rew = torch.where(phase_buf >= 1, rot_rew, torch.zeros_like(rot_rew))

    # ── 7. Grasp ──────────────────────────────────────────────────────────
    is_aligned   = (dot1 < -0.60) & (dot2 > 0.60)
    is_close_eef = (d_eef < 0.035) & is_aligned
    gripper_close = actions[:, 8] >= 0.0
    grasp_rew = torch.where(
        is_close_eef & gripper_close & (phase_buf == 1),
        torch.full_like(dist_rew, grasp_scale * 2.0),
        torch.zeros_like(dist_rew))

    # ── 8. Lift ───────────────────────────────────────────────────────────
    obj_height  = object_pos[:, 2] - initial_object_z
    is_grasping = is_close_eef & gripper_close & (phase_buf >= 1)
    
    # Tightened on-plat threshold to prevent hovering release
    is_on_plat   = is_over_plat & (torch.abs(dist_z_to_plat_surface) < 0.012)

    # Allow lift reward to persist if object is on platform (even if not grasping)
    lift_rew    = torch.where(is_grasping | is_on_plat,
                              lift_scale * torch.clamp(obj_height, 0.0, 0.06),
                              torch.zeros_like(dist_rew))

    w_obj_up2      = quat_apply(object_rot, torch.tensor([0.,0.,1.], device=dev).repeat(num_envs,1))
    obj_upright    = (w_obj_up2 * torch.tensor([0.,0.,1.], device=dev).repeat(num_envs,1)).sum(-1)
    is_upright     = obj_upright > 0.85
    orient_rew     = torch.pow(torch.clamp(obj_upright, 0.0), 8) * 20.0
    orient_rew     = torch.where((is_grasping | is_on_plat) & (obj_height > 0.01), orient_rew,
                                  torch.zeros_like(orient_rew))

    # ── 9. Transport & Placement (Phase 2 & 3) ────────────────────────────
    # Phase 2: Move above platform (3cm gap)
    is_over_plat = (dist_xy_to_plat < 0.05)
    target_z_gap = 0.03
    z_gap_err = torch.abs(dist_z_to_plat_surface - target_z_gap)

    # Stronger long-range pull to platform
    plat_attract_rew = (1.0 / (1.0 + 3.0 * dist_xy_to_plat)) * 40.0

    trans_rew = torch.where(((phase_buf == 2) | (phase_buf == 3)) & (is_grasping | is_on_plat),
                             plat_attract_rew + 10.0 * torch.exp(-20.0 * z_gap_err**2),
                             torch.zeros_like(dist_rew))
    trans_rew = torch.where((phase_buf >= 2) & (~is_upright), trans_rew * 0.1, trans_rew)

    # Phase 3: Landing & Release
    # Sharpen placement reward to peak at exact contact
    place_rew = 100.0 * torch.exp(-40.0 * torch.abs(dist_z_to_plat_surface))
    # Place reward should persist after release if it's on the platform
    place_rew = torch.where((phase_buf == 3) & (is_grasping | is_on_plat) & is_upright,
                             place_rew, torch.zeros_like(dist_rew))

    # ── 10. Grasp-balance synergy ──────────────────────────────────────────
    is_lifted = (obj_height > 0.03)
    grasp_bal_bonus = torch.where(is_grasping & is_balanced & is_lifted,
                                   torch.full_like(dist_rew, 5.0),
                                   torch.zeros_like(dist_rew))

    # ── 11. Release & Success ─────────────────────────────────────────────
    gripper_open    = actions[:, 8] < 0.0
    eef_dist_to_handle = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    
    # Successful release: on plat, upright, and gripper opens
    is_releasing_act = (phase_buf == 3) & is_on_plat & is_upright & gripper_open
    release_rew     = torch.where(is_releasing_act,
                                   torch.full_like(dist_rew, 100.0),
                                   torch.zeros_like(dist_rew))
    
    # Retreat: move EEF away from handle after release, specifically BACKWARD
    # Project retreat vector onto robot's backward axis (-X)
    back_dir = quat_apply(base_quat, torch.tensor([-1., 0., 0.], device=dev).repeat(num_envs, 1))
    retreat_vec = eef_pos - handle_pos
    retreat_back_proj = (retreat_vec * back_dir).sum(-1)
    retreat_up_proj   = retreat_vec[:, 2]
    
    # Strengthen retreat reward: make it more aggressive and reward further movement
    is_retreating = (phase_buf == 3) & gripper_open & is_on_plat
    retreat_rew = torch.where(is_retreating,
                               torch.clamp(retreat_back_proj, 0.0, 0.15) * 4000.0,
                               torch.zeros_like(dist_rew))
    
    # Retreat UP penalty: strongly penalize lifting arm while retreating (avoids hitting dumbbell top)
    retreat_up_pen = torch.where(is_retreating & (retreat_up_proj > 0.01),
                                  retreat_up_proj * 5000.0,
                                  torch.zeros_like(dist_rew))
    
    # Penalty for hovering over platform in Phase 3 without releasing
    # OR staying near the handle after releasing
    hover_pen       = torch.where((phase_buf == 3) & is_over_plat & (~gripper_open),
                                   torch.full_like(dist_rew, 5.0),
                                   torch.zeros_like(dist_rew))
    
    # Stay-near-handle penalty: if gripper is open in phase 3 but haven't retreated enough
    stay_near_pen = torch.where(is_retreating & (retreat_back_proj < 0.10),
                                 torch.full_like(dist_rew, 10.0),
                                 torch.zeros_like(dist_rew))

    # Platform collision penalty for EEF (stay above platform surface)
    plat_collision_pen = torch.where((phase_buf == 3) & is_over_plat & (eef_pos[:, 2] < plat_surface_z + 0.02),
                                      torch.full_like(dist_rew, 5.0),
                                      torch.zeros_like(dist_rew))

    is_success  = (phase_buf == 3) & is_on_plat & is_upright & is_balanced & gripper_open & (retreat_back_proj > 0.12)
    success_rew = torch.where(is_success,
                               torch.full_like(dist_rew, success_bonus),
                               torch.zeros_like(dist_rew))

    # ── 12. Braking (Phase 1, 2, 3) ─────────────────────────────────────────
    # Slightly reduced braking penalty to avoid jitter from conflicting balance/stop commands
    braking_pen = torch.where(phase_buf >= 1,
                               torch.sum(actions[:, 2:4]**2, dim=-1) * 1.5,
                               torch.zeros_like(dist_rew))

    # ── Penalties ─────────────────────────────────────────────────────────
    action_delta = actions - prev_actions
    smooth_pen   = torch.sum(action_delta**2, dim=-1) * action_penalty_scale
    energy_pen   = torch.sum(actions**2,      dim=-1) * action_penalty_scale * 0.3
    time_pen     = torch.full_like(bal_rew, 1.5)

    is_fallen = ((pitch_abs > fall_pitch_thr) | (roll_abs > fall_pitch_thr)) & (progress_buf >= 20)
    fall_pen  = torch.where(is_fallen,
                             torch.full_like(bal_rew, fall_penalty),
                             torch.zeros_like(bal_rew))

    total_pen = smooth_pen + energy_pen + time_pen + arm_motion_pen + fall_pen + braking_pen + hover_pen + plat_collision_pen + stay_near_pen + retreat_up_pen

    rewards = (
        bal_rew + alive_rew + height_rew + approach_rew +
        dist_rew + rot_rew + grasp_rew + lift_rew + orient_rew +
        trans_rew + place_rew + release_rew + retreat_rew +
        grasp_bal_bonus + success_rew - total_pen
    )

    # ── Reset conditions ───────────────────────────────────────────────────
    reset_buf = torch.where(is_fallen,  torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(is_success, torch.ones_like(reset_buf), reset_buf)

    obj_dropped = (object_pos[:, 2] < 0.20) & (~is_grasping) & (~is_on_plat)
    reset_buf   = torch.where(obj_dropped, torch.ones_like(reset_buf), reset_buf)

    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                             torch.ones_like(reset_buf), reset_buf)

    return (
        rewards, reset_buf,
        bal_rew, alive_rew, height_rew, approach_rew,
        dist_rew, rot_rew, grasp_rew, lift_rew, trans_rew, place_rew,
        release_rew, retreat_rew, orient_rew,
        grasp_bal_bonus, success_rew, total_pen,
        smooth_pen, energy_pen, arm_motion_pen, fall_pen, braking_pen, plat_collision_pen,
        is_success, is_fallen, is_balanced, is_grasping, is_lifted,
    )
