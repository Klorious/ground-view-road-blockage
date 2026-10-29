# V40 ground-view road reconstruction and blockage evaluation.
# The obsolete fixed-Y baseline has been removed; V40 thresholds and fallbacks remain.
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
BASE = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE / 'data'
DEFAULT_COCO = DEFAULT_DATASET / '_annotations.coco.json'
DEFAULT_OUTPUT = BASE / 'news_2d_road_aligned_results'
STATUS_NAMES = {0: 'small_or_clear', 1: 'half', 2: 'full', -1: 'insufficient'}
AUXILIARY_DIRECTION_ALIASES = {'road_line': {'road line', 'roadline'}, 'cone': {'cone', 'orange traffic cone'}, 'curb': {'curb', 'roadside curb'}, 'guardrail': {'guardrail', 'metal guardrail'}, 'barricade': {'barricade', 'yellow and black striped road barrier'}, 'roadside_wall': {'roadside wall', 'roadside retaining', 'roadside retaining wall'}}
OBSTACLE_CATEGORY_ALIASES = {'rock': {'rock'}, 'fallen_vegetation': {'fallen vegetation', 'treebranch', 'tree branch'}, 'debris_pile': {'debris pile', 'mud'}}

# =============================================================================
# 教授審閱導覽
# =============================================================================
# 這是 V40 刪除未使用程式後的主版本。後續優化應直接修改本檔。
# 建議依序閱讀：Parameters → reconstruct_road_corridor →
# continuous_direction_corridor → apply_v20_v12_base_v19_direction →
# road_aligned_method → evaluate。
# 本檔的區塊註解只說明功能與資料流，不改公式、門檻或回退順序。
# 函式註解分為三層：
#   【主要功能】核心資料流，附輸入、步驟、輸出或回退條件；
#   【階段函式／重建函式】完成一個演算法子階段，附其流程角色；
#   【各類工具】低層計算，以一行說明用途，閱讀主流程時可先略過。

# =============================================================================
# 01. 全域參數與核心資料結構
# =============================================================================
# Parameters 集中保存道路追蹤、遮擋重建、輔助曲線、橫斷面、分類與
# 信心門檻；CrossSection 則保存單一橫斷面的端點、阻塞率與來源狀態。
@dataclass
class Parameters:
    partial_threshold: float = 0.3
    full_threshold: float = 0.75
    close_radius: int = 2
    sample_spacing: float = 10.0
    smooth_window: int = 7
    row_band_height: int = 6
    row_min_valid_rows: int = 2
    row_short_gap_bands: int = 2
    row_max_occlusion_gap_bands: int = 60
    row_max_visible_continuation_gap_bands: int = 120
    row_occlusion_min_fraction: float = 0.2
    band_segment_merge_gap_px: int = 16
    band_width_growth_limit: float = 1.1
    band_tracking_margin_fraction: float = 0.2
    center_curve_smoothness: float = 18.0
    width_curve_smoothness: float = 24.0
    curve_robust_iterations: int = 3
    one_sided_max_bands: int = 72
    one_sided_fit_observations: int = 12
    one_sided_start_gap_bands: int = 2
    one_sided_empty_tolerance_bands: int = 2
    one_sided_min_obstacle_width_fraction: float = 0.08
    terminal_gate_min_obstacle_width_fraction: float = 0.18
    terminal_component_link_fraction: float = 0.25
    one_sided_core_fraction: float = 0.55
    one_sided_soft_weight: float = 0.2
    terminal_fragment_max_width_ratio: float = 0.65
    one_sided_max_width_growth: float = 1.02
    one_side_inferred_width_ratio: float = 0.72
    one_side_inferred_obstacle_fraction: float = 0.08
    one_side_inferred_window_bands: int = 10
    enable_one_side_lateral_inference: bool = False
    enable_interior_occlusion_inference: bool = True
    interior_occlusion_width_ratio: float = 0.78
    interior_occlusion_obstacle_fraction: float = 0.08
    interior_occlusion_anchor_window_bands: int = 30
    interior_occlusion_soft_weight: float = 0.7
    tangent_window_px: float = 40.0
    cross_section_gap: int = 3
    obstacle_link_gap: int = 20
    min_road_width: int = 6
    min_road_area_ratio: float = 0.002
    min_main_component_ratio: float = 0.6
    min_path_diagonal_ratio: float = 0.08
    horizontal_fallback_diagonal_ratio: float = 0.25
    min_valid_slice_ratio: float = 0.6
    branch_warning_threshold: int = 5
    required_consecutive_slices: int = 3
    enable_continuous_direction_tracking: bool = True
    direction_angle_step_degrees: float = 15.0
    direction_coarse_long_edge: int = 320
    direction_seed_tracks_per_angle: int = 2
    direction_top_k: int = 8
    direction_geometry_mode: str = 'global_angle_cross_sections'
    local_direction_refine_iterations: int = 2
    local_direction_search_width_factor: float = 1.35
    local_direction_max_center_shift_fraction: float = 0.45
    include_unknown_internal_gaps: bool = True
    enable_auxiliary_direction_evidence: bool = True
    auxiliary_min_component_pixels: int = 12
    auxiliary_min_component_span_px: float = 18.0
    auxiliary_min_elongation: float = 2.0
    auxiliary_min_associated_fraction: float = 0.55
    auxiliary_association_width_factor: float = 1.25
    auxiliary_max_road_angle_delta_deg: float = 35.0
    auxiliary_segment_stations: int = 8
    auxiliary_min_segment_evidence_weight: float = 0.12
    auxiliary_global_prior_weight: float = 0.85
    auxiliary_max_segment_delta_deg: float = 22.5
    auxiliary_max_adjacent_delta_deg: float = 12.0
    auxiliary_min_section_inside_ratio: float = 0.72
    auxiliary_initial_cluster_tolerance_deg: float = 18.0
    auxiliary_initial_candidate_tolerance_deg: float = 45.0
    auxiliary_initial_max_hypotheses: int = 8
    auxiliary_initial_coarse_bonus: float = 2.0
    auxiliary_initial_final_bonus: float = 3.2
    auxiliary_joint_local_road_weight: float = 0.65
    auxiliary_joint_local_max_disagreement_deg: float = 25.0
    auxiliary_joint_local_max_global_delta_deg: float = 35.0
    enable_auxiliary_road_reconstruction: bool = True
    v13_fragment_pair_max_gap_bands: int = 240
    v13_fragment_pair_max_cost: float = 1.3
    v13_fragment_pair_min_margin: float = 0.08
    v13_roadline_internal_width_ratio: float = 0.82
    v13_roadline_anchor_width_ratio: float = 0.7
    v13_internal_anchor_window_bands: int = 72
    v13_roadline_center_max_offset_fraction: float = 0.4
    v13_roadline_single_center_fraction: float = 0.14
    v13_roadline_pair_max_separation_fraction: float = 0.18
    v13_roadline_pair_min_overlap_bands: int = 8
    v13_roadline_center_min_span_bands: int = 8
    v13_roadline_center_fit_weight: float = 3.0
    v13_roadline_center_fit_smoothness: float = 2.0
    v13_roadline_center_blend: float = 0.85
    v14_roadline_direct_center_blend: float = 0.96
    v14_roadline_edge_shape_blend: float = 0.82
    v14_roadline_edge_min_offset_fraction: float = 0.18
    v14_roadline_edge_max_offset_fraction: float = 0.82
    v14_roadline_edge_max_fraction_mad: float = 0.18
    v15_aux_track_angle_tolerance_deg: float = 12.0
    v15_aux_track_min_consistency: float = 0.92
    v15_aux_track_max_longitudinal_gap_px: float = 150.0
    v15_aux_track_max_lateral_gap_px: float = 30.0
    v15_aux_track_min_merged_span_factor: float = 1.35
    v15_aux_track_fragmented_class_count: int = 3
    v15_aux_track_isolated_span_factor: float = 2.5
    v15_aux_track_isolated_diagonal_ratio: float = 0.1
    enable_v16_partial_auxiliary_spline: bool = True
    v16_min_curve_station_span: int = 8
    v16_min_local_evidence_weight: float = 0.1
    v16_min_local_consistency: float = 0.8
    v16_curve_smoothness: float = 8.0
    v16_curve_robust_iterations: int = 3
    v16_tangent_window_stations: int = 5
    v16_transition_stations: int = 10
    v16_core_direction_blend: float = 0.96
    v16_max_global_delta_deg: float = 55.0
    v16_max_adjacent_delta_deg: float = 4.0
    v16_min_section_inside_ratio: float = 0.58
    v16_max_endpoint_shift_fraction: float = 0.52
    enable_v17_auxiliary_guided_geometry: bool = True
    v17_guardrail_longitudinal_bin_px: float = 6.0
    v17_guardrail_normal_mode_bin_px: float = 4.0
    v17_guardrail_min_ridge_points: int = 8
    v17_roadline_curve_bin_px: float = 5.0
    v17_roadline_normal_mode_bin_px: float = 3.0
    v17_roadline_min_road_support: float = 0.35
    v17_roadline_min_centrality: float = 0.12
    v17_roadline_min_curve_points: int = 8
    v17_roadline_center_blend: float = 0.98
    v17_roadline_center_fit_weight: float = 12.0
    v17_center_transition_bands: int = 12
    v17_center_max_shift_width_fraction: float = 2.0
    enable_v18_auxiliary_curve_roles: bool = True
    v18_direct_center_max_offset_fraction: float = 0.12
    v18_ranked_center_max_offset_fraction: float = 0.36
    v18_ranked_center_margin_fraction: float = 0.08
    v18_direct_center_min_coverage: float = 0.12
    v18_shape_min_coverage: float = 0.1
    v18_shape_min_associated_fraction: float = 0.5
    v18_shape_max_turn_p95_deg: float = 22.0
    v18_shape_center_blend: float = 0.88
    v18_shape_max_shift_width_fraction: float = 0.85
    v18_shape_fit_weight: float = 4.0
    v18_guardrail_facing_smoothness: float = 6.0
    v18_curve_support_transition_bands: int = 3
    v18_local_auxiliary_max_global_delta_deg: float = 65.0
    v18_aux_global_override_max_geometry_deficit: float = 0.15
    v18_allow_single_boundary_position_constraint: bool = False
    enable_v19_auxiliary_road_envelope: bool = True
    v19_envelope_min_station_observations: int = 6
    v19_envelope_min_span_bands: int = 8
    v19_envelope_min_associated_fraction: float = 0.55
    v19_envelope_min_angle_consistency: float = 0.9
    v19_envelope_max_station_gap_bands: int = 4
    v19_envelope_max_offset_width_fraction: float = 0.82
    v19_envelope_bridge_width_px: int = 5
    v19_envelope_expand_geometry: bool = True
    v19_envelope_boundary_blend: float = 0.8
    v19_envelope_max_width_growth: float = 1.35
    enable_v23_auxiliary_segmentation_envelope: bool = True
    v23_edge_quantile: float = 0.1
    v23_pair_max_angle_delta_deg: float = 18.0
    v23_pair_min_overlap_stations: int = 6
    v23_pair_min_overlap_span_bands: int = 8
    v23_pair_min_v12_support_fraction: float = 0.25
    v23_center_roadline_max_offset_fraction: float = 0.3
    v23_side_aux_min_offset_fraction: float = 0.22
    enable_v24_obstacle_driven_reconstruction: bool = False
    v24_obstacle_min_cross_section_fraction: float = 0.03
    v24_lateral_visible_width_ratio: float = 0.85
    enable_v25_navigation_guided_reconstruction: bool = False
    v25_fragment_pair_reacquire_widths: float = 1.3
    v25_authoritative_short_tangent_bands: int = 8
    v25_authoritative_return_bands: int = 16
    v25_terminal_turn_limit_deg: float = 0.0
    v25_shape_max_backtrack_ratio: float = 0.06
    enable_v28_frozen_center_state_reconstruction: bool = True
    v28_terminal_max_bands: int = 4096
    v28_terminal_obstacle_fraction: float = 0.03
    v28_terminal_start_gap_widths: float = 1.25
    v28_lateral_visible_width_ratio: float = 0.85
    v28_lateral_obstacle_fraction: float = 0.03
    v28_lateral_width_window: int = 12
    enable_v29_direction_locked_terminal: bool = True
    enable_v29_continuous_cross_sections: bool = True
    v29_cross_section_crossing_history: int = 1
    v13_boundary_min_offset_fraction: float = 0.28
    v13_boundary_max_offset_fraction: float = 0.85
    v13_boundary_blend: float = 0.2
    v13_single_boundary_min_span_bands: int = 12
    v13_single_boundary_min_observations: int = 5
    v13_single_boundary_blend: float = 0.6
    v13_fragment_tangent_observations: int = 8
    v13_fragment_hermite_overshoot_width_fraction: float = 0.35
    v13_auxiliary_terminal_max_bands: int = 96
    v13_auxiliary_terminal_min_span_bands: int = 3
    v13_auxiliary_terminal_width_taper: float = 0.88
    v13_auxiliary_component_angle_delta_deg: float = 40.0
    v13_auxiliary_component_min_associated_fraction: float = 0.35
    enable_unified_reconstruction: bool = True
    enable_v20_v12_base_v19_direction: bool = True
    enable_v21_decoupled_geometry: bool = True
    v21_global_aux_min_span_diagonal_ratio: float = 0.18
    v21_global_roadline_min_span_diagonal_ratio: float = 0.14
    v21_global_aux_min_angle_consistency: float = 0.84
    v21_global_aux_min_confidence: float = 0.45
    v21_global_aux_max_road_score_deficit: float = 0.65
    v21_global_aux_final_bonus: float = 2.2
    v21_center_min_coverage: float = 0.55
    v21_center_median_balance_max: float = 0.18
    v21_center_p90_balance_max: float = 0.3
    v21_center_offset_mad_max: float = 0.12
    v21_centerline_bilateral_min_inside_ratio: float = 0.9
    v21_center_max_shift_width_fraction: float = 0.3
    v21_center_max_length_ratio: float = 1.2
    v21_center_max_turn_p95_deg: float = 45.0
    v21_center_min_inside_ratio: float = 0.95
    v21_center_max_backtrack_ratio: float = 0.03
    v21_center_transition_bands: int = 16
    enable_v22_authoritative_roadline_center: bool = True
    v22_ranked_center_median_balance_max: float = 0.35
    v22_ranked_center_p90_balance_max: float = 0.5
    v22_authoritative_tangent_fit_points: int = 10
    v22_authoritative_boundary_margin_fraction: float = 0.02
    v26_local_center_median_balance_max: float = 0.2
    v26_local_center_p90_balance_max: float = 0.3
    v21_cross_section_curvature_safety: float = 0.7
    enable_v30_auxiliary_navigation_fusion: bool = True
    v30_auxiliary_consensus_angle_deg: float = 15.0
    v30_auxiliary_consensus_station_gap: int = 12
    v30_auxiliary_consensus_min_tracks: int = 2
    v30_auxiliary_consensus_min_union_coverage: float = 0.12
    v30_auxiliary_consensus_score_boost: float = 0.35
    v30_strong_joint_direction_score: float = 0.6
    enable_v31_multi_aux_pretrack_direction: bool = True
    v31_consensus_angle_deg: float = 15.0
    v31_conflict_angle_deg: float = 30.0
    v31_min_union_coverage: float = 0.14
    v31_min_source_diversity: int = 2
    v31_pretrack_prior_weight: float = 2.8
    v31_pretrack_max_geometry_deficit: float = 1.35
    v31_local_support_radius_bands: int = 4
    v31_local_transition_bands: int = 10
    v31_local_min_consensus_weight: float = 0.55
    v31_local_max_adjacent_delta_deg: float = 3.0
    v31_road_direction_window_bands: int = 10
    v31_disable_shape_center_shift: bool = True
    enable_v32_authoritative_navigation_direction: bool = True
    v32_single_roadline_min_stations: int = 8
    v32_single_roadline_min_coverage: float = 0.15
    v32_single_side_min_stations: int = 10
    v32_single_side_min_coverage: float = 0.18
    v32_min_angle_consistency: float = 0.8
    v32_consensus_angle_deg: float = 15.0
    v32_conflict_angle_deg: float = 30.0
    v32_conflict_margin_ratio: float = 1.2
    v32_small_gap_bands: int = 6
    v32_transition_bands: int = 8
    v32_road_fallback_window_bands: int = 14
    v32_max_adjacent_delta_deg: float = 3.0
    v32_position_anchor_stride_bands: int = 4
    v32_position_anchor_smoothness: float = 120.0
    v32_corridor_projection_margin: float = 0.03
    v32_direction_association_width_factor: float = 2.5
    v32_direction_min_associated_fraction: float = 0.12
    enable_v33_physical_roadline_tracks: bool = True
    v33_track_merge_max_station_gap: int = 36
    v33_track_merge_max_overlap_bands: int = 3
    v33_track_merge_max_angle_deg: float = 22.0
    v33_track_merge_max_normal_width_fraction: float = 0.22
    v33_vanishing_point_residual_width_fraction: float = 0.2
    v33_center_min_coverage: float = 0.06
    v33_center_min_bilateral_support: float = 0.38
    v33_center_min_road_support: float = 0.48
    v33_center_min_score: float = 0.54
    v33_center_offset_soft_scale: float = 0.4
    v33_double_line_max_separation_width_fraction: float = 0.16
    v33_center_position_weight: float = 240.0
    v33_center_transition_bands: int = 10
    enable_v35_native_roadline_center: bool = True
    v35_single_track_min_centrality: float = 0.42
    v35_single_track_center_offset_fraction: float = 0.18
    v35_single_track_long_coverage: float = 0.3
    v35_multi_track_min_centrality: float = 0.25
    v35_max_boundary_hugging_fraction: float = 0.45
    v35_max_lateral_rank_mad: float = 0.24
    v35_center_uniqueness_margin: float = 0.035
    enable_v36_geometric_roadline_authority: bool = True
    v36_transition_min_stations: int = 6
    v36_transition_max_stations: int = 28
    v36_transition_width_factor: float = 1.25
    enable_v37_paired_cross_sections: bool = True
    v37_pair_search_radius: int = 4
    v37_pair_max_center_distance_fraction: float = 0.34
    v37_pair_max_tangent_alignment: float = 0.82
    v37_pair_min_corridor_fraction: float = 0.62
    v37_pair_max_index_step: int = 4
    v37_pair_center_weight: float = 5.0
    v37_pair_angle_weight: float = 2.4
    v37_pair_width_weight: float = 1.8
    v37_pair_index_weight: float = 0.1
    v37_center_min_pair_success: float = 0.45
    v37_center_min_bilateral_station_fraction: float = 0.34
    v37_center_min_side_extent_fraction: float = 0.22
    v37_edge_max_side_extent_fraction: float = 0.13
    v37_center_min_side_road_fraction: float = 0.08
    v37_center_max_lateral_trend_mad: float = 0.24
    v37_track_merge_max_lateral_width_fraction: float = 0.15
    v37_track_merge_max_tangent_delta_deg: float = 18.0
    enable_v38_roadline_group_roles: bool = True
    v38_group_min_common_stations: int = 6
    v38_group_min_common_fraction: float = 0.16
    v38_group_max_angle_delta_deg: float = 35.0
    v38_group_max_rank_mad: float = 0.2
    v38_group_min_rank_separation: float = 0.035
    v38_double_line_gap_ratio: float = 0.48
    v38_pair_progress_weight: float = 0.8
    v38_pair_max_progress_mismatch: float = 0.6
    v38_branch_jump_width_fraction: float = 0.75
    v38_branch_tangent_jump_deg: float = 72.0
    v38_pair_max_short_gap: int = 6
    enable_v38_pairing_quality_gate: bool = True
    enable_v40_v38_roadline_center_sections: bool = True
    v40_native_min_pair_success: float = 0.7
    v40_native_min_inside_ratio: float = 0.82
    v40_native_min_monotonic_ratio: float = 0.96
    v40_native_max_width_step_ratio: float = 0.42
    v40_center_min_score: float = 0.5
    v40_center_min_margin: float = 0.06
    v40_center_min_coverage: float = 0.1
RECONSTRUCTION_STATE_BY_SOURCE = {'direct': 'direct', 'auxiliary_boundary_inferred': 'direct', 'interpolated': 'internal_reconstruction', 'interior_occlusion_inferred': 'internal_reconstruction', 'roadline_supported_internal': 'internal_reconstruction', 'one_side_inferred': 'internal_reconstruction', 'extrapolated': 'terminal_extrapolation', 'auxiliary_supported_extrapolated': 'terminal_extrapolation'}
RECONSTRUCTION_EVIDENCE_BY_SOURCE = {'direct': ('road',), 'auxiliary_boundary_inferred': ('road', 'paired_roadside_boundary'), 'interpolated': ('road_anchor_pair',), 'interior_occlusion_inferred': ('road_anchor_pair', 'obstacle'), 'roadline_supported_internal': ('road_anchor_pair', 'road_line'), 'one_side_inferred': ('road_anchor_pair', 'one_side_obstacle'), 'extrapolated': ('road_trend', 'obstacle'), 'auxiliary_supported_extrapolated': ('road_trend', 'auxiliary')}

# =============================================================================
# 02. 道路重建來源、狀態與證據摘要
# =============================================================================
# 將 direct、interpolated、extrapolated 等幾何來源統一映射成「直接觀測、
# 內部重建、末端外推」三種狀態，供診斷統計與判定信心使用。
# 【狀態工具】將幾何來源名稱歸納為直接觀測、內部重建或末端外推。
def reconstruction_state_for_source(source: str) -> str:
    return RECONSTRUCTION_STATE_BY_SOURCE.get(source, 'direct')

# 【狀態工具】列出某種道路來源所依賴的 Road、錨點、障礙物或趨勢證據。
def reconstruction_evidence_for_source(source: str) -> list[str]:
    return list(RECONSTRUCTION_EVIDENCE_BY_SOURCE.get(source, ('road',)))

# 【統計工具】彙整各站來源，產生三狀態重建數量與支持證據數量。
def summarize_reconstruction_sources(source_types: list[str]) -> tuple[dict, dict]:
    state_counts = {'direct': 0, 'internal_reconstruction': 0, 'terminal_extrapolation': 0}
    evidence_counts: dict[str, int] = {}
    for source in source_types:
        state = reconstruction_state_for_source(source)
        state_counts[state] = state_counts.get(state, 0) + 1
        for evidence in reconstruction_evidence_for_source(source):
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
    return (state_counts, evidence_counts)

# 【狀態工具】依目前走廊來源重新整理重建流程的診斷欄位。
def refresh_reconstruction_metadata(corridor: dict) -> dict:
    sources = list(corridor.get('band_source_types', []))
    states = [reconstruction_state_for_source(source) for source in sources]
    evidence = [reconstruction_evidence_for_source(source) for source in sources]
    state_counts, evidence_counts = summarize_reconstruction_sources(sources)
    corridor['band_core_states'] = states
    corridor['band_support_evidence'] = evidence
    corridor['core_state_counts'] = state_counts
    corridor['support_evidence_counts'] = evidence_counts
    corridor['reconstruction_pipeline'] = 'unified_three_state'
    return corridor

@dataclass
class CrossSection:
    station_index: int
    center_x: float
    center_y: float
    left_x: int
    left_y: int
    right_x: int
    right_y: int
    road_width_px: int
    max_free_width_px: int
    free_width_ratio: float
    blockage_ratio: float
    free_intervals: list[dict] = field(default_factory=list)
    source_type: str = 'direct'
    source_weight: float = 1.0
    selected_free_interval_index: int | None = None
    free_lane_segment_id: int | None = None
    geometry_source: str = 'unknown'

# =============================================================================
# 03. 二值遮罩、連通元件與基礎數值工具
# =============================================================================
# 提供閉運算、最大連通區、方向估計與一維平滑等共用工具；此區不直接
# 決定分類，只為後續道路幾何建立乾淨且可追蹤的觀測。
# 【遮罩工具】以方形鄰域計算每個像素附近的 True 像素數。
def binary_box_count(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.int32)
    size = 2 * radius + 1
    padded = np.pad(mask.astype(np.uint8), radius, mode='constant')
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant')
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]

# 【遮罩工具】對二值遮罩執行閉運算，填補小裂縫並連接相鄰區域。
def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    size = (2 * radius + 1) ** 2
    dilated = binary_box_count(mask, radius) > 0
    return binary_box_count(dilated, radius) == size
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 【遮罩工具】保留最大連通元件，並回傳它占原遮罩的面積比例。
def largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    mask = mask.astype(bool)
    total = int(mask.sum())
    if total == 0:
        return (np.zeros_like(mask), 0.0)
    try:
        import cv2
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return (np.zeros_like(mask), 0.0)
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        result = labels == label
        return (result, float(result.sum() / total))
    except ImportError:
        pass
    height, width = mask.shape
    visited = np.zeros_like(mask)
    best: list[tuple[int, int]] = []
    for y, x in np.argwhere(mask):
        y, x = (int(y), int(x))
        if visited[y, x]:
            continue
        queue = deque([(y, x)])
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for dy, dx in NEIGHBORS_8:
                ny, nx = (cy + dy, cx + dx)
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and (not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        result[np.asarray(ys), np.asarray(xs)] = True
    return (result, float(len(best) / total))

# 【方向工具】由主要 Road 元件估計長軸方向、跨度與水平／垂直先驗。
def road_component_orientation_evidence(mask: np.ndarray, long_edge: int=240) -> dict:
    height, width = mask.shape
    scale = min(1.0, float(long_edge) / max(height, width))
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((small_width, small_height), Image.Resampling.NEAREST), dtype=np.uint8) > 0
    visited = np.zeros_like(small, dtype=bool)
    components: list[dict] = []
    total = int(small.sum())
    minimum_area = max(4, int(total * 0.01))
    for start_y, start_x in np.argwhere(small):
        start_y, start_x = (int(start_y), int(start_x))
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        count = 0
        sum_x = sum_y = 0.0
        min_x = max_x = start_x
        min_y = max_y = start_y
        while queue:
            y, x = queue.popleft()
            count += 1
            sum_x += x
            sum_y += y
            min_x, max_x = (min(min_x, x), max(max_x, x))
            min_y, max_y = (min(min_y, y), max(max_y, y))
            for dy, dx in NEIGHBORS_8:
                ny, nx = (y + dy, x + dx)
                if 0 <= ny < small_height and 0 <= nx < small_width and small[ny, nx] and (not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if count >= minimum_area:
            components.append({'area': count, 'center_x': sum_x / count, 'center_y': sum_y / count, 'width': max_x - min_x + 1, 'height': max_y - min_y + 1})
    components.sort(key=lambda item: item['area'], reverse=True)
    significant = components[:6]
    horizontal_score = vertical_score = 0.0
    arrangement = 'ambiguous'
    if len(significant) >= 2:
        best_pair = None
        best_distance = -1.0
        for first_index in range(len(significant)):
            for second_index in range(first_index + 1, len(significant)):
                first, second = (significant[first_index], significant[second_index])
                dx = abs(first['center_x'] - second['center_x']) / max(1, small_width)
                dy = abs(first['center_y'] - second['center_y']) / max(1, small_height)
                distance = math.hypot(dx, dy)
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (dx, dy)
        if best_pair is not None:
            horizontal_score, vertical_score = best_pair
            if horizontal_score > vertical_score * 1.35:
                arrangement = 'horizontal_components'
            elif vertical_score > horizontal_score * 1.15:
                arrangement = 'vertical_components'
    elif significant:
        component = significant[0]
        normalized_width = component['width'] / max(1, small_width)
        normalized_height = component['height'] / max(1, small_height)
        horizontal_score, vertical_score = (normalized_width, normalized_height)
        if normalized_width > normalized_height * 2.0:
            arrangement = 'horizontal_elongated_component'
        elif normalized_height > normalized_width * 1.25:
            arrangement = 'vertical_elongated_component'
    margin_x = max(1, int(round(small_width * 0.05)))
    margin_y = max(1, int(round(small_height * 0.05)))
    left_contact = float(np.any(small[:, :margin_x], axis=1).mean())
    right_contact = float(np.any(small[:, -margin_x:], axis=1).mean())
    bottom_contact = float(np.any(small[-margin_y:, :], axis=0).mean())
    horizontal_component_evidence = arrangement == 'horizontal_components' and len(significant) >= 2 and (horizontal_score >= 0.12) and (horizontal_score > vertical_score * 3.0)
    vertical_component_evidence = arrangement.startswith('vertical_')
    side_to_side_support = left_contact >= 0.04 and right_contact >= 0.04 and (bottom_contact < 0.08)
    if bottom_contact >= 0.12 or vertical_component_evidence:
        preferred_axis = 'forward_y'
        selection_basis = 'bottom_or_vertical_component_support'
    elif side_to_side_support and horizontal_component_evidence:
        preferred_axis = 'horizontal_x'
        selection_basis = 'two_side_contacts_without_bottom_contact'
    else:
        preferred_axis = 'forward_y'
        selection_basis = 'ambiguous_ground_view_default'
    return {'arrangement': arrangement, 'horizontal_score': float(horizontal_score), 'vertical_score': float(vertical_score), 'significant_components': len(significant), 'left_contact': left_contact, 'right_contact': right_contact, 'bottom_contact': bottom_contact, 'side_to_side_support': bool(side_to_side_support), 'preferred_axis': preferred_axis, 'selection_basis': selection_basis}

# 【數值工具】以滑動視窗平滑一維數值序列，同時處理缺值。
def smooth_scalar_series(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float64, copy=True)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    padded = np.pad(values.astype(np.float64), radius, mode='edge')
    median = np.asarray([np.median(padded[index:index + window]) for index in range(len(values))], dtype=np.float64)
    return np.convolve(np.pad(median, radius, mode='edge'), np.ones(window, dtype=np.float64) / window, mode='valid')

# 【逐帶工具】將帶索引換算成影像中的起訖列座標。
def band_vertical_bounds(height: int, band_height: int, band_index: int) -> tuple[int, int]:
    bottom = max(0, height - band_index * band_height)
    top = max(0, bottom - band_height)
    return (top, bottom)

# 【數值工具】找出布林序列中所有連續 True 區間。
def contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(part[0]), int(part[-1])) for part in np.split(indices, cuts)]

# 【數值工具】合併間隔小於門檻的一維區間，消除細小斷裂。
def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for left, right in runs[1:]:
        old_left, old_right = merged[-1]
        if left - old_right - 1 <= max(0, int(max_gap)):
            merged[-1] = (old_left, right)
        else:
            merged.append((left, right))
    return merged

# =============================================================================
# 04. V12 逐帶 Road 觀測與同一道路追蹤
# =============================================================================
# 沿候選方向逐帶讀取 Road 左右範圍，從近端開始維持道路身份，避免跳到
# 平行道路或岔路；輸出中心、寬度、左右邊界與缺口位置。
def tracked_road_band_observations(road_mask: np.ndarray, p: Parameters, seed_rank: int=0) -> list[dict]:
    # 【主要功能：逐帶追蹤同一道路】
    # 輸入 Road 遮罩與種子順位；每一帶保留最符合前一帶位置、寬度與
    # 重疊關係的 Road 區段。輸出是 reconstruct_road_corridor 的直接觀測。
    height, width = road_mask.shape
    band_height = max(2, int(p.row_band_height))
    total_bands = int(math.ceil(height / band_height))
    candidates_by_band: dict[int, list[tuple[float, float]]] = {}
    for band in range(total_bands):
        top, bottom = band_vertical_bounds(height, band_height, band)
        column_support = road_mask[top:bottom].sum(axis=0)
        visible_columns = column_support >= max(1, int(p.row_min_valid_rows))
        runs = [run for run in contiguous_true_runs(visible_columns) if run[1] - run[0] + 1 >= p.min_road_width]
        runs = merge_close_runs(runs, p.band_segment_merge_gap_px)
        if runs:
            candidates_by_band[band] = [(float(left), float(right)) for left, right in runs]
    if not candidates_by_band:
        return []
    ranked_seed_options = sorted(((band, run) for band, runs in candidates_by_band.items() for run in runs), key=lambda item: (item[1][1] - item[1][0] + 1.0, -abs((item[1][0] + item[1][1]) / 2.0 - (width - 1) / 2.0), -item[0]), reverse=True)
    seed_options: list[tuple[int, tuple[float, float]]] = []
    for candidate in ranked_seed_options:
        _, (left, right) = candidate
        width_candidate = right - left + 1.0
        duplicate = False
        for _, (kept_left, kept_right) in seed_options:
            overlap = min(right, kept_right) - max(left, kept_left) + 1.0
            minimum_width = min(width_candidate, kept_right - kept_left + 1.0)
            if overlap >= 0.35 * minimum_width:
                duplicate = True
                break
        if not duplicate:
            seed_options.append(candidate)
    if seed_rank < 0 or seed_rank >= len(seed_options):
        return []
    seed_band, (seed_left, seed_right) = seed_options[int(seed_rank)]

    def make_observation(band: int, left: float, right: float, **metadata) -> dict:
        top, bottom = band_vertical_bounds(height, band_height, band)
        return {'band': int(band), 'y': float((top + bottom - 1) / 2.0), 'left_x': float(left), 'right_x': float(right), **metadata}
    seed = make_observation(seed_band, seed_left, seed_right)

    def follow(direction: int) -> list[dict]:
        selected = [seed]
        bands_to_visit = range(seed_band + direction, total_bands if direction > 0 else -1, direction)
        for band in bands_to_visit:
            runs = candidates_by_band.get(band, [])
            if not runs:
                continue
            recent = selected[-min(12, len(selected)):]
            recent_bands = np.asarray([item['band'] for item in recent], dtype=np.float64)
            recent_left = np.asarray([item['left_x'] for item in recent], dtype=np.float64)
            recent_right = np.asarray([item['right_x'] for item in recent], dtype=np.float64)
            left_line = robust_endpoint_line(recent_bands, recent_left, direction)
            right_line = robust_endpoint_line(recent_bands, recent_right, direction)
            predicted_left = float(np.polyval(left_line, band))
            predicted_right = float(np.polyval(right_line, band))
            if predicted_left > predicted_right:
                predicted_left, predicted_right = (predicted_right, predicted_left)
            reference_width = max(float(np.median(recent_right - recent_left + 1.0)), float(p.min_road_width))
            predicted_center = (predicted_left + predicted_right) / 2.0
            margin = reference_width * max(0.0, p.band_tracking_margin_fraction)
            eligible = [run for run in runs if run[1] >= predicted_left - margin and run[0] <= predicted_right + margin]
            fragment_pair_metadata: dict = {}
            if not eligible:
                missing_bands = abs(band - int(selected[-1]['band'])) - 1
                if missing_bands > p.row_max_visible_continuation_gap_bands:
                    if not p.enable_v25_navigation_guided_reconstruction or missing_bands > int(p.v13_fragment_pair_max_gap_bands):
                        continue
                    nearest = min(runs, key=lambda run: abs((run[0] + run[1]) / 2.0 - predicted_center))
                    nearest_center = (nearest[0] + nearest[1]) / 2.0
                    nearest_width = nearest[1] - nearest[0] + 1.0
                    center_cost = abs(nearest_center - predicted_center) / max(reference_width, 1.0)
                    width_cost = abs(math.log(max(nearest_width / max(reference_width, 1.0), 1e-06)))
                    pair_cost = center_cost + 0.35 * width_cost
                    if center_cost > float(p.v25_fragment_pair_reacquire_widths) or pair_cost > float(p.v13_fragment_pair_max_cost):
                        continue
                    eligible = [nearest]
                    fragment_pair_metadata = {'fragment_pair_reacquired': True, 'fragment_pair_cost': float(pair_cost), 'fragment_pair_center_cost': float(center_cost), 'fragment_pair_width_cost': float(width_cost)}
                else:
                    reacquire_limit = reference_width * min(1.0, 0.25 + 0.035 * max(0, missing_bands))
                    nearest = min(runs, key=lambda run: abs((run[0] + run[1]) / 2.0 - predicted_center))
                    nearest_center = (nearest[0] + nearest[1]) / 2.0
                    if abs(nearest_center - predicted_center) > reacquire_limit:
                        continue
                    eligible = [nearest]
            left = min((run[0] for run in eligible))
            right = max((run[1] for run in eligible))
            maximum_width = reference_width * max(1.0, p.band_width_growth_limit)
            if right - left + 1.0 > maximum_width:
                clipped_left = max(float(left), predicted_center - maximum_width / 2.0)
                clipped_right = min(float(right), predicted_center + maximum_width / 2.0)
                if clipped_right - clipped_left + 1.0 >= p.min_road_width:
                    left, right = (clipped_left, clipped_right)
            selected.append(make_observation(band, left, right, **fragment_pair_metadata))
        return selected
    upper = follow(1)
    lower = follow(-1)
    combined = list(reversed(lower[1:])) + upper
    return sorted(combined, key=lambda item: item['band'])

# 【觀測函式】以每帶 Road 外包絡建立較寬鬆的道路觀測，供回退路徑使用。
def envelope_road_band_observations(road_mask: np.ndarray, p: Parameters) -> list[dict]:
    height, _ = road_mask.shape
    band_height = max(2, int(p.row_band_height))
    observations: list[dict] = []
    total_bands = int(math.ceil(height / band_height))
    for band in range(total_bands):
        top, bottom = band_vertical_bounds(height, band_height, band)
        left_values, right_values, y_values = ([], [], [])
        for y in range(top, bottom):
            xs = np.flatnonzero(road_mask[y])
            if len(xs) < p.min_road_width:
                continue
            left_values.append(float(xs.min()))
            right_values.append(float(xs.max()))
            y_values.append(float(y))
        if len(y_values) < p.row_min_valid_rows:
            continue
        left = float(np.median(left_values))
        right = float(np.median(right_values))
        if right - left + 1.0 < p.min_road_width:
            continue
        observations.append({'band': band, 'y': float(np.median(y_values)), 'left_x': left, 'right_x': right})
    return observations

# 【重建工具】檢查兩個 Road 錨點間的缺口是否具有足夠障礙物支持。
def gap_has_obstacle_support(obstacles: np.ndarray, first: dict, second: dict, band_height: int, p: Parameters) -> tuple[bool, float]:
    gap = int(second['band'] - first['band'] - 1)
    if gap <= 0:
        return (True, 0.0)
    if gap <= p.row_short_gap_bands:
        return (True, 0.0)
    height, width = obstacles.shape
    supported = 0
    for step, band in enumerate(range(first['band'] + 1, second['band']), 1):
        fraction = step / (gap + 1)
        left = (1.0 - fraction) * first['left_x'] + fraction * second['left_x']
        right = (1.0 - fraction) * first['right_x'] + fraction * second['right_x']
        x0 = max(0, int(math.floor(min(left, right))))
        x1 = min(width, int(math.ceil(max(left, right))) + 1)
        top, bottom = band_vertical_bounds(height, band_height, band)
        if bottom > top and x1 > x0 and np.any(obstacles[top:bottom, x0:x1]):
            supported += 1
    support_fraction = supported / max(1, gap)
    obstacle_supported = gap <= p.row_max_occlusion_gap_bands and support_fraction >= p.row_occlusion_min_fraction
    visible_road_continuation = gap <= int(p.row_max_visible_continuation_gap_bands)
    geometry_fragment_pair = bool(p.enable_v25_navigation_guided_reconstruction and second.get('fragment_pair_reacquired', False) and (gap <= int(p.v13_fragment_pair_max_gap_bands)))
    return (obstacle_supported or visible_road_continuation or geometry_fragment_pair, float(support_fraction))

# 【擬合工具】利用端點附近可靠觀測估計道路邊界的穩健延伸直線。
def robust_endpoint_line(bands: np.ndarray, values: np.ndarray, direction: int) -> np.ndarray:
    bands = np.asarray(bands, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(bands) < 2:
        return np.asarray([0.0, float(values[0])], dtype=np.float64)
    delta_band = np.diff(bands)
    valid = np.abs(delta_band) > 1e-09
    slopes = np.diff(values)[valid] / delta_band[valid]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    anchor_index = -1 if direction > 0 else 0
    anchor_band = float(bands[anchor_index])
    anchor_value = float(values[anchor_index])
    return np.asarray([slope, anchor_value - slope * anchor_band], dtype=np.float64)

# 【擬合工具】以加權觀測誤差加二階平滑懲罰擬合道路曲線。
def regularized_curve_fit(targets: np.ndarray, weights: np.ndarray, smoothness: float, robust_iterations: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    base_weights = np.asarray(weights, dtype=np.float64)
    count = len(targets)
    if count <= 2:
        return targets.copy()
    second_difference = np.zeros((count - 2, count), dtype=np.float64)
    indices = np.arange(count - 2)
    second_difference[indices, indices] = 1.0
    second_difference[indices, indices + 1] = -2.0
    second_difference[indices, indices + 2] = 1.0
    penalty = max(0.0, float(smoothness)) * (second_difference.T @ second_difference)
    effective = base_weights.copy()
    fitted = targets.copy()
    for _ in range(max(1, int(robust_iterations))):
        matrix = np.diag(effective) + penalty + np.eye(count) * 1e-08
        fitted = np.linalg.solve(matrix, effective * targets)
        observed = base_weights >= 0.99
        if observed.sum() < 3:
            continue
        residual = fitted[observed] - targets[observed]
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-06
        threshold = max(1.0, 2.5 * scale)
        factors = np.ones_like(residual)
        large = np.abs(residual) > threshold
        factors[large] = threshold / np.abs(residual[large])
        effective = base_weights.copy()
        effective[np.flatnonzero(observed)] *= factors
    return fitted

# 【擬合工具】由錨點前後的可靠邊界觀測估計穩健斜率。
def _robust_boundary_slope(observations: list[dict], boundary_key: str, anchor_band: int, before_anchor: bool, count: int) -> float | None:
    ordered = sorted(observations, key=lambda item: int(item['band']))
    if before_anchor:
        local = [item for item in ordered if int(item['band']) <= anchor_band]
        local = local[-max(2, int(count)):]
    else:
        local = [item for item in ordered if int(item['band']) >= anchor_band]
        local = local[:max(2, int(count))]
    if len(local) < 2:
        return None
    bands = np.asarray([item['band'] for item in local], dtype=np.float64)
    values = np.asarray([item[boundary_key] for item in local], dtype=np.float64)
    delta = np.diff(bands)
    valid = np.abs(delta) > 1e-09
    slopes = np.diff(values)[valid] / delta[valid]
    return float(np.median(slopes)) if len(slopes) else None

# 【重建函式】以 Hermite 曲線連接已配對道路片段的左右邊界。
def apply_fragment_pair_hermite_boundaries(smooth_left: np.ndarray, smooth_right: np.ndarray, dense_bands: np.ndarray, observations: list[dict], accepted_gaps: list[dict], p: Parameters) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    left = np.asarray(smooth_left, dtype=np.float64).copy()
    right = np.asarray(smooth_right, dtype=np.float64).copy()
    band_to_index = {int(value): index for index, value in enumerate(dense_bands)}
    records: list[dict] = []

    def hermite(start: float, end: float, start_slope: float, end_slope: float, span: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, span + 1)
        h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
        h10 = t ** 3 - 2.0 * t ** 2 + t
        h01 = -2.0 * t ** 3 + 3.0 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00 * start + h10 * span * start_slope + h01 * end + h11 * span * end_slope
    for gap in accepted_gaps:
        if gap.get('connection_basis') != 'geometry_fragment_pair':
            continue
        start_band = int(gap['start_band'])
        end_band = int(gap['end_band'])
        if start_band not in band_to_index or end_band not in band_to_index:
            continue
        start_index = band_to_index[start_band]
        end_index = band_to_index[end_band]
        span = end_index - start_index
        if span <= 2:
            continue
        local_width = max(1.0, float(right[start_index] - left[start_index] + 1.0), float(right[end_index] - left[end_index] + 1.0))
        generated: dict[str, np.ndarray] = {}
        slopes: dict[str, tuple[float, float]] = {}
        for name, values, key in (('left', left, 'left_x'), ('right', right, 'right_x')):
            secant = float((values[end_index] - values[start_index]) / span)
            start_slope = _robust_boundary_slope(observations, key, start_band, True, p.v13_fragment_tangent_observations)
            end_slope = _robust_boundary_slope(observations, key, end_band, False, p.v13_fragment_tangent_observations)
            if start_slope is None:
                start_slope = secant
            if end_slope is None:
                end_slope = secant
            slope_limit = abs(secant) + 2.0 * local_width / max(1, span)
            start_slope = float(np.clip(start_slope, -slope_limit, slope_limit))
            end_slope = float(np.clip(end_slope, -slope_limit, slope_limit))
            curve = hermite(float(values[start_index]), float(values[end_index]), start_slope, end_slope, span)
            overshoot = local_width * float(p.v13_fragment_hermite_overshoot_width_fraction)
            lower = min(values[start_index], values[end_index]) - overshoot
            upper = max(values[start_index], values[end_index]) + overshoot
            generated[name] = np.clip(curve, lower, upper)
            slopes[name] = (start_slope, end_slope)
        left[start_index + 1:end_index] = generated['left'][1:-1]
        right[start_index + 1:end_index] = generated['right'][1:-1]
        records.append({'start_band': start_band, 'end_band': end_band, 'missing_bands': int(gap.get('missing_bands', span - 1)), 'curve_model': 'cubic_hermite_boundary_pair', 'left_endpoint_slopes': [float(slopes['left'][0]), float(slopes['left'][1])], 'right_endpoint_slopes': [float(slopes['right'][0]), float(slopes['right'][1])]})
    return (left, right, records)

# 【末端工具】移除被障礙物壓縮的末端 Road 碎片，避免其拉偏外推方向。
def trim_terminal_occluded_observations(observations: list[dict], obstacles: np.ndarray, direction: int, band_height: int, p: Parameters) -> tuple[list[dict], list[dict]]:
    if len(observations) < 5:
        return (observations, [])
    ordered = sorted(observations, key=lambda item: item['band'])
    scan = ordered if direction > 0 else list(reversed(ordered))
    suspicious = [False] * len(scan)
    height, width = obstacles.shape
    trusted = list(scan[:4])
    for index in range(4, len(scan)):
        reference = trusted[-p.one_sided_fit_observations:]
        bands = np.asarray([item['band'] for item in reference], dtype=np.float64)
        centers = np.asarray([(item['left_x'] + item['right_x']) / 2.0 for item in reference], dtype=np.float64)
        half_widths = np.asarray([(item['right_x'] - item['left_x'] + 1.0) / 2.0 for item in reference], dtype=np.float64)
        target_band = int(scan[index]['band'])
        predicted_center = float(np.polyval(np.polyfit(bands, centers, 1), target_band))
        predicted_half_width = float(np.polyval(np.polyfit(bands, half_widths, 1), target_band))
        predicted_half_width = float(np.clip(predicted_half_width, p.min_road_width / 2.0, width / 2.0))
        x0 = max(0, int(math.floor(predicted_center - predicted_half_width)))
        x1 = min(width, int(math.ceil(predicted_center + predicted_half_width)) + 1)
        top, bottom = band_vertical_bounds(height, band_height, target_band)
        if x1 <= x0 or bottom <= top:
            continue
        obstacle_columns = np.any(obstacles[top:bottom, x0:x1], axis=0)
        obstacle_fraction = float(obstacle_columns.mean()) if obstacle_columns.size else 0.0
        observed_width = float(scan[index]['right_x'] - scan[index]['left_x'] + 1.0)
        predicted_width = max(float(p.min_road_width), predicted_half_width * 2.0)
        visible_width_ratio = observed_width / predicted_width
        obstacle_threshold = p.v24_obstacle_min_cross_section_fraction if p.enable_v24_obstacle_driven_reconstruction else p.one_sided_min_obstacle_width_fraction
        width_ratio_threshold = p.v24_lateral_visible_width_ratio if p.enable_v24_obstacle_driven_reconstruction else p.terminal_fragment_max_width_ratio
        suspicious[index] = obstacle_fraction >= obstacle_threshold and visible_width_ratio <= width_ratio_threshold
        if not suspicious[index]:
            trusted.append(scan[index])
    if not suspicious[-1]:
        return (observations, [])
    first_suspicious = len(scan) - 1
    empty = 0
    for index in range(len(scan) - 1, 3, -1):
        if suspicious[index]:
            first_suspicious = index
            empty = 0
        else:
            empty += 1
            if empty > p.one_sided_empty_tolerance_bands:
                break
            first_suspicious = index
    kept_scan = scan[:first_suspicious]
    removed_scan = scan[first_suspicious:]
    if len(kept_scan) < 2:
        return (observations, [])
    kept = sorted(kept_scan, key=lambda item: item['band'])
    removed = sorted(removed_scan, key=lambda item: item['band'])
    return (kept, removed)

# 【末端重建】沿近期可靠道路趨勢，在障礙支持下產生保守末端觀測。
def terminal_obstacle_extrapolation(road_mask: np.ndarray, obstacles: np.ndarray, selected: list[dict], direction: int, band_height: int, p: Parameters) -> list[dict]:
    height, width = road_mask.shape
    total_bands = int(math.ceil(height / band_height))
    ordered = sorted(selected, key=lambda item: item['band'])
    reliable = [item for item in ordered if item.get('source_type', 'direct') == 'direct']
    if len(reliable) < 2:
        reliable = ordered
    fit_count = min(len(reliable), max(2, int(p.one_sided_fit_observations)))
    fitting = reliable[-fit_count:] if direction > 0 else reliable[:fit_count]
    bands = np.asarray([item['band'] for item in fitting], dtype=np.float64)
    centers = np.asarray([(item['left_x'] + item['right_x']) / 2.0 for item in fitting], dtype=np.float64)
    half_widths = np.asarray([(item['right_x'] - item['left_x'] + 1.0) / 2.0 for item in fitting], dtype=np.float64)
    left_values = centers - half_widths
    right_values = centers + half_widths
    left_coefficients = robust_endpoint_line(bands, left_values, direction)
    right_coefficients = robust_endpoint_line(bands, right_values, direction)
    vertical_span_px = float(np.ptp(bands) * band_height)
    horizontal_center_travel = float(abs(centers[-1] - centers[0]))
    if vertical_span_px >= horizontal_center_travel:
        anchor_band = float(bands[-1] if direction > 0 else bands[0])
        anchor_left = float(np.polyval(left_coefficients, anchor_band))
        anchor_right = float(np.polyval(right_coefficients, anchor_band))
        left_coefficients[0] = max(0.0, float(left_coefficients[0]))
        right_coefficients[0] = min(0.0, float(right_coefficients[0]))
        left_coefficients[1] = anchor_left - left_coefficients[0] * anchor_band
        right_coefficients[1] = anchor_right - right_coefficients[0] * anchor_band
    endpoint = reliable[-1] if direction > 0 else reliable[0]
    endpoint_band = int(endpoint['band'])
    median_half_width = max(p.min_road_width / 2.0, float(np.median(half_widths)))
    min_half_width = max(p.min_road_width / 2.0, median_half_width * 0.6)
    max_half_width = min(width / 2.0, median_half_width * max(1.0, p.one_sided_max_width_growth))
    candidates: list[dict] = []
    maximum_steps = total_bands if p.enable_v24_obstacle_driven_reconstruction else max(0, int(p.one_sided_max_bands))
    for step in range(1, maximum_steps + 1):
        band = endpoint_band + direction * step
        if band < 0 or band >= total_bands:
            break
        predicted_left = float(np.polyval(left_coefficients, band))
        predicted_right = float(np.polyval(right_coefficients, band))
        center = (predicted_left + predicted_right) / 2.0
        half_width = float(np.clip(abs(predicted_right - predicted_left) / 2.0, min_half_width, max_half_width))
        left = float(np.clip(center - half_width, 0, width - 1))
        right = float(np.clip(center + half_width, 0, width - 1))
        x0 = max(0, int(math.floor(left)))
        x1 = min(width, int(math.ceil(right)) + 1)
        top, bottom = band_vertical_bounds(height, band_height, band)
        if x1 <= x0 or bottom <= top:
            break
        obstacle_patch = obstacles[top:bottom, x0:x1]
        obstacle_columns = np.any(obstacle_patch, axis=0)
        width_fraction = float(obstacle_columns.mean()) if obstacle_columns.size else 0.0
        threshold = p.v24_obstacle_min_cross_section_fraction if p.enable_v24_obstacle_driven_reconstruction else p.one_sided_min_obstacle_width_fraction
        supported = width_fraction >= threshold
        obstacle_xs = np.flatnonzero(obstacle_columns)
        support_interval = None
        if len(obstacle_xs):
            support_interval = (int(x0 + obstacle_xs.min()), int(x0 + obstacle_xs.max()))
        component_connected = supported
        road_hit = bool(np.any(road_mask[top:bottom, x0:x1]))
        if not supported:
            break
        item = {'band': int(band), 'center_x': center, 'half_width': half_width, 'obstacle_width_fraction': width_fraction, 'supported': supported, 'component_connected': component_connected, 'support_interval': list(support_interval) if support_interval else None}
        candidates.append(item)
    return candidates

# 【側向重建】依可靠側邊、附近寬度與障礙支持推估單側缺失邊界。
def infer_one_sided_missing_widths(observations: list[dict], obstacles: np.ndarray, band_height: int, p: Parameters) -> tuple[list[dict], list[dict]]:
    if len(observations) < 3:
        return ([dict(item) for item in observations], [])
    height, width = obstacles.shape
    adjusted = [dict(item) for item in observations]
    widths = np.asarray([float(item['right_x'] - item['left_x'] + 1.0) for item in observations])
    radius = max(2, int(p.one_side_inferred_window_bands))
    expected = np.empty_like(widths)
    for index in range(len(widths)):
        lo = max(0, index - radius)
        hi = min(len(widths), index + radius + 1)
        expected[index] = max(widths[index], float(np.percentile(widths[lo:hi], 80)))
    inferences: list[dict] = []
    for index, (item, visible_width, expected_width) in enumerate(zip(adjusted, widths, expected)):
        if expected_width < p.min_road_width:
            continue
        visible_width_ratio = p.v24_lateral_visible_width_ratio if p.enable_v24_obstacle_driven_reconstruction else p.one_side_inferred_width_ratio
        if visible_width >= expected_width * visible_width_ratio:
            continue
        missing = float(expected_width - visible_width)
        if missing < 2.0:
            continue
        top, bottom = band_vertical_bounds(height, band_height, int(item['band']))
        old_left = float(item['left_x'])
        old_right = float(item['right_x'])

        def support(x0: float, x1: float) -> float:
            left = max(0, int(math.floor(min(x0, x1))))
            right = min(width, int(math.ceil(max(x0, x1))) + 1)
            if bottom <= top or right <= left:
                return 0.0
            return float(obstacles[top:bottom, left:right].mean())
        left_support = support(old_left - missing, old_left - 1.0)
        right_support = support(old_right + 1.0, old_right + missing)
        threshold = p.v24_obstacle_min_cross_section_fraction if p.enable_v24_obstacle_driven_reconstruction else p.one_side_inferred_obstacle_fraction
        if max(left_support, right_support) < threshold:
            continue
        if left_support > right_support:
            item['left_x'] = max(0.0, old_left - missing)
            side = 'left'
        else:
            item['right_x'] = min(float(width - 1), old_right + missing)
            side = 'right'
        item['source_type'] = 'one_side_inferred'
        inferences.append({'band': int(item['band']), 'side': side, 'old_left_x': old_left, 'old_right_x': old_right, 'new_left_x': float(item['left_x']), 'new_right_x': float(item['right_x']), 'visible_width_px': float(visible_width), 'expected_width_px': float(expected_width), 'left_obstacle_fraction': left_support, 'right_obstacle_fraction': right_support})
    return (adjusted, inferences)

# 【內部重建】以前後完整 Road 錨點補回中段被障礙遮住的左右邊界。
def infer_interior_occluded_boundaries(observations: list[dict], obstacles: np.ndarray, band_height: int, p: Parameters) -> tuple[list[dict], list[dict]]:
    adjusted = [dict(item) for item in observations]
    if len(adjusted) < 3 or not p.enable_interior_occlusion_inference:
        return (adjusted, [])
    height, image_width = obstacles.shape
    widths = np.asarray([float(item['right_x'] - item['left_x'] + 1.0) for item in adjusted])
    bands = np.asarray([int(item['band']) for item in adjusted], dtype=np.int32)
    radius = max(2, int(p.interior_occlusion_anchor_window_bands))
    inferences: list[dict] = []

    def side_support(top: int, bottom: int, x0: float, x1: float) -> float:
        left = max(0, int(math.floor(min(x0, x1))))
        right = min(image_width, int(math.ceil(max(x0, x1))) + 1)
        if bottom <= top or right <= left:
            return 0.0
        return float(obstacles[top:bottom, left:right].mean())
    for index in range(1, len(adjusted) - 1):
        before_indices = [candidate for candidate in range(index - 1, -1, -1) if bands[index] - bands[candidate] <= radius]
        after_indices = [candidate for candidate in range(index + 1, len(adjusted)) if bands[candidate] - bands[index] <= radius]
        if not before_indices or not after_indices:
            continue
        before = max(before_indices, key=lambda candidate: (widths[candidate], -abs(index - candidate)))
        after = max(after_indices, key=lambda candidate: (widths[candidate], -abs(index - candidate)))
        band_span = int(bands[after] - bands[before])
        if band_span <= 0:
            continue
        fraction = float(bands[index] - bands[before]) / float(band_span)
        expected_left = (1.0 - fraction) * float(observations[before]['left_x']) + fraction * float(observations[after]['left_x'])
        expected_right = (1.0 - fraction) * float(observations[before]['right_x']) + fraction * float(observations[after]['right_x'])
        if expected_left > expected_right:
            expected_left, expected_right = (expected_right, expected_left)
        expected_width = expected_right - expected_left + 1.0
        visible_width = widths[index]
        if expected_width < p.min_road_width:
            continue
        visible_width_ratio = p.v24_lateral_visible_width_ratio if p.enable_v24_obstacle_driven_reconstruction else p.interior_occlusion_width_ratio
        if visible_width >= expected_width * visible_width_ratio:
            continue
        item = adjusted[index]
        old_left = float(item['left_x'])
        old_right = float(item['right_x'])
        top, bottom = band_vertical_bounds(height, band_height, int(item['band']))
        left_support = 0.0
        right_support = 0.0
        if expected_left < old_left - 1.0:
            left_support = side_support(top, bottom, expected_left, old_left - 1.0)
        if expected_right > old_right + 1.0:
            right_support = side_support(top, bottom, old_right + 1.0, expected_right)
        threshold = float(p.v24_obstacle_min_cross_section_fraction if p.enable_v24_obstacle_driven_reconstruction else p.interior_occlusion_obstacle_fraction)
        expand_left = left_support >= threshold
        expand_right = right_support >= threshold
        if not expand_left and (not expand_right):
            continue
        if expand_left:
            item['left_x'] = max(0.0, expected_left)
        if expand_right:
            item['right_x'] = min(float(image_width - 1), expected_right)
        if item['right_x'] - item['left_x'] + 1.0 < p.min_road_width:
            continue
        item['source_type'] = 'interior_occlusion_inferred'
        inferences.append({'band': int(item['band']), 'old_left_x': old_left, 'old_right_x': old_right, 'new_left_x': float(item['left_x']), 'new_right_x': float(item['right_x']), 'visible_width_px': float(visible_width), 'expected_width_px': float(expected_width), 'left_anchor_band': int(bands[before]), 'right_anchor_band': int(bands[after]), 'left_obstacle_fraction': float(left_support), 'right_obstacle_fraction': float(right_support), 'expanded_left': bool(expand_left), 'expanded_right': bool(expand_right)})
    return (adjusted, inferences)

# 【遮罩工具】只使用指定來源的左右邊界區段建立推測道路遮罩。
def mask_from_boundary_sources(shape: tuple[int, int], left_boundary: np.ndarray, right_boundary: np.ndarray, source_types: list[str], included_sources: set[str], line_width: int) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    count = min(len(left_boundary), len(right_boundary), len(source_types))
    selected = [index for index in range(count) if source_types[index] in included_sources]
    for index in selected:
        left = tuple(np.rint(left_boundary[index]).astype(int))
        right = tuple(np.rint(right_boundary[index]).astype(int))
        draw.line((*left, *right), fill=1, width=max(1, int(line_width)))
    selected_set = set(selected)
    for index in selected:
        if index + 1 not in selected_set:
            continue
        polygon = [tuple(np.rint(left_boundary[index]).astype(int)), tuple(np.rint(right_boundary[index]).astype(int)), tuple(np.rint(right_boundary[index + 1]).astype(int)), tuple(np.rint(left_boundary[index + 1]).astype(int))]
        draw.polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)

# =============================================================================
# 05. V12 基礎道路重建
# =============================================================================
# 以逐帶觀測建立平滑中心與左右邊界，處理前後雙錨點缺口、內部遮擋、
# 單側缺失及保守末端外推，產生 V40 後續使用的基礎道路本體。
def reconstruct_road_corridor(road_mask: np.ndarray, obstacles: np.ndarray, p: Parameters, observation_mode: str='tracked', seed_rank: int=0, track_id: int=0) -> dict:
    # 【主要功能：建立一條候選道路的 V12 基礎幾何】
    # 1. 取得逐帶左右 Road 觀測；2. 判斷內部缺口與末端遮擋；
    # 3. 分別擬合中心與半寬；4. 輸出中心線、左右邊界、來源及重建遮罩。
    # 本函式處理單一方向／單一種子，不負責在多個候選中選出主要道路。
    height, width = road_mask.shape
    band_height = max(2, int(p.row_band_height))
    total_bands = int(math.ceil(height / band_height))
    if observation_mode == 'envelope':
        observations = envelope_road_band_observations(road_mask, p)
    else:
        observations = tracked_road_band_observations(road_mask, p, seed_rank=seed_rank)
    if len(observations) < 2:
        return {'ok': False, 'reason': 'too_few_boundary_bands'}
    groups: list[list[dict]] = [[observations[0]]]
    accepted_gap_info = []
    for observation in observations[1:]:
        previous = groups[-1][-1]
        accepted, support_fraction = gap_has_obstacle_support(obstacles, previous, observation, band_height, p)
        if accepted:
            groups[-1].append(observation)
            gap = observation['band'] - previous['band'] - 1
            if gap > 0:
                obstacle_supported = gap <= p.row_max_occlusion_gap_bands and support_fraction >= p.row_occlusion_min_fraction
                accepted_gap_info.append({'start_band': previous['band'], 'end_band': observation['band'], 'missing_bands': int(gap), 'obstacle_support_fraction': support_fraction, 'connection_basis': 'obstacle_supported' if obstacle_supported else 'geometry_fragment_pair' if observation.get('fragment_pair_reacquired', False) else 'visible_road_continuation', 'fragment_pair_cost': observation.get('fragment_pair_cost')})
        else:
            groups.append([observation])
    selected = max(groups, key=lambda group: (group[-1]['band'] - group[0]['band'] + 1, len(group), -group[0]['band']))
    if len(selected) < 2:
        return {'ok': False, 'reason': 'corridor_segment_too_short'}
    selected, removed_lower = trim_terminal_occluded_observations(selected, obstacles, -1, band_height, p)
    selected, removed_upper = trim_terminal_occluded_observations(selected, obstacles, 1, band_height, p)
    if len(selected) < 2:
        return {'ok': False, 'reason': 'too_few_reliable_boundary_bands'}
    direct_selected = [dict(item) for item in selected]
    selected, interior_occlusion_inferences = infer_interior_occluded_boundaries(selected, obstacles, band_height, p)
    if p.enable_one_side_lateral_inference:
        selected, one_side_inferences = infer_one_sided_missing_widths(selected, obstacles, band_height, p)
    else:
        selected = [dict(item) for item in selected]
        one_side_inferences = []
    internal_reconstruction_records = [{**dict(item), 'core_state': 'internal_reconstruction', 'detail_source': 'interior_occlusion_inferred'} for item in interior_occlusion_inferences] + [{**dict(item), 'core_state': 'internal_reconstruction', 'detail_source': 'one_side_inferred'} for item in one_side_inferences]
    lower_extension = terminal_obstacle_extrapolation(road_mask, obstacles, selected, -1, band_height, p)
    upper_extension = terminal_obstacle_extrapolation(road_mask, obstacles, selected, 1, band_height, p)
    observation_bands = np.asarray([item['band'] for item in selected], dtype=np.float64)
    dense_start = int(observation_bands[0])
    dense_end = int(observation_bands[-1])
    if lower_extension:
        dense_start = min(dense_start, min((item['band'] for item in lower_extension)))
    if upper_extension:
        dense_end = max(dense_end, max((item['band'] for item in upper_extension)))
    dense_bands = np.arange(dense_start, dense_end + 1, dtype=np.float64)
    raw_left = np.asarray([item['left_x'] for item in selected], dtype=np.float64)
    raw_right = np.asarray([item['right_x'] for item in selected], dtype=np.float64)
    raw_y = np.asarray([item['y'] for item in selected], dtype=np.float64)
    direct_raw_left = np.asarray([item['left_x'] for item in direct_selected], dtype=np.float64)
    direct_raw_right = np.asarray([item['right_x'] for item in direct_selected], dtype=np.float64)
    direct_raw_y = np.asarray([item['y'] for item in direct_selected], dtype=np.float64)
    raw_centers = (raw_left + raw_right) / 2.0
    raw_half_widths = (raw_right - raw_left + 1.0) / 2.0
    center_targets = np.interp(dense_bands, observation_bands, raw_centers)
    width_targets = np.interp(dense_bands, observation_bands, raw_half_widths)
    fit_weights = np.zeros(len(dense_bands), dtype=np.float64)
    dense_index = {int(band): index for index, band in enumerate(dense_bands)}
    for item, band, center, half_width in zip(selected, observation_bands, raw_centers, raw_half_widths):
        index = dense_index[int(band)]
        center_targets[index] = center
        width_targets[index] = half_width
        source_type = item.get('source_type', 'direct')
        if source_type == 'interior_occlusion_inferred':
            fit_weights[index] = p.interior_occlusion_soft_weight
        elif source_type == 'one_side_inferred':
            fit_weights[index] = p.one_sided_soft_weight
        else:
            fit_weights[index] = 1.0
    for item in lower_extension + upper_extension:
        index = dense_index[int(item['band'])]
        center_targets[index] = item['center_x']
        width_targets[index] = item['half_width']
        fit_weights[index] = p.one_sided_soft_weight
    centers = regularized_curve_fit(center_targets, fit_weights, p.center_curve_smoothness, p.curve_robust_iterations)
    half_widths = regularized_curve_fit(width_targets, fit_weights, p.width_curve_smoothness, p.curve_robust_iterations)
    centers = np.clip(centers, 0, width - 1)
    half_widths = np.clip(half_widths, p.min_road_width / 2.0, width / 2.0)
    dense_y = np.asarray([(sum(band_vertical_bounds(height, band_height, int(band))) - 1) / 2.0 for band in dense_bands], dtype=np.float64)
    smooth_left = np.clip(centers - half_widths, 0, width - 1)
    smooth_right = np.clip(centers + half_widths, 0, width - 1)
    first_observed_index = dense_index[int(observation_bands[0])]
    last_observed_index = dense_index[int(observation_bands[-1])]
    if lower_extension:
        for index in range(first_observed_index - 1, -1, -1):
            smooth_left[index] = min(smooth_left[index], smooth_left[index + 1])
            smooth_right[index] = max(smooth_right[index], smooth_right[index + 1])
    if upper_extension:
        for index in range(last_observed_index + 1, len(dense_bands)):
            smooth_left[index] = max(smooth_left[index], smooth_left[index - 1])
            smooth_right[index] = min(smooth_right[index], smooth_right[index - 1])
    if p.enable_v25_navigation_guided_reconstruction:
        smooth_left, smooth_right, fragment_pair_curve_records = apply_fragment_pair_hermite_boundaries(smooth_left, smooth_right, dense_bands, selected, accepted_gap_info, p)
    else:
        fragment_pair_curve_records = []
    too_narrow = smooth_right - smooth_left + 1.0 < p.min_road_width
    if np.any(too_narrow):
        narrow_centers = (smooth_left[too_narrow] + smooth_right[too_narrow]) / 2.0
        smooth_left[too_narrow] = narrow_centers - (p.min_road_width - 1.0) / 2.0
        smooth_right[too_narrow] = narrow_centers + (p.min_road_width - 1.0) / 2.0
    centers = (smooth_left + smooth_right) / 2.0
    half_widths = (smooth_right - smooth_left + 1.0) / 2.0
    centerline = np.column_stack((centers, dense_y))
    left_boundary = np.column_stack((smooth_left, dense_y))
    right_boundary = np.column_stack((smooth_right, dense_y))
    corridor_canvas = Image.new('1', (width, height), 0)
    corridor_draw = ImageDraw.Draw(corridor_canvas)
    polygon = [tuple(point) for point in left_boundary]
    polygon += [tuple(point) for point in right_boundary[::-1]]
    if len(polygon) >= 4:
        corridor_draw.polygon(polygon, fill=1)
    inferred_geometry_envelope = np.asarray(corridor_canvas, dtype=bool)
    inferred_occlusion_mask = inferred_geometry_envelope & obstacles & ~road_mask
    observed_set = {int(value) for value in observation_bands}
    first_observed, last_observed = (min(observed_set), max(observed_set))
    interpolated_flags = [first_observed <= int(value) <= last_observed and int(value) not in observed_set for value in dense_bands]
    extrapolated_flags = [int(value) < first_observed or int(value) > last_observed for value in dense_bands]
    one_side_inferred_set = {int(item['band']) for item in one_side_inferences}
    interior_occlusion_inferred_set = {int(item['band']) for item in interior_occlusion_inferences}
    band_source_types = []
    for band, interpolated, extrapolated in zip(dense_bands, interpolated_flags, extrapolated_flags):
        integer_band = int(band)
        if extrapolated:
            source = 'extrapolated'
        elif interpolated:
            source = 'interpolated'
        elif integer_band in interior_occlusion_inferred_set:
            source = 'interior_occlusion_inferred'
        elif integer_band in one_side_inferred_set:
            source = 'one_side_inferred'
        else:
            source = 'direct'
        band_source_types.append(source)
    band_core_states = [reconstruction_state_for_source(source) for source in band_source_types]
    band_support_evidence = [reconstruction_evidence_for_source(source) for source in band_source_types]
    if p.enable_v24_obstacle_driven_reconstruction:
        obstacle_supported_reconstruction_mask = mask_from_boundary_sources(road_mask.shape, left_boundary, right_boundary, band_source_types, {'interior_occlusion_inferred', 'one_side_inferred', 'extrapolated'}, line_width=band_height) & inferred_geometry_envelope & ~road_mask
    else:
        obstacle_supported_reconstruction_mask = np.zeros_like(road_mask, dtype=bool)
    if p.include_unknown_internal_gaps:
        inferred_internal_gap_mask = mask_from_boundary_sources(road_mask.shape, left_boundary, right_boundary, band_source_types, {'interpolated'}, line_width=band_height) & ~road_mask
    else:
        inferred_internal_gap_mask = np.zeros_like(road_mask, dtype=bool)
    full_road_corridor = road_mask | inferred_occlusion_mask | inferred_internal_gap_mask | obstacle_supported_reconstruction_mask
    missing_flags = np.logical_or(interpolated_flags, extrapolated_flags)
    missing_ratio = float(np.mean(missing_flags)) if len(missing_flags) else 0.0
    interpolation_ratio = float(np.mean(interpolated_flags)) if interpolated_flags else 0.0
    extrapolation_ratio = float(np.mean(extrapolated_flags)) if extrapolated_flags else 0.0
    vertical_span = abs(float(centerline[-1, 1] - centerline[0, 1]))
    horizontal_travel = float(np.abs(np.diff(centerline[:, 0])).sum())
    median_width = float(np.median(smooth_right - smooth_left + 1.0))
    warnings = []
    if observation_mode == 'envelope':
        warnings.append('horizontal_envelope_fallback')
    if interpolation_ratio > 0.2:
        warnings.append('large_interpolated_corridor_fraction')
    if lower_extension or upper_extension:
        warnings.append('one_sided_extrapolation_used')
    if horizontal_travel > max(1.0, vertical_span * 1.25):
        warnings.append('corridor_centerline_may_be_unstable')
    if median_width > max(1.0, vertical_span * 1.5):
        warnings.append('road_may_be_horizontal')
    raw_left_points = np.column_stack((direct_raw_left, direct_raw_y))
    raw_right_points = np.column_stack((direct_raw_right, direct_raw_y))
    selected_start, selected_end = (int(observation_bands[0]), int(observation_bands[-1]))
    selected_gap_info = [item for item in accepted_gap_info if selected_start <= int(item['start_band']) <= int(item['end_band']) <= selected_end]
    curve_by_gap = {(int(item['start_band']), int(item['end_band'])): item for item in fragment_pair_curve_records}
    for item in selected_gap_info:
        curve = curve_by_gap.get((int(item['start_band']), int(item['end_band'])))
        if curve is not None:
            item.update({key: value for key, value in curve.items() if key not in {'start_band', 'end_band', 'missing_bands'}})
    one_sided_info = []
    one_sided_geometry = []
    for name, extension in (('lower', lower_extension), ('upper', upper_extension)):
        if extension:
            extension_indices = [dense_index[int(item['band'])] for item in extension]
            one_sided_info.append({'direction': name, 'bands': len(extension), 'start_band': int(extension[0]['band']), 'end_band': int(extension[-1]['band']), 'mean_obstacle_width_fraction': float(np.mean([item['obstacle_width_fraction'] for item in extension]))})
            one_sided_geometry.append({'direction': name, 'centerline': centerline[extension_indices].tolist(), 'left': left_boundary[extension_indices].tolist(), 'right': right_boundary[extension_indices].tolist()})
    return {'ok': True, 'raw_centerline': np.column_stack(((direct_raw_left + direct_raw_right) / 2.0, direct_raw_y)), 'centerline': centerline, 'raw_left': raw_left_points, 'raw_right': raw_right_points, 'left': left_boundary, 'right': right_boundary, 'analysis_geometry_corridor': inferred_geometry_envelope, 'full_road_corridor': full_road_corridor, 'inferred_occlusion_mask': inferred_occlusion_mask, 'obstacle_supported_reconstruction_mask': obstacle_supported_reconstruction_mask, 'inferred_internal_gap_mask': inferred_internal_gap_mask, 'inferred_geometry_envelope': inferred_geometry_envelope, 'corridor_mask': full_road_corridor, 'band_source_types': band_source_types, 'band_core_states': band_core_states, 'band_support_evidence': band_support_evidence, 'internal_reconstruction_records': internal_reconstruction_records, 'one_side_inferences': one_side_inferences, 'interior_occlusion_inferences': interior_occlusion_inferences, 'interpolated_flags': interpolated_flags, 'extrapolated_flags': extrapolated_flags, 'missing_ratio': missing_ratio, 'interpolation_ratio': interpolation_ratio, 'extrapolation_ratio': extrapolation_ratio, 'warnings': warnings, 'observed_bands': len(selected), 'span_bands': len(dense_bands), 'median_width': median_width, 'accepted_gaps': selected_gap_info, 'fragment_pair_curve_records': fragment_pair_curve_records, 'one_sided_extrapolations': one_sided_info, 'one_sided_geometry': one_sided_geometry, 'source_counts': {source: int(sum((value == source for value in band_source_types))) for source in ('direct', 'interpolated', 'interior_occlusion_inferred', 'one_side_inferred', 'extrapolated')}, 'core_state_counts': summarize_reconstruction_sources(band_source_types)[0], 'support_evidence_counts': summarize_reconstruction_sources(band_source_types)[1], 'reconstruction_pipeline': 'unified_three_state', 'discarded_terminal_fragment_bands': len(removed_lower) + len(removed_upper), 'curve_fit_method': 'robust_global_second_difference+cubic_hermite_fragment_pairs' if fragment_pair_curve_records else 'robust_global_second_difference', 'observation_mode': observation_mode, 'track_id': int(track_id), 'seed_rank': int(seed_rank)}

# =============================================================================
# 06. 道路幾何重採樣、座標轉換與分析走廊
# =============================================================================
# 建立沿道路前進的 u 軸與橫向 v 軸，在道路座標中追蹤後再轉回影像；
# 同時負責中心／邊界重採樣、局部切線及分析走廊遮罩。
# 【座標工具】將在轉置影像中建立的走廊座標轉回原影像座標。
def transpose_corridor_to_original_coordinates(corridor: dict) -> dict:
    if not corridor.get('ok'):
        return corridor
    transformed = dict(corridor)
    point_keys = ('raw_centerline', 'centerline', 'raw_left', 'raw_right', 'left', 'right')
    for key in point_keys:
        points = np.asarray(corridor[key], dtype=np.float64)
        transformed[key] = points[:, [1, 0]].copy()
    transformed['full_road_corridor'] = np.asarray(corridor.get('full_road_corridor', corridor['corridor_mask']), dtype=bool).T.copy()
    transformed['inferred_occlusion_mask'] = np.asarray(corridor.get('inferred_occlusion_mask', np.zeros_like(corridor['corridor_mask'])), dtype=bool).T.copy()
    transformed['obstacle_supported_reconstruction_mask'] = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(corridor['corridor_mask'])), dtype=bool).T.copy()
    transformed['inferred_internal_gap_mask'] = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(corridor['corridor_mask'])), dtype=bool).T.copy()
    transformed['inferred_geometry_envelope'] = np.asarray(corridor.get('inferred_geometry_envelope', corridor['corridor_mask']), dtype=bool).T.copy()
    transformed['analysis_geometry_corridor'] = np.asarray(corridor.get('analysis_geometry_corridor', corridor.get('inferred_geometry_envelope', corridor['corridor_mask'])), dtype=bool).T.copy()
    transformed['corridor_mask'] = transformed['full_road_corridor']
    transformed_geometry = []
    for item in corridor.get('one_sided_geometry', []):
        converted = dict(item)
        for key in ('centerline', 'left', 'right'):
            points = np.asarray(item.get(key, []), dtype=np.float64)
            converted[key] = points[:, [1, 0]].tolist() if len(points) else []
        transformed_geometry.append(converted)
    transformed['one_sided_geometry'] = transformed_geometry
    warnings = [warning for warning in corridor.get('warnings', []) if warning not in {'road_may_be_horizontal', 'horizontal_envelope_fallback'}]
    warnings.append('transposed_horizontal_tracking')
    transformed['warnings'] = warnings
    transformed['observation_mode'] = 'transposed_tracked'
    return transformed

# 【幾何工具】計算折線各相鄰點距離的總和。
def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

# 【幾何工具】依固定弧長間距重新取樣折線。
def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 0:
        return points[:1].copy()
    targets = np.arange(0.0, total + 1e-09, max(1.0, spacing))
    if total - targets[-1] > max(1.0, spacing) * 0.25:
        targets = np.append(targets, total)
    return np.column_stack([np.interp(targets, cumulative, points[:, 0]), np.interp(targets, cumulative, points[:, 1])])

# 【幾何工具】以共同站序同步重採樣中心線與左右邊界。
def resample_corridor_geometry(centerline: np.ndarray, left_boundary: np.ndarray, right_boundary: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(centerline), len(left_boundary), len(right_boundary))
    if count < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        return (empty, empty.copy(), empty.copy())
    centerline = np.asarray(centerline[:count], dtype=np.float64)
    left_boundary = np.asarray(left_boundary[:count], dtype=np.float64)
    right_boundary = np.asarray(right_boundary[:count], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return (empty, empty.copy(), empty.copy())
    targets = np.arange(0.0, total + 1e-09, max(1.0, spacing))
    if total - targets[-1] > max(1.0, spacing) * 0.25:
        targets = np.append(targets, total)

    def interpolate(points: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(targets, cumulative, points[:, 0]), np.interp(targets, cumulative, points[:, 1])])
    return (interpolate(centerline), interpolate(left_boundary), interpolate(right_boundary))
SOURCE_WEIGHTS = {'direct': 1.0, 'interpolated': 0.7, 'interior_occlusion_inferred': 0.7, 'roadline_supported_internal': 0.8, 'auxiliary_boundary_inferred': 0.65, 'one_side_inferred': 0.5, 'auxiliary_supported_extrapolated': 0.55, 'extrapolated': 0.35}

# 【狀態工具】將原站點的來源標記同步映射到重採樣後的中心線。
def resample_corridor_sources(centerline: np.ndarray, source_types: list[str], spacing: float) -> list[str]:
    centerline = np.asarray(centerline, dtype=np.float64)
    count = min(len(centerline), len(source_types))
    if count < 2:
        return []
    centerline = centerline[:count]
    sources = list(source_types[:count])
    lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 0:
        return sources[:1]
    targets = np.arange(0.0, total + 1e-09, max(1.0, spacing))
    if total - targets[-1] > max(1.0, spacing) * 0.25:
        targets = np.append(targets, total)
    indices = np.searchsorted(cumulative, targets, side='left')
    indices = np.clip(indices, 0, count - 1)
    for position, index in enumerate(indices):
        if index > 0 and abs(targets[position] - cumulative[index - 1]) <= abs(cumulative[index] - targets[position]):
            indices[position] = index - 1
    return [sources[int(index)] for index in indices]

# 【方向工具】以局部視窗內的點擬合道路切線，降低單點鋸齒影響。
def fitted_local_tangent(stations: np.ndarray, index: int, sample_spacing: float, window_px: float) -> np.ndarray:
    if len(stations) < 2:
        return np.asarray([0.0, -1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(window_px / max(sample_spacing, 1e-06))))
    start = max(0, index - radius)
    end = min(len(stations), index + radius + 1)
    local = stations[start:end]
    if len(local) < 2:
        before = stations[max(0, index - 1)]
        after = stations[min(len(stations) - 1, index + 1)]
        return after - before
    centered = local - local.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    forward = stations[min(len(stations) - 1, index + 1)] - stations[max(0, index - 1)]
    if float(np.dot(tangent, forward)) < 0:
        tangent = -tangent
    return tangent

# 【方向工具】將既有切線場插值到等距重採樣後的中心站點。
def _v31_resample_tangent_field(centerline: np.ndarray, tangents: np.ndarray, spacing: float) -> np.ndarray:
    centerline = np.asarray(centerline, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    count = min(len(centerline), len(tangents))
    if count < 2:
        return np.empty((0, 2), dtype=np.float64)
    centerline, tangents = (centerline[:count], tangents[:count])
    lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-06:
        return tangents[:1]
    targets = np.arange(0.0, total + 1e-09, max(1.0, float(spacing)))
    if total - targets[-1] > max(1.0, float(spacing)) * 0.25:
        targets = np.append(targets, total)
    oriented = tangents.copy()
    oriented /= np.maximum(np.linalg.norm(oriented, axis=1)[:, None], 1e-06)
    for index in range(1, count):
        if float(oriented[index] @ oriented[index - 1]) < 0.0:
            oriented[index] *= -1.0
    angles = np.unwrap(np.arctan2(oriented[:, 1], oriented[:, 0]))
    sampled_angles = np.interp(targets, cumulative, angles)
    return np.column_stack((np.cos(sampled_angles), np.sin(sampled_angles)))

# 【方向工具】融合中心切線與左右邊界配對，產生不易破壞寬度的橫斷方向。
def v21_width_safe_cross_section_tangents(stations: np.ndarray, left: np.ndarray, right: np.ndarray, p: Parameters) -> np.ndarray:
    stations = np.asarray(stations, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    count = min(len(stations), len(left), len(right))
    if count < 2:
        return np.empty((0, 2), dtype=np.float64)
    tangents = np.asarray([fitted_local_tangent(stations, index, max(1.0, float(p.sample_spacing)), max(float(p.tangent_window_px), 3.0 * float(p.sample_spacing))) for index in range(count)], dtype=np.float64)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-06)
    for index in range(1, count):
        if float(tangents[index] @ tangents[index - 1]) < 0.0:
            tangents[index] *= -1.0
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    safe_angles = angles.copy()
    widths = np.linalg.norm(right[:count] - left[:count], axis=1)
    for index in range(1, count):
        spacing = max(1.0, float(np.linalg.norm(stations[index] - stations[index - 1])))
        half_width = max(float(p.min_road_width), 0.25 * float(widths[index - 1] + widths[index]))
        maximum_delta = float(p.v21_cross_section_curvature_safety) * math.atan2(spacing, half_width)
        delta = float(angles[index] - safe_angles[index - 1])
        safe_angles[index] = safe_angles[index - 1] + float(np.clip(delta, -maximum_delta, maximum_delta))
    return np.column_stack((np.cos(safe_angles), np.sin(safe_angles)))

# 【影像工具】以最近鄰方式按比例縮放二值遮罩。
def resize_boolean_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return np.asarray(mask, dtype=bool).copy()
    height, width = mask.shape
    target = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.resize(target, Image.Resampling.NEAREST)) > 0

# 【影像工具】將二值遮罩同步縮放到指定高寬。
def resize_boolean_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if mask.shape == (target_height, target_width):
        return np.asarray(mask, dtype=bool).copy()
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.resize((target_width, target_height), Image.Resampling.NEAREST)) > 0

# 【影像工具】依長邊上限計算分析解析度與縮放比例。
def analysis_target_shape(shape: tuple[int, int], long_edge_limit: int) -> tuple[tuple[int, int], float]:
    height, width = (int(shape[0]), int(shape[1]))
    longest = max(height, width)
    if long_edge_limit <= 0 or longest <= long_edge_limit:
        return ((height, width), 1.0)
    requested_scale = float(long_edge_limit) / float(longest)
    target_height = max(2, int(round(height * requested_scale)))
    target_width = max(2, int(round(width * requested_scale)))
    actual_scale = min(target_height / height, target_width / width)
    return ((target_height, target_width), float(actual_scale))

# 【參數工具】依影像縮放比例同步調整所有像素制門檻。
def scaled_parameters_for_analysis(p: Parameters, scale: float) -> Parameters:
    if scale >= 0.999:
        return p

    def scaled_int(value: int, minimum: int=1) -> int:
        return max(minimum, int(round(float(value) * scale)))
    return replace(p, close_radius=scaled_int(p.close_radius), sample_spacing=max(2.0, float(p.sample_spacing) * scale), row_band_height=scaled_int(p.row_band_height, minimum=2), row_min_valid_rows=scaled_int(p.row_min_valid_rows), band_segment_merge_gap_px=scaled_int(p.band_segment_merge_gap_px), tangent_window_px=max(8.0, float(p.tangent_window_px) * scale), cross_section_gap=scaled_int(p.cross_section_gap), obstacle_link_gap=scaled_int(p.obstacle_link_gap), min_road_width=scaled_int(p.min_road_width, minimum=2), auxiliary_min_component_pixels=max(4, int(round(float(p.auxiliary_min_component_pixels) * scale * scale))), auxiliary_min_component_span_px=max(6.0, float(p.auxiliary_min_component_span_px) * scale), v15_aux_track_max_longitudinal_gap_px=max(24.0, float(p.v15_aux_track_max_longitudinal_gap_px) * scale), v15_aux_track_max_lateral_gap_px=max(6.0, float(p.v15_aux_track_max_lateral_gap_px) * scale))

# 【座標工具】依候選道路角度建立影像與道路 u／v 座標的仿射框架。
def road_coordinate_frame(shape: tuple[int, int], angle_degrees: float) -> dict:
    height, width = shape
    theta = math.radians(float(angle_degrees) % 180.0)
    cosine, sine = (math.cos(theta), math.sin(theta))
    corners = np.asarray([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]])
    u_values = corners[:, 0] * cosine + corners[:, 1] * sine
    v_values = -corners[:, 0] * sine + corners[:, 1] * cosine
    u_min, u_max = (float(u_values.min()), float(u_values.max()))
    v_min, v_max = (float(v_values.min()), float(v_values.max()))
    return {'angle_degrees': float(angle_degrees) % 180.0, 'cosine': cosine, 'sine': sine, 'u_min': u_min, 'v_min': v_min, 'shape': (max(2, int(math.ceil(u_max - u_min)) + 1), max(2, int(math.ceil(v_max - v_min)) + 1)), 'original_shape': (height, width)}

# 【座標工具】將影像遮罩旋轉／投影到道路前進與橫向座標。
def mask_to_road_coordinates(mask: np.ndarray, frame: dict) -> np.ndarray:
    out_height, out_width = frame['shape']
    original_height, original_width = mask.shape
    y_prime, x_prime = np.indices((out_height, out_width), dtype=np.float64)
    u = y_prime + float(frame['u_min'])
    v = x_prime + float(frame['v_min'])
    cosine, sine = (float(frame['cosine']), float(frame['sine']))
    xs = np.rint(cosine * u - sine * v).astype(np.int32)
    ys = np.rint(sine * u + cosine * v).astype(np.int32)
    valid = (xs >= 0) & (xs < original_width) & (ys >= 0) & (ys < original_height)
    transformed = np.zeros((out_height, out_width), dtype=bool)
    transformed[valid] = np.asarray(mask, dtype=bool)[ys[valid], xs[valid]]
    return transformed

# 【座標工具】將道路 u／v 座標中的點轉回原影像座標。
def points_from_road_coordinates(points: np.ndarray, frame: dict) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if not len(values):
        return np.empty((0, 2), dtype=np.float64)
    u = values[:, 1] + float(frame['u_min'])
    v = values[:, 0] + float(frame['v_min'])
    cosine, sine = (float(frame['cosine']), float(frame['sine']))
    return np.column_stack((cosine * u - sine * v, sine * u + cosine * v))

# 【遮罩工具】用左右邊界封閉多邊形，建立連續道路走廊遮罩。
def corridor_polygon_mask(shape: tuple[int, int], left_boundary: np.ndarray, right_boundary: np.ndarray) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    count = min(len(left_boundary), len(right_boundary))
    if count >= 2:
        left = np.asarray(left_boundary[:count], dtype=np.float64)
        right = np.asarray(right_boundary[:count], dtype=np.float64)
        polygon = [tuple(point) for point in left]
        polygon += [tuple(point) for point in right[::-1]]
        ImageDraw.Draw(canvas).polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)

# 【幾何回退】以局部切線重新量測道路邊界，供 V11 消融模式使用。
def local_tangent_refine_corridor(corridor: dict, road_mask: np.ndarray, obstacle_mask: np.ndarray, p: Parameters) -> dict:
    if not corridor.get('ok') or p.local_direction_refine_iterations <= 0:
        return corridor
    transformed = dict(corridor)
    centers = np.asarray(corridor['centerline'], dtype=np.float64).copy()
    left = np.asarray(corridor['left'], dtype=np.float64).copy()
    right = np.asarray(corridor['right'], dtype=np.float64).copy()
    sources = list(corridor.get('band_source_types', []))
    count = min(len(centers), len(left), len(right), len(sources))
    refined_count = 0
    height, width = road_mask.shape
    for _ in range(max(1, int(p.local_direction_refine_iterations))):
        next_centers, next_left, next_right = (centers.copy(), left.copy(), right.copy())
        iteration_count = 0
        for index in range(count):
            if sources[index] != 'direct':
                continue
            tangent = fitted_local_tangent(centers[:count], index, max(1.0, float(p.row_band_height)), float(p.tangent_window_px))
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-06:
                continue
            tangent /= norm
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            pair_vector = right[index] - left[index]
            expected_width = max(float(np.linalg.norm(pair_vector)), p.min_road_width)
            pair_samples = max(2, int(math.ceil(expected_width)) + 1)
            pair_x = np.rint(np.linspace(left[index, 0], right[index, 0], pair_samples)).astype(np.int32)
            pair_y = np.rint(np.linspace(left[index, 1], right[index, 1], pair_samples)).astype(np.int32)
            pair_valid = (pair_x >= 0) & (pair_x < width) & (pair_y >= 0) & (pair_y < height)
            if np.any(obstacle_mask[pair_y[pair_valid], pair_x[pair_valid]]):
                continue
            if float(np.dot(normal, pair_vector)) < 0:
                normal = -normal
            radius = max(int(math.ceil(expected_width * p.local_direction_search_width_factor / 2.0)), int(p.min_road_width))
            offsets = np.arange(-radius, radius + 1, dtype=np.float64)
            xs = np.rint(centers[index, 0] + offsets * normal[0]).astype(np.int32)
            ys = np.rint(centers[index, 1] + offsets * normal[1]).astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not np.any(valid):
                continue
            xs, ys, offsets = (xs[valid], ys[valid], offsets[valid])
            keep = np.ones(len(xs), dtype=bool)
            keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
            xs, ys, offsets = (xs[keep], ys[keep], offsets[keep])
            road_values = road_mask[ys, xs]
            runs = merge_close_runs(contiguous_true_runs(road_values), max(1, int(round(p.cross_section_gap))))
            center_position = int(np.argmin(np.abs(offsets)))
            containing = [run for run in runs if run[0] <= center_position <= run[1]]
            if not containing:
                continue
            run_start, run_end = containing[0]
            measured_width = float(offsets[run_end] - offsets[run_start] + 1.0)
            if measured_width < p.min_road_width:
                continue
            if measured_width < 0.75 * expected_width:
                continue
            if measured_width > expected_width * p.local_direction_search_width_factor:
                continue
            midpoint_offset = float((offsets[run_start] + offsets[run_end]) / 2.0)
            if abs(midpoint_offset) > expected_width * p.local_direction_max_center_shift_fraction:
                continue
            refined_center = centers[index]
            half_width = expected_width / 2.0
            next_left[index] = refined_center - half_width * normal
            next_right[index] = refined_center + half_width * normal
            next_centers[index] = refined_center
            iteration_count += 1
        centers, left, right = (next_centers, next_left, next_right)
        refined_count = max(refined_count, iteration_count)
    transformed['centerline'] = centers
    transformed['left'] = left
    transformed['right'] = right
    transformed['local_direction_refined_sections'] = int(refined_count)
    transformed['local_direction_refined'] = bool(refined_count)
    return transformed

# =============================================================================
# 07. 輔助 segmentation 類別解析與走向曲線提取
# =============================================================================
# 將 Roadline、curb、guardrail、barricade、roadside wall 與 cone 轉成
# 可比較的方向候選與平滑曲線；此階段只建立證據，尚未決定最終中心線。
# 【角度工具】計算兩個軸向角度的最小差值，忽略正反向等價。
def angular_distance_degrees(first: float, second: float) -> float:
    difference = abs((float(first) - float(second)) % 180.0)
    return min(difference, 180.0 - difference)

# 【角度工具】計算目標角度相對參考角度的帶正負軸向差。
def signed_axial_angle_difference(target: float, reference: float) -> float:
    return float((float(target) - float(reference) + 90.0) % 180.0 - 90.0)

# 【資料工具】統一類別名稱的大小寫、底線與空白格式。
def normalized_category_name(name: str) -> str:
    value = str(name).strip().casefold().replace('_', ' ').replace('-', ' ')
    return ' '.join(value.split())

# 【資料工具】由 COCO 類別表解析各種道路方向輔助物的 category id。
def resolve_auxiliary_category_ids(categories: list[dict]) -> dict[str, set[int]]:
    resolved = {name: set() for name in AUXILIARY_DIRECTION_ALIASES}
    for category in categories:
        normalized = normalized_category_name(category.get('name', ''))
        for family, aliases in AUXILIARY_DIRECTION_ALIASES.items():
            if normalized in aliases:
                resolved[family].add(int(category['id']))
    return resolved

# 【資料工具】由 COCO 類別表解析 rock、fallen vegetation 與 debris pile。
def resolve_obstacle_category_ids(categories: list[dict]) -> dict[str, set[int]]:
    resolved = {name: set() for name in OBSTACLE_CATEGORY_ALIASES}
    for category in categories:
        normalized = normalized_category_name(category.get('name', ''))
        for family, aliases in OBSTACLE_CATEGORY_ALIASES.items():
            if normalized in aliases:
                resolved[family].add(int(category['id']))
    return resolved

# 【遮罩工具】檢查所有輔助遮罩尺寸並補齊缺少的類別。
def validate_auxiliary_masks(auxiliary_masks: dict[str, np.ndarray] | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    result = {'road_line': np.zeros(shape, dtype=bool), 'cone': np.zeros(shape, dtype=bool), 'roadside_boundary': np.zeros(shape, dtype=bool)}
    if not auxiliary_masks:
        return result
    key_aliases = {'road_line': 'road_line', 'road line': 'road_line', 'cone': 'cone', 'curb': 'roadside_boundary', 'guardrail': 'roadside_boundary', 'barricade': 'roadside_boundary', 'roadside_wall': 'roadside_boundary', 'roadside wall': 'roadside_boundary', 'roadside_boundary': 'roadside_boundary', 'roadside boundary': 'roadside_boundary'}
    for raw_name, raw_mask in auxiliary_masks.items():
        normalized = normalized_category_name(raw_name)
        canonical = key_aliases.get(normalized)
        if canonical is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f'auxiliary mask {raw_name!r} shape {mask.shape} does not match {shape}')
        result[canonical] |= mask
    return result

# 【遮罩工具】將遮罩拆成符合最小面積的連通元件點集合。
def connected_component_point_sets(mask: np.ndarray, minimum_area: int) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return []
    try:
        import cv2
        count, labels, stats, _ = cv2.connectedComponentsWithStats(values.astype(np.uint8), connectivity=8)
        components = []
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area:
                continue
            ys, xs = np.nonzero(labels == label)
            components.append(np.column_stack((xs, ys)).astype(np.float64))
        return components
    except ImportError:
        pass
    height, width = values.shape
    visited = np.zeros_like(values)
    components: list[np.ndarray] = []
    for start_y, start_x in np.argwhere(values):
        start_y, start_x = (int(start_y), int(start_x))
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        points: list[tuple[float, float]] = []
        while queue:
            y, x = queue.popleft()
            points.append((float(x), float(y)))
            for dy, dx in NEIGHBORS_8:
                ny, nx = (y + dy, x + dx)
                if 0 <= ny < height and 0 <= nx < width and values[ny, nx] and (not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(points) >= minimum_area:
            components.append(np.asarray(points, dtype=np.float64))
    return components

# 【方向工具】以元件點雲的主成分估計延伸角度、跨度與細長程度。
def axial_angle_from_points(points: np.ndarray) -> dict | None:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 2:
        return None
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    major_value = float(max(0.0, eigenvalues[order[-1]]))
    minor_value = float(max(0.0, eigenvalues[order[0]]))
    vector = eigenvectors[:, order[-1]]
    if major_value <= 1e-09:
        return None
    angle = float(math.degrees(math.atan2(vector[1], vector[0])) % 180.0)
    projections = centered @ vector
    span = float(projections.max() - projections.min())
    elongation = float((major_value + 1.0) / (minor_value + 1.0))
    center = values.mean(axis=0)
    return {'angle_degrees': angle, 'span_px': span, 'elongation': elongation, 'center_xy': center.tolist(), 'axis_vector_xy': vector.tolist(), 'endpoint_a_xy': (center + vector * float(projections.min())).tolist(), 'endpoint_b_xy': (center + vector * float(projections.max())).tolist()}

# 【曲線工具】沿主方向分箱並擷取橫向眾數，形成抗支柱干擾的代表脊線。
def longitudinal_modal_ridge_points(points: np.ndarray, angle_degrees: float | None=None, longitudinal_bin_px: float=6.0, normal_mode_bin_px: float=4.0, minimum_points: int=8) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < max(2, int(minimum_points)):
        return values.copy()
    if angle_degrees is None:
        geometry = axial_angle_from_points(values)
        if geometry is None:
            return values.copy()
        angle_degrees = float(geometry['angle_degrees'])
    radians = math.radians(float(angle_degrees))
    axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    longitudinal = values @ axis
    lateral = values @ normal
    u_step = max(1.0, float(longitudinal_bin_px))
    v_step = max(1.0, float(normal_mode_bin_px))
    u_origin = float(longitudinal.min())
    u_bins = np.floor((longitudinal - u_origin) / u_step).astype(np.int32)
    ridge: list[np.ndarray] = []
    previous_lateral: float | None = None
    for u_bin in np.unique(u_bins):
        indices = np.flatnonzero(u_bins == u_bin)
        if not len(indices):
            continue
        local_lateral = lateral[indices]
        v_origin = float(local_lateral.min())
        v_bins = np.floor((local_lateral - v_origin) / v_step).astype(np.int32)
        unique_bins, counts = np.unique(v_bins, return_counts=True)
        best_count = int(counts.max())
        candidates = unique_bins[counts == best_count]
        target = float(previous_lateral) if previous_lateral is not None else float(np.median(lateral))
        candidate_centers = v_origin + (candidates.astype(np.float64) + 0.5) * v_step
        selected_bin = int(candidates[np.argmin(np.abs(candidate_centers - target))])
        selected_indices = indices[v_bins == selected_bin]
        if not len(selected_indices):
            continue
        representative = np.median(values[selected_indices], axis=0)
        ridge.append(representative)
        previous_lateral = float(representative @ normal)
    if len(ridge) < int(minimum_points):
        return values.copy()
    result = np.asarray(ridge, dtype=np.float64)
    order = np.argsort(result @ axis)
    return result[order]

# 【角度工具】計算多個軸向角度的加權平均及一致程度。
def weighted_axial_mean_degrees(angles: Iterable[float], weights: Iterable[float]) -> tuple[float, float]:
    angle_values = np.asarray(list(angles), dtype=np.float64)
    weight_values = np.asarray(list(weights), dtype=np.float64)
    valid = np.isfinite(angle_values) & np.isfinite(weight_values) & (weight_values > 0)
    if not np.any(valid):
        return (0.0, 0.0)
    radians = np.deg2rad(angle_values[valid] * 2.0)
    weights_valid = weight_values[valid]
    cosine = float(np.sum(weights_valid * np.cos(radians)))
    sine = float(np.sum(weights_valid * np.sin(radians)))
    total = float(weights_valid.sum())
    mean = float(math.degrees(math.atan2(sine, cosine)) / 2.0 % 180.0)
    consistency = float(math.hypot(cosine, sine) / max(total, 1e-09))
    return (mean, float(np.clip(consistency, 0.0, 1.0)))

# 【遮罩工具】整理各類輔助 segmentation，保留原始類別供診斷使用。
def raw_auxiliary_masks_by_category(auxiliary_masks: dict[str, np.ndarray] | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    categories = ('road_line', 'cone', 'curb', 'guardrail', 'barricade', 'roadside_wall', 'roadside_boundary')
    result = {name: np.zeros(shape, dtype=bool) for name in categories}
    if not auxiliary_masks:
        return result
    aliases = {'road line': 'road_line', 'roadline': 'road_line', 'cone': 'cone', 'orange traffic cone': 'cone', 'curb': 'curb', 'roadside curb': 'curb', 'guardrail': 'guardrail', 'metal guardrail': 'guardrail', 'barricade': 'barricade', 'yellow and black striped road barrier': 'barricade', 'roadside wall': 'roadside_wall', 'roadside retaining': 'roadside_wall', 'roadside retaining wall': 'roadside_wall', 'roadside boundary': 'roadside_boundary'}
    for raw_name, raw_mask in auxiliary_masks.items():
        category = aliases.get(normalized_category_name(raw_name))
        if category is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f'auxiliary mask {raw_name!r} shape {mask.shape} does not match {shape}')
        result[category] |= mask
    return result

# 【軌跡工具】判斷兩個破碎輔助元件能否依方向與位置合併成同一軌跡。
def _v15_component_track_compatible(first: dict, second: dict, p: Parameters) -> tuple[bool, dict]:
    first_geometry = first['geometry']
    second_geometry = second['geometry']
    angle_delta = angular_distance_degrees(first_geometry['angle_degrees'], second_geometry['angle_degrees'])
    if angle_delta > float(p.v15_aux_track_angle_tolerance_deg):
        return (False, {'angle_delta_deg': float(angle_delta)})
    mean_angle, _ = weighted_axial_mean_degrees([first_geometry['angle_degrees'], second_geometry['angle_degrees']], [first_geometry['span_px'], second_geometry['span_px']])
    radians = math.radians(mean_angle)
    axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    first_points = np.asarray(first['points'], dtype=np.float64)
    second_points = np.asarray(second['points'], dtype=np.float64)
    first_projection = first_points @ axis
    second_projection = second_points @ axis
    first_interval = (float(first_projection.min()), float(first_projection.max()))
    second_interval = (float(second_projection.min()), float(second_projection.max()))
    longitudinal_gap = max(0.0, second_interval[0] - first_interval[1], first_interval[0] - second_interval[1])
    first_center = np.asarray(first_geometry['center_xy'], dtype=np.float64)
    second_center = np.asarray(second_geometry['center_xy'], dtype=np.float64)
    lateral_gap = abs(float((second_center - first_center) @ normal))
    if longitudinal_gap > float(p.v15_aux_track_max_longitudinal_gap_px):
        return (False, {'angle_delta_deg': float(angle_delta), 'longitudinal_gap_px': float(longitudinal_gap), 'lateral_gap_px': float(lateral_gap)})
    if lateral_gap > float(p.v15_aux_track_max_lateral_gap_px):
        return (False, {'angle_delta_deg': float(angle_delta), 'longitudinal_gap_px': float(longitudinal_gap), 'lateral_gap_px': float(lateral_gap)})
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    minimum_interval_span = max(1.0, min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0]))
    overlap_ratio = float(overlap / minimum_interval_span)
    if overlap_ratio > 0.35 and lateral_gap > 2.5:
        return (False, {'angle_delta_deg': float(angle_delta), 'longitudinal_gap_px': float(longitudinal_gap), 'lateral_gap_px': float(lateral_gap), 'overlap_ratio': overlap_ratio})
    center_distance = float(np.linalg.norm(second_center - first_center))
    if center_distance > 1e-06:
        connection_lateral_ratio = lateral_gap / center_distance
        if longitudinal_gap > 0 and connection_lateral_ratio > 0.32:
            return (False, {'angle_delta_deg': float(angle_delta), 'longitudinal_gap_px': float(longitudinal_gap), 'lateral_gap_px': float(lateral_gap), 'connection_lateral_ratio': float(connection_lateral_ratio)})
    return (True, {'angle_delta_deg': float(angle_delta), 'longitudinal_gap_px': float(longitudinal_gap), 'lateral_gap_px': float(lateral_gap), 'overlap_ratio': overlap_ratio})

# 【階段函式】把各類輔助元件串成方向連續的候選軌跡並計算品質。
def build_auxiliary_direction_tracks(auxiliary_masks: dict[str, np.ndarray] | None, shape: tuple[int, int], p: Parameters) -> dict:
    raw_masks = raw_auxiliary_masks_by_category(auxiliary_masks, shape)
    diagonal = max(1.0, math.hypot(float(shape[0]), float(shape[1])))
    accepted_tracks: list[dict] = []
    rejected_tracks: list[dict] = []
    category_summary: dict[str, dict] = {}
    linear_categories = ('road_line', 'curb', 'guardrail', 'barricade', 'roadside_wall', 'roadside_boundary')
    for category in linear_categories:
        components = connected_component_point_sets(raw_masks[category], int(p.auxiliary_min_component_pixels))
        eligible: list[dict] = []
        for index, component in enumerate(components):
            raw_component = np.asarray(component, dtype=np.float64)
            initial_geometry = axial_angle_from_points(raw_component)
            component_points = raw_component
            if category == 'guardrail' and initial_geometry is not None:
                component_points = longitudinal_modal_ridge_points(raw_component, angle_degrees=float(initial_geometry['angle_degrees']), longitudinal_bin_px=float(p.v17_guardrail_longitudinal_bin_px), normal_mode_bin_px=float(p.v17_guardrail_normal_mode_bin_px), minimum_points=int(p.v17_guardrail_min_ridge_points))
            geometry = axial_angle_from_points(component_points)
            record = {'category': category, 'source_family': 'road_line' if category == 'road_line' else 'roadside_boundary', 'points': np.asarray(component_points, dtype=np.float64), 'raw_points': raw_component, 'component_index': int(index), 'geometry': geometry}
            if geometry is None:
                record.update({'accepted': False, 'status': 'degenerate'})
                rejected_tracks.append(record)
                continue
            if geometry['span_px'] < float(p.auxiliary_min_component_span_px):
                record.update({'accepted': False, 'status': 'too_short'})
                rejected_tracks.append(record)
                continue
            if geometry['elongation'] < float(p.auxiliary_min_elongation):
                record.update({'accepted': False, 'status': 'not_directional'})
                rejected_tracks.append(record)
                continue
            eligible.append(record)
        parent = list(range(len(eligible)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first_index: int, second_index: int):
            first_root, second_root = (find(first_index), find(second_index))
            if first_root != second_root:
                parent[second_root] = first_root
        pair_links: list[dict] = []
        for first_index in range(len(eligible)):
            for second_index in range(first_index + 1, len(eligible)):
                compatible, metrics = _v15_component_track_compatible(eligible[first_index], eligible[second_index], p)
                if compatible:
                    union(first_index, second_index)
                    pair_links.append({'first': first_index, 'second': second_index, **metrics})
        groups: dict[int, list[int]] = {}
        for index in range(len(eligible)):
            groups.setdefault(find(index), []).append(index)
        category_accepted = 0
        category_rejected = 0
        category_merged = 0
        for member_indices in groups.values():
            members = [eligible[index] for index in member_indices]
            component_points = [member['points'] for member in members]
            raw_component_points = [member['raw_points'] for member in members]
            combined_points = np.concatenate(component_points, axis=0)
            combined_raw_points = np.concatenate(raw_component_points, axis=0)
            combined_geometry = axial_angle_from_points(combined_points)
            component_angles = [member['geometry']['angle_degrees'] for member in members]
            component_spans = [member['geometry']['span_px'] for member in members]
            mean_angle, consistency = weighted_axial_mean_degrees(component_angles, component_spans)
            component_count = len(members)
            span = 0.0 if combined_geometry is None else float(combined_geometry['span_px'])
            if component_count >= 2:
                accepted = bool(consistency >= float(p.v15_aux_track_min_consistency) and span >= float(p.auxiliary_min_component_span_px) * float(p.v15_aux_track_min_merged_span_factor))
                status = 'merged_consistent' if accepted else 'merged_inconsistent'
            else:
                fragmented_class = len(eligible) >= int(p.v15_aux_track_fragmented_class_count)
                minimum_isolated_span = max(float(p.auxiliary_min_component_span_px) * float(p.v15_aux_track_isolated_span_factor), diagonal * float(p.v15_aux_track_isolated_diagonal_ratio))
                accepted = bool(not fragmented_class or span >= minimum_isolated_span)
                status = 'standalone_consistent' if accepted else 'fragmented_inconsistent'
            track = {'category': category, 'source_family': 'road_line' if category == 'road_line' else 'roadside_boundary', 'points': combined_points, 'raw_points': combined_raw_points, 'components': component_points, 'raw_components': raw_component_points, 'component_count': int(component_count), 'merged_fragments': bool(component_count >= 2), 'accepted': bool(accepted), 'status': status, 'angle_degrees': float(mean_angle), 'angle_consistency': float(consistency), 'span_px': span, 'elongation': 0.0 if combined_geometry is None else float(combined_geometry['elongation']), 'geometry': combined_geometry, 'member_component_indices': [int(member['component_index']) for member in members]}
            if accepted:
                accepted_tracks.append(track)
                category_accepted += 1
                category_merged += int(component_count >= 2)
            else:
                rejected_tracks.append(track)
                category_rejected += 1
        category_summary[category] = {'mask_pixels': int(raw_masks[category].sum()), 'component_count': int(len(components)), 'eligible_component_count': int(len(eligible)), 'accepted_track_count': int(category_accepted), 'rejected_track_count': int(category_rejected), 'merged_track_count': int(category_merged)}
    return {'accepted': accepted_tracks, 'rejected': rejected_tracks, 'summary': category_summary, 'raw_masks': raw_masks}

# 【方向候選】在不依賴既有 Road 結果下建立輔助物全域方向假設。
def independent_auxiliary_direction_hypotheses(auxiliary_masks: dict[str, np.ndarray] | None, shape: tuple[int, int], p: Parameters, road_mask: np.ndarray | None=None) -> list[dict]:
    track_package = build_auxiliary_direction_tracks(auxiliary_masks, shape, p)
    raw_masks = track_package['raw_masks']
    diagonal = max(1.0, math.hypot(*shape))
    items: list[dict] = []
    family_weights = {'road_line': 1.0, 'roadside_boundary': 0.82}
    for track in track_package['accepted']:
        family = str(track['source_family'])
        geometry = track.get('geometry')
        if family not in family_weights or geometry is None:
            continue
        span_score = float(np.clip(float(track['span_px']) / max(1.0, 0.22 * diagonal), 0.12, 1.0))
        elongation_score = float(np.clip((math.log1p(float(track['elongation'])) - math.log(2.0)) / 3.0, 0.15, 1.0))
        merge_support = min(1.2, 1.0 + 0.08 * (int(track['component_count']) - 1))
        items.append({'angle_degrees': float(track['angle_degrees']), 'weight': float(family_weights[family] * span_score * elongation_score * float(track['angle_consistency']) * merge_support), 'source_family': family, 'source_category': str(track['category']), 'span_px': float(track['span_px']), 'elongation': float(track['elongation']), 'center_xy': geometry['center_xy'], 'axis_vector_xy': geometry['axis_vector_xy'], 'endpoint_a_xy': geometry['endpoint_a_xy'], 'endpoint_b_xy': geometry['endpoint_b_xy'], 'fragment_count': int(track['component_count']), 'track_points': np.asarray(track['points'], dtype=np.float64)})
    cone_components = connected_component_point_sets(raw_masks['cone'], max(2, int(p.auxiliary_min_component_pixels // 2)))
    if len(cone_components) >= 3:
        centroids = np.asarray([component.mean(axis=0) for component in cone_components])
        geometry = axial_angle_from_points(centroids)
        if geometry is not None and geometry['span_px'] >= p.auxiliary_min_component_span_px:
            items.append({'angle_degrees': float(geometry['angle_degrees']), 'weight': float(0.35 * np.clip(geometry['span_px'] / max(1.0, 0.25 * diagonal), 0.15, 1.0)), 'source_family': 'cone_sequence', 'span_px': float(geometry['span_px']), 'elongation': float(geometry['elongation'])})
    if not items:
        return []
    pair_hypotheses: list[dict] = []
    roadside_items = sorted([item for item in items if item['source_family'] == 'roadside_boundary'], key=lambda item: (item['weight'], item['span_px']), reverse=True)[:8]
    height, width = shape
    road_values = np.asarray(road_mask, dtype=bool) if road_mask is not None and np.shape(road_mask) == shape else None
    for first_index in range(len(roadside_items)):
        for second_index in range(first_index + 1, len(roadside_items)):
            first = roadside_items[first_index]
            second = roadside_items[second_index]
            boundary_delta = angular_distance_degrees(first['angle_degrees'], second['angle_degrees'])
            if boundary_delta < 16.0 or boundary_delta > 72.0:
                continue
            point_a = np.asarray(first['center_xy'], dtype=np.float64)
            point_b = np.asarray(second['center_xy'], dtype=np.float64)
            direction_a = np.asarray(first['axis_vector_xy'], dtype=np.float64)
            direction_b = np.asarray(second['axis_vector_xy'], dtype=np.float64)
            matrix = np.column_stack((direction_a, -direction_b))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 0.08:
                continue
            parameters = np.linalg.solve(matrix, point_b - point_a)
            vanishing_point = point_a + float(parameters[0]) * direction_a
            if vanishing_point[0] < -width or vanishing_point[0] > 2.0 * width or vanishing_point[1] < -height or (vanishing_point[1] > 2.0 * height):
                continue
            first_endpoints = [np.asarray(first['endpoint_a_xy'], dtype=np.float64), np.asarray(first['endpoint_b_xy'], dtype=np.float64)]
            second_endpoints = [np.asarray(second['endpoint_a_xy'], dtype=np.float64), np.asarray(second['endpoint_b_xy'], dtype=np.float64)]
            first_near = max(first_endpoints, key=lambda point: float(np.linalg.norm(point - vanishing_point)))
            second_near = max(second_endpoints, key=lambda point: float(np.linalg.norm(point - vanishing_point)))
            near_midpoint = (first_near + second_near) / 2.0
            center_vector = near_midpoint - vanishing_point
            center_span = float(np.linalg.norm(center_vector))
            if center_span < 0.1 * diagonal:
                continue
            center_angle = float(math.degrees(math.atan2(center_vector[1], center_vector[0])) % 180.0)
            inside_ratio = 0.5
            if road_values is not None:
                samples = np.linspace(0.12, 0.95, 72)
                points = vanishing_point[None, :] + samples[:, None] * center_vector[None, :]
                xs = np.rint(points[:, 0]).astype(np.int32)
                ys = np.rint(points[:, 1]).astype(np.int32)
                valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                if int(valid.sum()) < 12:
                    continue
                inside_ratio = float(road_values[ys[valid], xs[valid]].mean())
                if inside_ratio < 0.28:
                    continue
            pair_strength = float(min(first['weight'], second['weight']))
            confidence = float(np.clip(0.55 + 0.3 * inside_ratio + 0.25 * min(1.0, pair_strength), 0.0, 1.0))
            pair_hypotheses.append({'angle_degrees': center_angle, 'support_weight': float(first['weight'] + second['weight']), 'confidence': confidence, 'angle_consistency': float(np.clip(1.0 - abs(boundary_delta - 36.0) / 72.0, 0.0, 1.0)), 'source_families': ['roadside_pair_center'], 'source_counts': {'roadside_pair_center': 1}, 'component_count': 2, 'maximum_span_px': float(max(first['span_px'], second['span_px'])), 'longitudinal_union_span_px': float(center_span), 'longitudinal_union_coverage': float(np.clip(center_span / diagonal, 0.0, 1.0)), 'source_category_diversity': 2, 'v31_strong_consensus': bool(center_span / diagonal >= float(p.v31_min_union_coverage) and confidence >= 0.7), 'road_center_support_ratio': inside_ratio, 'boundary_angle_delta_deg': float(boundary_delta)})
    remaining = list(range(len(items)))
    hypotheses: list[dict] = []
    tolerance = float(p.auxiliary_initial_cluster_tolerance_deg)
    while remaining:
        seed = max(remaining, key=lambda index: items[index]['weight'])
        center = float(items[seed]['angle_degrees'])
        members: list[int] = []
        for _ in range(3):
            members = [index for index in remaining if angular_distance_degrees(items[index]['angle_degrees'], center) <= tolerance]
            if not members:
                members = [seed]
            center, consistency = weighted_axial_mean_degrees([items[index]['angle_degrees'] for index in members], [items[index]['weight'] for index in members])
        by_family: dict[str, list[float]] = {}
        for index in members:
            by_family.setdefault(items[index]['source_family'], []).append(float(items[index]['weight']))
        capped_family_weights = {family: float(sum(sorted(weights, reverse=True)[:3])) for family, weights in by_family.items()}
        support_weight = float(sum((min(1.5, weight) for weight in capped_family_weights.values())))
        source_diversity = len(capped_family_weights)
        direction = np.asarray([math.cos(math.radians(float(center))), math.sin(math.radians(float(center)))], dtype=np.float64)
        projected_intervals: list[tuple[float, float]] = []
        category_names: set[str] = set()
        for index in members:
            category_names.add(str(items[index].get('source_category', items[index]['source_family'])))
            points = np.asarray(items[index].get('track_points', []), dtype=np.float64)
            if points.ndim == 2 and points.shape[1] >= 2 and len(points):
                projected = points[:, :2] @ direction
                projected_intervals.append((float(np.min(projected)), float(np.max(projected))))
        projected_intervals.sort()
        merged_intervals: list[list[float]] = []
        for start, end in projected_intervals:
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
        union_span = float(sum((max(0.0, end - start) for start, end in merged_intervals)))
        union_coverage = float(np.clip(union_span / diagonal, 0.0, 1.0))
        category_diversity = len(category_names)
        confidence = float(np.clip(support_weight / 1.25 * (0.8 + 0.1 * min(2, source_diversity) + 0.1 * min(3, category_diversity)), 0.0, 1.0))
        hypotheses.append({'angle_degrees': float(center), 'support_weight': support_weight, 'confidence': confidence, 'angle_consistency': float(consistency), 'source_families': sorted(capped_family_weights), 'source_counts': {family: int(sum((items[index]['source_family'] == family for index in members))) for family in capped_family_weights}, 'source_categories': sorted(category_names), 'source_category_diversity': int(category_diversity), 'longitudinal_union_span_px': union_span, 'longitudinal_union_coverage': union_coverage, 'v31_strong_consensus': bool(len(members) >= 2 and category_diversity >= int(p.v31_min_source_diversity) and (union_coverage >= float(p.v31_min_union_coverage)) and (float(consistency) >= 0.84)), 'component_count': len(members), 'maximum_span_px': float(max((items[index]['span_px'] for index in members)))})
        member_set = set(members)
        remaining = [index for index in remaining if index not in member_set]
    if pair_hypotheses:
        for hypothesis in hypotheses:
            if hypothesis['source_families'] == ['roadside_boundary']:
                hypothesis['confidence'] = float(0.35 * float(hypothesis['confidence']))
        hypotheses.extend(pair_hypotheses)
    hypotheses.sort(key=lambda item: (item['confidence'], item['support_weight'], item['angle_consistency']), reverse=True)
    return hypotheses[:max(1, int(p.auxiliary_initial_max_hypotheses))]

# 【評分工具】計算某候選道路角度獲得多少獨立輔助方向支持。
def auxiliary_hypothesis_prior(angle_degrees: float, hypotheses: list[dict], tolerance_degrees: float) -> float:
    if not hypotheses:
        return 0.0
    tolerance = max(1.0, float(tolerance_degrees))
    return float(max((float(item['confidence']) * max(0.0, 1.0 - angular_distance_degrees(angle_degrees, float(item['angle_degrees'])) / tolerance) for item in hypotheses)))

# 【配對工具】為輸入點尋找最近的道路中心站點及其距離。
def _nearest_station_indices(points: np.ndarray, centers: np.ndarray, maximum_samples: int=256) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    if len(values) > maximum_samples:
        indices = np.linspace(0, len(values) - 1, maximum_samples).astype(np.int32)
        values = values[indices]
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return (nearest.astype(np.int32), distances[np.arange(len(values)), nearest])

# 【曲線工具】從輔助元件擷取面向道路的一側，建立平滑代表曲線。
def road_facing_auxiliary_curve(raw_points: np.ndarray, centers: np.ndarray, widths: np.ndarray, association_width_factor: float, smoothness: float, maximum_samples: int=4096) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_points, dtype=np.float64)
    if len(values) > maximum_samples:
        take = np.linspace(0, len(values) - 1, maximum_samples).astype(np.int32)
        values = values[take]
    if len(values) < 2 or len(centers) < 2:
        return (values.copy(), np.zeros(len(values), dtype=np.int32))
    nearest, distances = _nearest_station_indices(values, centers, maximum_samples=max(maximum_samples, len(values)))
    limits = widths[nearest] * float(association_width_factor)
    keep = distances <= limits
    values, nearest, distances = (values[keep], nearest[keep], distances[keep])
    if len(values) < 2:
        return (values, nearest)
    observed_indices: list[int] = []
    observed_points: list[np.ndarray] = []
    for station in np.unique(nearest):
        station_mask = nearest == station
        station_points = values[station_mask]
        station_distances = distances[station_mask]
        if not len(station_points):
            continue
        threshold = float(np.percentile(station_distances, 12.0)) + 1.5
        facing = station_points[station_distances <= threshold]
        observed_indices.append(int(station))
        observed_points.append(np.median(facing, axis=0))
    if len(observed_points) < 2:
        return (np.asarray(observed_points), np.asarray(observed_indices, dtype=np.int32))
    indices = np.asarray(observed_indices, dtype=np.int32)
    points = np.asarray(observed_points, dtype=np.float64)
    lo, hi = (int(indices.min()), int(indices.max()))
    full_indices = np.arange(lo, hi + 1, dtype=np.int32)
    weights = np.zeros(len(full_indices), dtype=np.float64)
    weights[indices - lo] = 1.0
    fitted = np.column_stack([regularized_curve_fit(np.interp(full_indices, indices, points[:, axis]), weights, float(smoothness), 3) for axis in range(2)])
    return (fitted, full_indices)

# 【階段函式】將獨立輔助候選與目前道路走廊關聯，輸出可用方向證據。
def associate_auxiliary_direction_evidence(auxiliary_masks: dict[str, np.ndarray] | None, corridor: dict, shape: tuple[int, int], p: Parameters) -> dict:
    masks = validate_auxiliary_masks(auxiliary_masks, shape)
    track_package = build_auxiliary_direction_tracks(auxiliary_masks, shape, p)
    nonempty = {name: int(mask.sum()) for name, mask in masks.items() if np.any(mask)}
    empty_result = {'status': 'no_masks' if not nonempty else 'insufficient_associated_evidence', 'mask_pixel_counts': nonempty, 'evidence': [], 'source_counts': {}, 'associated_source_count': 0, 'calibrated': False, 'terminal_extension_used': False, 'v15_track_summary': track_package['summary']}
    if not p.enable_auxiliary_direction_evidence or not nonempty or (not corridor.get('ok')):
        if not p.enable_auxiliary_direction_evidence:
            empty_result['status'] = 'disabled'
        return empty_result
    centers = np.asarray(corridor.get('centerline', []), dtype=np.float64)
    left = np.asarray(corridor.get('left', []), dtype=np.float64)
    right = np.asarray(corridor.get('right', []), dtype=np.float64)
    count = min(len(centers), len(left), len(right))
    if count < 2:
        return empty_result
    centers, left, right = (centers[:count], left[:count], right[:count])
    widths = np.linalg.norm(right - left, axis=1)
    median_width = float(max(p.min_road_width, np.median(widths)))
    base_angle = float(corridor.get('selected_direction_angle_deg', 90.0))
    theta = math.radians(base_angle)
    base_normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    family_weights = {'road_line': 1.0, 'roadside_boundary': 0.85, 'cone_sequence': 0.45}
    evidence: list[dict] = []

    def append_linear_component(family: str, points: np.ndarray, diagnostic_points: np.ndarray | None=None, source_category: str | None=None, fragment_count: int=1):
        geometry = axial_angle_from_points(points)
        if geometry is None:
            return
        if geometry['span_px'] < float(p.auxiliary_min_component_span_px):
            return
        if geometry['elongation'] < float(p.auxiliary_min_elongation):
            return
        angle_delta = angular_distance_degrees(geometry['angle_degrees'], base_angle)
        association_angle_limit = float(p.v18_local_auxiliary_max_global_delta_deg if p.enable_v18_auxiliary_curve_roles else p.auxiliary_max_road_angle_delta_deg)
        if angle_delta > association_angle_limit:
            return
        nearest, distances = _nearest_station_indices(points, centers)
        distance_limit = median_width * float(p.auxiliary_association_width_factor)
        associated = distances <= distance_limit
        associated_fraction = float(associated.mean()) if len(associated) else 0.0
        if associated_fraction < float(p.auxiliary_min_associated_fraction):
            return
        associated_indices = nearest[associated]
        if not len(associated_indices):
            return
        start = int(associated_indices.min())
        end = int(associated_indices.max())
        coverage = float((end - start + 1) / max(1, count))
        centroid = np.asarray(points, dtype=np.float64).mean(axis=0)
        centroid_index = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))
        median_offset = float((centroid - centers[centroid_index]) @ base_normal)
        side = 'left' if median_offset < -0.15 * median_width else 'right' if median_offset > 0.15 * median_width else 'interior_or_unknown'
        elongation_score = float(np.clip((geometry['elongation'] - p.auxiliary_min_elongation) / 6.0, 0.2, 1.0))
        angle_score = max(0.0, 1.0 - angle_delta / max(1.0, association_angle_limit))
        quality = float(family_weights[family] * associated_fraction * max(0.15, min(1.0, coverage * 2.0)) * elongation_score * max(0.25, angle_score))
        display_points = np.asarray(diagnostic_points if diagnostic_points is not None else points, dtype=np.float64)
        bbox_min = np.floor(display_points.min(axis=0)).astype(np.int32)
        bbox_max = np.ceil(display_points.max(axis=0)).astype(np.int32)
        station_curve_xy: list[list[float]] = []
        if source_category == 'guardrail':
            kept_points, kept_stations = road_facing_auxiliary_curve(display_points, centers, widths, float(p.auxiliary_association_width_factor), float(p.v18_guardrail_facing_smoothness))
            curve_keep = np.ones(len(kept_points), dtype=bool)
        else:
            curve_points = np.asarray(points, dtype=np.float64)
            if len(curve_points) > 1024:
                take = np.linspace(0, len(curve_points) - 1, 1024).astype(np.int32)
                curve_points = curve_points[take]
            curve_nearest, curve_distances = _nearest_station_indices(curve_points, centers, maximum_samples=max(1024, len(curve_points)))
            curve_keep = curve_distances <= distance_limit
            kept_points = curve_points[curve_keep]
            kept_stations = curve_nearest[curve_keep]
        if np.any(curve_keep):
            for station_index in np.unique(kept_stations):
                station_points = kept_points[kept_stations == station_index]
                station_point = np.median(station_points, axis=0)
                station_curve_xy.append([float(station_index / max(1, count - 1)), float(station_point[0]), float(station_point[1])])
        evidence.append({'source_family': family, 'source_category': source_category or family, 'fragment_count': int(fragment_count), 'merged_fragments': bool(fragment_count >= 2), 'angle_degrees': float(geometry['angle_degrees']), 'angle_delta_from_road_deg': float(angle_delta), 'station_start': start, 'station_end': end, 'station_start_fraction': float(start / max(1, count - 1)), 'station_end_fraction': float(end / max(1, count - 1)), 'coverage': coverage, 'associated_fraction': associated_fraction, 'span_px': float(geometry['span_px']), 'elongation': float(geometry['elongation']), 'side': side, 'quality_weight': quality, 'station_curve_xy': station_curve_xy, 'component_bbox_xyxy': [int(bbox_min[0]), int(bbox_min[1]), int(bbox_max[0]), int(bbox_max[1])]})
    for track in track_package['accepted']:
        append_linear_component(str(track['source_family']), np.asarray(track['points'], dtype=np.float64), diagnostic_points=np.asarray(track.get('raw_points', track['points']), dtype=np.float64), source_category=str(track['category']), fragment_count=int(track['component_count']))
    cone_components = connected_component_point_sets(masks['cone'], max(2, int(p.auxiliary_min_component_pixels // 2)))
    if len(cone_components) >= 3:
        centroids = np.asarray([item.mean(axis=0) for item in cone_components])
        nearest, distances = _nearest_station_indices(centroids, centers)
        keep = distances <= median_width * float(p.auxiliary_association_width_factor)
        if int(keep.sum()) >= 3:
            cone_points = centroids[keep]
            diagnostic_cone_points = np.concatenate([component for component, keep_component in zip(cone_components, keep) if keep_component], axis=0)
            append_linear_component('cone_sequence', cone_points, diagnostic_points=diagnostic_cone_points)
    source_counts: dict[str, int] = {}
    for item in evidence:
        family = item['source_family']
        source_counts[family] = source_counts.get(family, 0) + 1
    status = 'associated' if evidence else 'insufficient_associated_evidence'
    return {'status': status, 'mask_pixel_counts': nonempty, 'evidence': evidence, 'source_counts': source_counts, 'associated_source_count': len(source_counts), 'calibrated': False, 'terminal_extension_used': False, 'v15_track_summary': track_package['summary']}

# =============================================================================
# 08. 輔助證據與主要道路的關聯
# =============================================================================
# 檢查輔助曲線是否鄰近目前主要 Road、方向是否相容及覆蓋是否足夠，
# 排除隔壁道路或零碎無方向性的標記，再交給實體軌跡與角色判定。
# 【關聯函式】篩選鄰近主要道路且方向、覆蓋與位置合理的輔助元件。
def _v13_associated_auxiliary_components(auxiliary_masks: dict[str, np.ndarray] | None, corridor: dict, shape: tuple[int, int], p: Parameters, direction_only: bool=False) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    track_package = build_auxiliary_direction_tracks(auxiliary_masks, shape, p)
    selected_masks = {'road_line': np.zeros(shape, dtype=bool), 'roadside_boundary': np.zeros(shape, dtype=bool)}
    centers = np.asarray(corridor.get('centerline', []), dtype=np.float64)
    left = np.asarray(corridor.get('left', []), dtype=np.float64)
    right = np.asarray(corridor.get('right', []), dtype=np.float64)
    count = min(len(centers), len(left), len(right))
    if count < 2:
        return ([], selected_masks, track_package['summary'])
    centers, left, right = (centers[:count], left[:count], right[:count])
    widths = np.linalg.norm(right - left, axis=1)
    normals = (right - left) / np.maximum(widths[:, None], 1e-06)
    base_angle = float(corridor.get('selected_direction_angle_deg', 90.0))
    records: list[dict] = []
    for track in track_package['accepted']:
        family = str(track['source_family'])
        component = np.asarray(track['points'], dtype=np.float64)
        raw_component = np.asarray(track.get('raw_points', track['points']), dtype=np.float64)
        geometry = track.get('geometry')
        if geometry is None:
            continue
        if not direction_only and angular_distance_degrees(float(track['angle_degrees']), base_angle) > float(p.v18_local_auxiliary_max_global_delta_deg if p.enable_v18_auxiliary_curve_roles else p.v13_auxiliary_component_angle_delta_deg):
            continue
        if len(component) > 768:
            take = np.linspace(0, len(component) - 1, 768).astype(np.int32)
            points = component[take]
        else:
            points = component
        nearest, distances = _nearest_station_indices(points, centers, maximum_samples=max(768, len(points)))
        limits = widths[nearest] * float(p.v32_direction_association_width_factor if direction_only else p.auxiliary_association_width_factor)
        associated = distances <= limits
        associated_fraction = float(associated.mean()) if len(associated) else 0.0
        if associated_fraction < float(p.v32_direction_min_associated_fraction if direction_only else p.v13_auxiliary_component_min_associated_fraction):
            continue
        points = points[associated]
        nearest = nearest[associated]
        if not len(points):
            continue
        offsets = np.einsum('ij,ij->i', points - centers[nearest], normals[nearest])
        ys = np.rint(raw_component[:, 1]).astype(np.int32)
        xs = np.rint(raw_component[:, 0]).astype(np.int32)
        valid = (xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])
        selected_masks[family][ys[valid], xs[valid]] = True
        curve_points = longitudinal_modal_ridge_points(component, angle_degrees=float(track['angle_degrees']), longitudinal_bin_px=float(p.v17_roadline_curve_bin_px) if family == 'road_line' else float(p.v17_guardrail_longitudinal_bin_px), normal_mode_bin_px=float(p.v17_roadline_normal_mode_bin_px) if family == 'road_line' else float(p.v17_guardrail_normal_mode_bin_px), minimum_points=int(p.v17_roadline_min_curve_points) if family == 'road_line' else int(p.v17_guardrail_min_ridge_points))
        records.append({'source_family': family, 'source_category': str(track['category']), 'fragment_count': int(track['component_count']), 'merged_fragments': bool(track['merged_fragments']), 'points': points, 'raw_points': raw_component, 'curve_points': curve_points, 'station_indices': nearest, 'signed_offsets': offsets, 'station_start': int(nearest.min()), 'station_end': int(nearest.max()), 'angle_degrees': float(track['angle_degrees']), 'angle_consistency': float(track['angle_consistency']), 'span_px': float(track['span_px']), 'associated_fraction': associated_fraction})
    return (records, selected_masks, track_package['summary'])

# 【曲線工具】依道路站序整理單一輔助紀錄的曲線點。
def _v33_ordered_record_curve(record: dict, centers: np.ndarray) -> np.ndarray:
    curve = np.asarray(record.get('curve_points', []), dtype=np.float64)
    if curve.ndim == 2 and curve.shape[1] >= 2:
        curve = curve[:, :2]
        curve = curve[np.all(np.isfinite(curve), axis=1)]
    else:
        curve = np.empty((0, 2), dtype=np.float64)
    if len(curve) < 2:
        points = np.asarray(record.get('points', []), dtype=np.float64)
        stations = np.asarray(record.get('station_indices', []), dtype=np.int32)
        valid = points.ndim == 2 and points.shape[1] >= 2 and (len(points) == len(stations))
        if valid:
            keep = (stations >= 0) & (stations < len(centers)) & np.all(np.isfinite(points[:, :2]), axis=1)
            points, stations = (points[keep, :2], stations[keep])
            unique = np.unique(stations)
            curve = np.asarray([np.median(points[stations == station], axis=0) for station in unique], dtype=np.float64)
    if len(curve) < 2:
        return curve
    nearest, _ = _nearest_station_indices(curve, centers, maximum_samples=max(768, len(curve)))
    if int(nearest[-1]) < int(nearest[0]):
        curve = curve[::-1]
    return curve

# 【曲線工具】使用舊版穩定方法重新擬合單一平滑輔助走向線。
def _v34_legacy_smooth_track_curve(record: dict, centers: np.ndarray, p: Parameters) -> np.ndarray:
    values = np.asarray(record.get('raw_points', []), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        values = np.asarray(record.get('points', []), dtype=np.float64)
    curve = np.empty((0, 2), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] >= 2 and (len(values) >= 2):
        curve = np.asarray(auxiliary_component_direction_curve(values[:, :2], p), dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] < 2 or len(curve) < 2:
        curve = _v33_ordered_record_curve(record, centers)
    else:
        curve = curve[:, :2]
        nearest, _ = _nearest_station_indices(curve, centers, maximum_samples=max(768, len(curve)))
        if len(nearest) and int(nearest[-1]) < int(nearest[0]):
            curve = curve[::-1]
    return curve

# 【幾何工具】將近似直線的輔助曲線轉成點、方向與殘差模型。
def _v33_curve_line_model(curve: np.ndarray) -> dict | None:
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] < 2 or len(curve) < 2:
        return None
    center = np.median(curve[:, :2], axis=0)
    centered = curve[:, :2] - center
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= max(float(np.linalg.norm(axis)), 1e-06)
    projections = centered @ axis
    endpoint_a = center + axis * float(projections.min())
    endpoint_b = center + axis * float(projections.max())
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    residual = float(np.median(np.abs(centered @ normal)))
    return {'center': center, 'axis': axis, 'normal': normal, 'endpoint_a': endpoint_a, 'endpoint_b': endpoint_b, 'span': float(projections.max() - projections.min()), 'residual': residual, 'angle_deg': float(math.degrees(math.atan2(axis[1], axis[0])) % 180.0)}

# 【幾何工具】計算兩個直線模型的交點，供共同消失點分析。
def _v33_line_intersection(first: dict, second: dict) -> np.ndarray | None:
    a = np.asarray(first['axis'], dtype=np.float64)
    b = np.asarray(second['axis'], dtype=np.float64)
    matrix = np.column_stack((a, -b))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.0001:
        return None
    rhs = np.asarray(second['center'] - first['center'], dtype=np.float64)
    parameters = np.linalg.solve(matrix, rhs)
    point = np.asarray(first['center'], dtype=np.float64) + parameters[0] * a
    return point if np.all(np.isfinite(point)) else None

# 【軌跡工具】取得輔助紀錄在主要道路站序上的起訖範圍。
def _v33_record_station_span(record: dict) -> tuple[int, int]:
    stations = np.asarray(record.get('station_indices', []), dtype=np.int32)
    if len(stations):
        return (int(stations.min()), int(stations.max()))
    return (int(record.get('station_start', 0)), int(record.get('station_end', 0)))

# 【階段函式】合併屬於同一實體 Roadline 的斷裂片段並重建平滑曲線。
def _v33_merge_physical_roadline_records(records: list[dict], centers: np.ndarray, widths: np.ndarray, p: Parameters) -> list[dict]:
    copied = [dict(record) for record in records]
    roadline_indices = [index for index, record in enumerate(copied) if str(record.get('source_family', '')) == 'road_line']
    parent = {index: index for index in roadline_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = (find(first), find(second))
        if root_a != root_b:
            parent[root_b] = root_a
    descriptors: dict[int, dict] = {}
    for index in roadline_indices:
        curve = _v33_ordered_record_curve(copied[index], centers)
        model = _v33_curve_line_model(curve)
        if model is not None:
            descriptors[index] = {'curve': curve, 'model': model, 'span': _v33_record_station_span(copied[index])}
    median_width = max(1.0, float(np.median(widths))) if len(widths) else 1.0
    for offset, first_index in enumerate(roadline_indices):
        if first_index not in descriptors:
            continue
        first = descriptors[first_index]
        for second_index in roadline_indices[offset + 1:]:
            if second_index not in descriptors:
                continue
            second = descriptors[second_index]
            first_lo, first_hi = first['span']
            second_lo, second_hi = second['span']
            overlap = min(first_hi, second_hi) - max(first_lo, second_lo) + 1
            if overlap > int(p.v33_track_merge_max_overlap_bands):
                continue
            gap = max(0, max(first_lo, second_lo) - min(first_hi, second_hi) - 1)
            if gap > int(p.v33_track_merge_max_station_gap):
                continue
            angle_delta = angular_distance_degrees(float(first['model']['angle_deg']), float(second['model']['angle_deg']))
            if angle_delta > float(p.v37_track_merge_max_tangent_delta_deg if bool(p.enable_v37_paired_cross_sections) else p.v33_track_merge_max_angle_deg):
                continue
            if first_hi <= second_lo:
                earlier, later = (first, second)
                boundary_station = min(max(first_hi, 0), len(widths) - 1)
            else:
                earlier, later = (second, first)
                boundary_station = min(max(second_hi, 0), len(widths) - 1)
            earlier_curve = np.asarray(earlier['curve'], dtype=np.float64)
            later_curve = np.asarray(later['curve'], dtype=np.float64)
            earlier_end = earlier_curve[-1]
            later_start = later_curve[0]
            connection = later_start - earlier_end
            distance = float(np.linalg.norm(connection))
            local_width = max(median_width, float(widths[boundary_station]) if len(widths) else median_width)
            mean_axis = np.asarray(earlier['model']['axis'], dtype=np.float64)
            if float(mean_axis @ later['model']['axis']) < 0.0:
                later_axis = -np.asarray(later['model']['axis'], dtype=np.float64)
            else:
                later_axis = np.asarray(later['model']['axis'], dtype=np.float64)
            mean_axis = mean_axis + later_axis
            mean_axis /= max(float(np.linalg.norm(mean_axis)), 1e-06)
            normal = np.asarray([-mean_axis[1], mean_axis[0]], dtype=np.float64)
            normal_residual = abs(float(connection @ normal)) / local_width
            if normal_residual > float(p.v37_track_merge_max_lateral_width_fraction if bool(p.enable_v37_paired_cross_sections) else p.v33_track_merge_max_normal_width_fraction):
                continue
            if distance > 1e-06:
                connection_angle = float(math.degrees(math.atan2(connection[1], connection[0])) % 180.0)
                if angular_distance_degrees(connection_angle, float(math.degrees(math.atan2(mean_axis[1], mean_axis[0])) % 180.0)) > 32.0:
                    continue
            union(first_index, second_index)
    groups: dict[int, list[int]] = {}
    for index in roadline_indices:
        groups.setdefault(find(index), []).append(index)
    used = set(roadline_indices)
    output = [record for index, record in enumerate(copied) if index not in used]
    for members in groups.values():
        members = sorted(members, key=lambda index: _v33_record_station_span(copied[index]))
        base = dict(copied[members[0]])
        observed_segments = []
        observed_station_spans: list[list[int]] = []
        all_raw, all_points, all_stations, all_offsets = ([], [], [], [])
        for member in members:
            record = copied[member]
            curve = _v33_ordered_record_curve(record, centers)
            if len(curve) >= 2:
                observed_segments.append(curve.tolist())
                station_start, station_end = _v33_record_station_span(record)
                observed_station_spans.append([int(station_start), int(station_end)])
            raw = np.asarray(record.get('raw_points', []), dtype=np.float64)
            points = np.asarray(record.get('points', []), dtype=np.float64)
            stations = np.asarray(record.get('station_indices', []), dtype=np.int32)
            offsets = np.asarray(record.get('signed_offsets', []), dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] >= 2:
                all_raw.append(raw[:, :2])
            if points.ndim == 2 and points.shape[1] >= 2 and (len(points) == len(stations)):
                all_points.append(points[:, :2])
                all_stations.append(stations)
                all_offsets.append(offsets if len(offsets) == len(stations) else np.zeros(len(stations), dtype=np.float64))
        if all_raw:
            raw_points = np.vstack(all_raw)
            geometry = axial_angle_from_points(raw_points)
            angle = float(geometry['angle_degrees'] if geometry is not None else base.get('angle_degrees', 0.0))
            base['raw_points'] = raw_points
            base['curve_points'] = longitudinal_modal_ridge_points(raw_points, angle_degrees=angle, longitudinal_bin_px=float(p.v17_roadline_curve_bin_px), normal_mode_bin_px=float(p.v17_roadline_normal_mode_bin_px), minimum_points=int(p.v17_roadline_min_curve_points))
            base['angle_degrees'] = angle
        if all_points:
            base['points'] = np.vstack(all_points)
            base['station_indices'] = np.concatenate(all_stations)
            base['signed_offsets'] = np.concatenate(all_offsets)
            base['station_start'] = int(base['station_indices'].min())
            base['station_end'] = int(base['station_indices'].max())
        base['fragment_count'] = int(sum((max(1, int(copied[member].get('fragment_count', 1))) for member in members)))
        base['merged_fragments'] = bool(len(members) > 1 or base['fragment_count'] > 1)
        base['v33_observed_segments'] = observed_segments
        base['v33_observed_station_spans'] = observed_station_spans
        base['v33_physical_component_count'] = int(len(members))
        output.append(base)
    output.sort(key=lambda record: (0 if str(record.get('source_family', '')) == 'road_line' else 1, _v33_record_station_span(record)[0], str(record.get('source_category', ''))))
    counters: dict[str, int] = {}
    for record in output:
        category = str(record.get('source_category', record.get('source_family', 'auxiliary')))
        display = {'road_line': 'Roadline', 'guardrail': 'Guardrail', 'curb': 'Curb', 'barricade': 'Barricade', 'roadside_wall': 'Roadside Wall', 'cone': 'Cone Sequence'}.get(category, category.replace('_', ' ').title())
        counters[display] = counters.get(display, 0) + 1
        record['v33_track_id'] = f'{display} Track {counters[display]}'
    return output

# =============================================================================
# 09. 中央 Roadline 的左右邊界配對與品質檢查
# =============================================================================
# 依中心站序單調配對左右道路邊界，限制分支跳轉、橫斷面交叉與寬度突變；
# 配對品質決定是否採用 Roadline 專用橫斷面或回退基礎道路模式。
# 【方向工具】由折線局部視窗計算每個站點的單位切線。
def _v37_unit_vectors(points: np.ndarray, window: int=2) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    tangents = np.zeros_like(points)
    if len(points) < 2:
        return tangents
    for index in range(len(points)):
        low = max(0, index - window)
        high = min(len(points) - 1, index + window)
        vector = points[high] - points[low]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-06 and index:
            vector = tangents[index - 1]
            norm = float(np.linalg.norm(vector))
        tangents[index] = vector / max(norm, 1e-06)
        if index and float(tangents[index] @ tangents[index - 1]) < 0.0:
            tangents[index] *= -1.0
    return tangents

# 【幾何工具】將曲線累積弧長正規化為 0 到 1 的道路進度。
def _v38_normalized_arclength(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    steps[~np.isfinite(steps)] = 0.0
    cumulative = np.concatenate(([0.0], np.cumsum(np.maximum(steps, 0.0))))
    total = float(cumulative[-1])
    return cumulative / total if total > 1e-06 else np.linspace(0.0, 1.0, len(values))

# 【邊界工具】依大距離跳躍將平滑邊界拆成不同連續分支。
def _v38_boundary_branch_ids(points: np.ndarray, reference_width: float, p: Parameters) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    ids = np.zeros(len(values), dtype=np.int32)
    if len(values) < 2:
        return ids
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    positive = steps[np.isfinite(steps) & (steps > 1e-06)]
    median_step = float(np.median(positive)) if len(positive) else 1.0
    jump_limit = max(4.0 * median_step, float(p.v38_branch_jump_width_fraction) * max(reference_width, 1.0))
    tangents = _v37_unit_vectors(values, window=2)
    angle_limit = math.radians(float(p.v38_branch_tangent_jump_deg))
    branch = 0
    for index in range(1, len(values)):
        jump = not np.isfinite(steps[index - 1]) or steps[index - 1] > jump_limit
        dot = float(np.clip(tangents[index - 1] @ tangents[index], -1.0, 1.0))
        turn = math.acos(dot) > angle_limit
        if jump or turn:
            branch += 1
        ids[index] = branch
    return ids

# 【幾何工具】計算點在線段上的投影位置、比例與距離。
def _v37_point_segment_projection(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float, np.ndarray]:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    squared = float(vector @ vector)
    if squared <= 1e-09:
        return (float('inf'), 0.0, np.asarray(first, dtype=np.float64))
    fraction = float((np.asarray(point, dtype=np.float64) - np.asarray(first, dtype=np.float64)) @ vector / squared)
    projection = np.asarray(first, dtype=np.float64) + np.clip(fraction, 0.0, 1.0) * vector
    return (float(np.linalg.norm(point - projection)), fraction, projection)

# 【幾何工具】判斷兩條橫斷線段是否在內部相交。
def _v37_segments_cross(first_left: np.ndarray, first_right: np.ndarray, second_left: np.ndarray, second_right: np.ndarray) -> bool:
    a = np.asarray(first_left, dtype=np.float64)
    b = np.asarray(first_right, dtype=np.float64)
    c = np.asarray(second_left, dtype=np.float64)
    d = np.asarray(second_right, dtype=np.float64)
    first = b - a
    second = d - c
    denominator = float(first[0] * second[1] - first[1] * second[0])
    if abs(denominator) <= 1e-09:
        return False
    offset = c - a
    u = float((offset[0] * second[1] - offset[1] * second[0]) / denominator)
    v = float((offset[0] * first[1] - offset[1] * first[0]) / denominator)
    return 0.02 < u < 0.98 and 0.02 < v < 0.98

# 【量測工具】計算一條線段有多少比例落在指定遮罩內。
def _v37_mask_fraction_on_segment(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    distance = float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
    count = max(3, int(math.ceil(distance)) + 1)
    xs = np.rint(np.linspace(first[0], second[0], count)).astype(np.int32)
    ys = np.rint(np.linspace(first[1], second[1], count)).astype(np.int32)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    return float(mask[ys[valid], xs[valid]].mean()) if np.any(valid) else 0.0

def _v37_pair_boundaries_monotone(stations: np.ndarray, tangents: np.ndarray, left_boundary: np.ndarray, right_boundary: np.ndarray, corridor_mask: np.ndarray, p: Parameters, *, anchor_indices: np.ndarray | None=None, road_mask: np.ndarray | None=None) -> dict:
    # 【主要功能：依道路站序聯合配對左右邊界】
    # 為每個中心站點選擇一組左／右端點，並以單調站序、方向平滑、寬度
    # 連續、分支身份及不交叉條件限制整段配對。輸出端點與配對品質指標。
    stations = np.asarray(stations, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    left_boundary = np.asarray(left_boundary, dtype=np.float64)
    right_boundary = np.asarray(right_boundary, dtype=np.float64)
    count = len(stations)
    output_left = np.full((count, 2), np.nan, dtype=np.float64)
    output_right = np.full((count, 2), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    sources = np.full(count, 'unpaired', dtype=object)
    left_indices = np.full(count, -1, dtype=np.int32)
    right_indices = np.full(count, -1, dtype=np.int32)
    relative_position = np.full(count, np.nan, dtype=np.float64)
    left_road_fraction = np.full(count, np.nan, dtype=np.float64)
    right_road_fraction = np.full(count, np.nan, dtype=np.float64)
    left_progress_values = np.full(count, np.nan, dtype=np.float64)
    right_progress_values = np.full(count, np.nan, dtype=np.float64)
    selected_left_branches = np.full(count, -1, dtype=np.int32)
    selected_right_branches = np.full(count, -1, dtype=np.int32)
    if count == 0 or len(left_boundary) < 2 or len(right_boundary) < 2 or (len(tangents) != count):
        return {'left': output_left, 'right': output_right, 'valid': valid, 'sources': sources, 'left_indices': left_indices, 'right_indices': right_indices, 'relative_position': relative_position, 'left_road_fraction': left_road_fraction, 'right_road_fraction': right_road_fraction, 'left_progress': left_progress_values, 'right_progress': right_progress_values, 'left_branch_ids': selected_left_branches, 'right_branch_ids': selected_right_branches, 'success_ratio': 0.0, 'crossing_rejections': 0, 'backtrack_rejections': 0}
    if anchor_indices is None or len(anchor_indices) != count:
        anchor_indices, _ = _nearest_station_indices(stations, 0.5 * (left_boundary + right_boundary), maximum_samples=max(768, count))
    anchor_indices = np.asarray(anchor_indices, dtype=np.int32)
    left_progress = _v38_normalized_arclength(left_boundary)
    right_progress = _v38_normalized_arclength(right_boundary)
    common_count = min(len(left_boundary), len(right_boundary))
    common_widths = np.linalg.norm(right_boundary[:common_count] - left_boundary[:common_count], axis=1) if common_count else np.empty(0, dtype=np.float64)
    finite_widths = common_widths[np.isfinite(common_widths) & (common_widths > 1.0)]
    reference_width = float(np.median(finite_widths)) if len(finite_widths) else 1.0
    left_branches = _v38_boundary_branch_ids(left_boundary, reference_width, p)
    right_branches = _v38_boundary_branch_ids(right_boundary, reference_width, p)
    search = max(1, int(p.v37_pair_search_radius))
    candidate_rows: list[list[dict]] = []
    for station_index, (center, tangent, anchor) in enumerate(zip(stations, tangents, anchor_indices)):
        tangent_norm = float(np.linalg.norm(tangent))
        tangent = tangent / max(tangent_norm, 1e-06)
        candidates: list[dict] = []
        left_range = range(max(0, int(anchor) - search), min(len(left_boundary), int(anchor) + search + 1))
        right_range = range(max(0, int(anchor) - search), min(len(right_boundary), int(anchor) + search + 1))
        for left_index in left_range:
            left_point = left_boundary[left_index]
            for right_index in right_range:
                right_point = right_boundary[right_index]
                transverse = right_point - left_point
                width = float(np.linalg.norm(transverse))
                if width < max(2.0, 0.65 * float(p.min_road_width)):
                    continue
                distance, fraction, _ = _v37_point_segment_projection(center, left_point, right_point)
                if not -0.04 <= fraction <= 1.04:
                    continue
                if distance > max(3.0, float(p.v37_pair_max_center_distance_fraction) * width):
                    continue
                transverse_unit = transverse / width
                if abs(float(transverse_unit @ tangent)) > float(p.v37_pair_max_tangent_alignment):
                    continue
                corridor_fraction = _v37_mask_fraction_on_segment(left_point, right_point, corridor_mask)
                if corridor_fraction < float(p.v37_pair_min_corridor_fraction):
                    continue
                progress_mismatch = abs(float(left_progress[left_index] - right_progress[right_index]))
                if bool(p.enable_v38_roadline_group_roles) and progress_mismatch > float(p.v38_pair_max_progress_mismatch):
                    continue
                left_distance = float(np.linalg.norm(center - left_point))
                right_distance = float(np.linalg.norm(right_point - center))
                base_cost = float(float(p.v37_pair_center_weight) * (distance / width) ** 2 + 0.35 * (1.0 - corridor_fraction) ** 2 + float(p.v37_pair_index_weight) * ((left_index - anchor) ** 2 + (right_index - anchor) ** 2) + float(p.v38_pair_progress_weight) * progress_mismatch ** 2)
                candidates.append({'left_index': int(left_index), 'right_index': int(right_index), 'left': left_point, 'right': right_point, 'width': width, 'angle': float(math.atan2(transverse[1], transverse[0])), 'fraction': float(np.clip(fraction, 0.0, 1.0)), 'left_distance': left_distance, 'right_distance': right_distance, 'left_progress': float(left_progress[left_index]), 'right_progress': float(right_progress[right_index]), 'left_branch': int(left_branches[left_index]), 'right_branch': int(right_branches[right_index]), 'base_cost': base_cost})
        candidate_rows.append(candidates)
    crossing_rejections = 0
    backtrack_rejections = 0
    last_selected_left = -1
    last_selected_right = -1
    accepted_segments: list[tuple[np.ndarray, np.ndarray]] = []
    row = 0
    while row < count:
        while row < count and (not candidate_rows[row]):
            row += 1
        if row >= count:
            break
        run_start = row
        while row < count and candidate_rows[row]:
            row += 1
        run_end = row
        costs = np.asarray([candidate['base_cost'] if candidate['left_index'] >= last_selected_left and candidate['right_index'] >= last_selected_right and (not any((_v37_segments_cross(previous_left, previous_right, candidate['left'], candidate['right']) for previous_left, previous_right in accepted_segments))) else np.inf for candidate in candidate_rows[run_start]], dtype=np.float64)
        if not np.any(np.isfinite(costs)):
            continue
        parents: list[np.ndarray] = [np.full(len(costs), -1, dtype=np.int32)]
        for current_row in range(run_start + 1, run_end):
            previous_candidates = candidate_rows[current_row - 1]
            current_candidates = candidate_rows[current_row]
            next_costs = np.full(len(current_candidates), np.inf, dtype=np.float64)
            next_parents = np.full(len(current_candidates), -1, dtype=np.int32)
            for current_index, current in enumerate(current_candidates):
                for previous_index, previous in enumerate(previous_candidates):
                    if not np.isfinite(costs[previous_index]):
                        continue
                    delta_left = current['left_index'] - previous['left_index']
                    delta_right = current['right_index'] - previous['right_index']
                    if delta_left < 0 or delta_right < 0:
                        backtrack_rejections += 1
                        continue
                    if bool(p.enable_v38_roadline_group_roles) and (current['left_branch'] != previous['left_branch'] or current['right_branch'] != previous['right_branch']):
                        continue
                    if delta_left > int(p.v37_pair_max_index_step) or delta_right > int(p.v37_pair_max_index_step):
                        continue
                    if _v37_segments_cross(previous['left'], previous['right'], current['left'], current['right']):
                        crossing_rejections += 1
                        continue
                    angle_delta = abs(float(np.angle(np.exp(1j * (current['angle'] - previous['angle'])))))
                    width_delta = math.log(max(current['width'], 1e-06)) - math.log(max(previous['width'], 1e-06))
                    transition_cost = float(float(p.v37_pair_angle_weight) * angle_delta ** 2 + float(p.v37_pair_width_weight) * width_delta ** 2 + 0.04 * ((delta_left - 1) ** 2 + (delta_right - 1) ** 2))
                    value = costs[previous_index] + current['base_cost'] + transition_cost
                    if value < next_costs[current_index]:
                        next_costs[current_index] = value
                        next_parents[current_index] = previous_index
            if not np.any(np.isfinite(next_costs)):
                run_end = current_row
                row = current_row
                break
            costs = next_costs
            parents.append(next_parents)
        solved_length = len(parents)
        if solved_length:
            selected = int(np.argmin(costs))
            path = [selected]
            for parent_values in reversed(parents[1:]):
                selected = int(parent_values[selected])
                if selected < 0:
                    break
                path.append(selected)
            path.reverse()
            for offset, candidate_index in enumerate(path):
                station_index = run_start + offset
                candidate = candidate_rows[station_index][candidate_index]
                if candidate['left_index'] < last_selected_left or candidate['right_index'] < last_selected_right:
                    backtrack_rejections += 1
                    continue
                if any((_v37_segments_cross(previous_left, previous_right, candidate['left'], candidate['right']) for previous_left, previous_right in accepted_segments)):
                    crossing_rejections += 1
                    continue
                output_left[station_index] = candidate['left']
                output_right[station_index] = candidate['right']
                valid[station_index] = True
                sources[station_index] = 'paired_boundaries'
                left_indices[station_index] = candidate['left_index']
                right_indices[station_index] = candidate['right_index']
                relative_position[station_index] = candidate['fraction']
                left_road_fraction[station_index] = _v37_mask_fraction_on_segment(candidate['left'], stations[station_index], road_mask)
                right_road_fraction[station_index] = _v37_mask_fraction_on_segment(stations[station_index], candidate['right'], road_mask)
                left_progress_values[station_index] = candidate['left_progress']
                right_progress_values[station_index] = candidate['right_progress']
                selected_left_branches[station_index] = candidate['left_branch']
                selected_right_branches[station_index] = candidate['right_branch']
                last_selected_left = int(candidate['left_index'])
                last_selected_right = int(candidate['right_index'])
                accepted_segments.append((np.asarray(candidate['left'], dtype=np.float64), np.asarray(candidate['right'], dtype=np.float64)))
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) >= 2:
        for first, second in zip(valid_indices, valid_indices[1:]):
            if second - first <= 1 or second - first > int(p.v38_pair_max_short_gap) or selected_left_branches[first] != selected_left_branches[second] or (selected_right_branches[first] != selected_right_branches[second]):
                continue
            for index in range(first + 1, second):
                fraction = float((index - first) / (second - first))
                output_left[index] = (1.0 - fraction) * output_left[first] + fraction * output_left[second]
                output_right[index] = (1.0 - fraction) * output_right[first] + fraction * output_right[second]
                valid[index] = True
                sources[index] = 'neighbor_interpolated'
                left_indices[index] = int(round((1.0 - fraction) * left_indices[first] + fraction * left_indices[second]))
                right_indices[index] = int(round((1.0 - fraction) * right_indices[first] + fraction * right_indices[second]))
                left_progress_values[index] = (1.0 - fraction) * left_progress_values[first] + fraction * left_progress_values[second]
                right_progress_values[index] = (1.0 - fraction) * right_progress_values[first] + fraction * right_progress_values[second]
                selected_left_branches[index] = selected_left_branches[first]
                selected_right_branches[index] = selected_right_branches[first]
                _, relative_position[index], _ = _v37_point_segment_projection(stations[index], output_left[index], output_right[index])
                left_road_fraction[index] = _v37_mask_fraction_on_segment(output_left[index], stations[index], road_mask)
                right_road_fraction[index] = _v37_mask_fraction_on_segment(stations[index], output_right[index], road_mask)
    return {'left': output_left, 'right': output_right, 'valid': valid, 'sources': sources, 'left_indices': left_indices, 'right_indices': right_indices, 'relative_position': relative_position, 'left_road_fraction': left_road_fraction, 'right_road_fraction': right_road_fraction, 'left_progress': left_progress_values, 'right_progress': right_progress_values, 'left_branch_ids': selected_left_branches, 'right_branch_ids': selected_right_branches, 'left_boundary_branch_count': int(left_branches.max() + 1) if len(left_branches) else 0, 'right_boundary_branch_count': int(right_branches.max() + 1) if len(right_branches) else 0, 'success_ratio': float(valid.mean()) if count else 0.0, 'crossing_rejections': int(crossing_rejections), 'backtrack_rejections': int(backtrack_rejections)}

# 【配對工具】只接受能補洞且不破壞原有效結果的邊界配對修復。
def _v38_merge_pairings_without_regression(initial: dict, repaired: dict) -> dict:
    initial_valid = np.asarray(initial.get('valid', []), dtype=bool)
    repaired_valid = np.asarray(repaired.get('valid', []), dtype=bool)
    if len(initial_valid) != len(repaired_valid):
        return initial
    output: dict = {}
    fill = ~initial_valid & repaired_valid
    array_keys = ('left', 'right', 'sources', 'left_indices', 'right_indices', 'relative_position', 'left_road_fraction', 'right_road_fraction', 'left_progress', 'right_progress', 'left_branch_ids', 'right_branch_ids')
    for key in array_keys:
        first = np.asarray(initial.get(key, []))
        second = np.asarray(repaired.get(key, []))
        if len(first) == len(initial_valid) and len(second) == len(initial_valid):
            merged = first.copy()
            merged[fill] = second[fill]
            output[key] = merged
    output['valid'] = initial_valid | fill
    output['success_ratio'] = float(np.mean(output['valid'])) if len(initial_valid) else 0.0
    output['crossing_rejections'] = int(initial.get('crossing_rejections', 0))
    output['backtrack_rejections'] = int(initial.get('backtrack_rejections', 0))
    output['left_boundary_branch_count'] = max(int(initial.get('left_boundary_branch_count', 0)), int(repaired.get('left_boundary_branch_count', 0)))
    output['right_boundary_branch_count'] = max(int(initial.get('right_boundary_branch_count', 0)), int(repaired.get('right_boundary_branch_count', 0)))
    output['quality_gate'] = {'policy': 'initial_pairs_frozen_repair_fills_missing_only', 'initial_valid': int(initial_valid.sum()), 'repair_candidates': int(repaired_valid.sum()), 'repair_added': int(fill.sum()), 'initial_preserved': int(initial_valid.sum())}
    return output

# 【品質工具】由成功率、單調性、交叉數及寬度平滑度評估專用配對。
def _v40_pairing_quality(pairing: dict, p: Parameters) -> dict:
    valid = np.asarray(pairing.get('valid', []), dtype=bool)
    left = np.asarray(pairing.get('left', []), dtype=np.float64)
    right = np.asarray(pairing.get('right', []), dtype=np.float64)
    left_indices = np.asarray(pairing.get('left_indices', []), dtype=np.int32)
    right_indices = np.asarray(pairing.get('right_indices', []), dtype=np.int32)
    relative = np.asarray(pairing.get('relative_position', []), dtype=np.float64)
    success = float(valid.mean()) if len(valid) else 0.0
    finite_inside = np.isfinite(relative) & valid if len(relative) == len(valid) else np.zeros(len(valid), dtype=bool)
    inside_ratio = float(np.mean((relative[finite_inside] >= -0.02) & (relative[finite_inside] <= 1.02))) if np.any(finite_inside) else 0.0
    chosen = np.flatnonzero(valid)
    monotonic_checks = [bool(left_indices[second] >= left_indices[first] and right_indices[second] >= right_indices[first]) for first, second in zip(chosen[:-1], chosen[1:]) if second < len(left_indices) and second < len(right_indices)]
    monotonic_ratio = float(np.mean(monotonic_checks)) if monotonic_checks else 0.0
    widths = np.linalg.norm(right[valid] - left[valid], axis=1) if len(left) == len(valid) and len(right) == len(valid) else np.asarray([], dtype=np.float64)
    if len(widths) >= 2:
        width_steps = np.abs(np.diff(widths)) / np.maximum(widths[:-1], 1.0)
        width_step_p90 = float(np.percentile(width_steps, 90))
    else:
        width_step_p90 = float('inf')
    accepted = bool(success >= float(p.v40_native_min_pair_success) and inside_ratio >= float(p.v40_native_min_inside_ratio) and (monotonic_ratio >= float(p.v40_native_min_monotonic_ratio)) and (width_step_p90 <= float(p.v40_native_max_width_step_ratio)))
    return {'accepted': accepted, 'success_ratio': success, 'inside_ratio': inside_ratio, 'monotonic_ratio': monotonic_ratio, 'width_step_p90': width_step_p90, 'crossing_count': int(pairing.get('crossing_rejections', 0))}

# =============================================================================
# 10. 實體 Roadline 合併及 Center／Edge／Direction 角色
# =============================================================================
# 合併屬於同一實體線的破碎片段，評估 Road 雙側支持、相對位置、覆蓋與
# 曲線品質；統一產生 C/E/D/X 角色，供導航中心與除錯圖共同使用。
# 【角色工具】計算實體輔助軌跡成為 center、edge 或方向證據的各項指標。
def _v33_track_role_metrics(record: dict, centers: np.ndarray, left: np.ndarray, right: np.ndarray, road_mask: np.ndarray, corridor_mask: np.ndarray, p: Parameters) -> dict:
    curve = _v33_ordered_record_curve(record, centers)
    result = {'role': 'D', 'center_score': 0.0, 'road_support': 0.0, 'bilateral_support': 0.0, 'road_centrality': 0.0, 'inside_corridor_fraction': 0.0, 'normalized_offset': float('inf'), 'coverage': 0.0, 'curve': curve}
    if len(curve) < 2:
        result['role'] = 'X'
        return result
    if bool(p.enable_v37_paired_cross_sections):
        if len(curve) > 180:
            take = np.linspace(0, len(curve) - 1, 180).astype(np.int32)
            curve = curve[take]
        anchors, _ = _nearest_station_indices(curve, centers, maximum_samples=max(768, len(curve)))
        tangents = _v37_unit_vectors(curve, window=2)
        pairing = _v37_pair_boundaries_monotone(curve, tangents, left, right, corridor_mask, p, anchor_indices=anchors, road_mask=road_mask)
        valid = np.asarray(pairing['valid'], dtype=bool)
        relative = np.asarray(pairing['relative_position'], dtype=np.float64)
        left_support = np.asarray(pairing['left_road_fraction'], dtype=np.float64)
        right_support = np.asarray(pairing['right_road_fraction'], dtype=np.float64)
        finite = valid & np.isfinite(relative)
        minimum_side_extent = np.minimum(relative, 1.0 - relative)
        side_road = np.minimum(left_support, right_support)
        bilateral = finite & (left_support >= float(p.v37_center_min_side_road_fraction)) & (right_support >= float(p.v37_center_min_side_road_fraction))
        pair_success = float(valid.mean()) if len(valid) else 0.0
        bilateral_fraction = float(bilateral.sum() / max(1, finite.sum()))
        side_extent_median = float(np.nanmedian(minimum_side_extent[finite])) if np.any(finite) else 0.0
        side_road_median = float(np.nanmedian(side_road[finite])) if np.any(finite) else 0.0
        relative_median = float(np.nanmedian(relative[finite])) if np.any(finite) else float('nan')
        relative_mad = float(np.nanmedian(np.abs(relative[finite] - relative_median))) if np.any(finite) else float('inf')
        inside_flags = []
        for point in curve:
            x, y = (int(round(point[0])), int(round(point[1])))
            inside_flags.append(bool(0 <= x < corridor_mask.shape[1] and 0 <= y < corridor_mask.shape[0] and corridor_mask[y, x]))
        inside = float(np.mean(inside_flags)) if inside_flags else 0.0
        observed = np.unique(np.asarray(record.get('station_indices', anchors), dtype=np.int32))
        observed = observed[(observed >= 0) & (observed < len(centers))]
        coverage = float(len(observed) / max(1, len(centers)))
        consistency = float(np.clip(record.get('angle_consistency', 1.0), 0.0, 1.0))
        score = float(0.3 * pair_success + 0.28 * bilateral_fraction + 0.18 * min(1.0, side_extent_median / 0.35) + 0.1 * min(1.0, side_road_median / 0.3) + 0.08 * inside + 0.06 * consistency)
        center = bool(pair_success >= float(p.v37_center_min_pair_success) and bilateral_fraction >= float(p.v37_center_min_bilateral_station_fraction) and (side_extent_median >= float(p.v37_center_min_side_extent_fraction)) and (side_road_median >= float(p.v37_center_min_side_road_fraction)) and (relative_mad <= float(p.v37_center_max_lateral_trend_mad)))
        likely_edge = bool(pair_success >= 0.3 and (side_extent_median <= float(p.v37_edge_max_side_extent_fraction) or bilateral_fraction < 0.2))
        role = 'C' if center else 'E' if likely_edge or inside >= 0.35 else 'D'
        result.update({'role': role, 'center_score': score, 'road_support': float(np.nanmean(np.maximum(left_support[finite], right_support[finite]))) if np.any(finite) else 0.0, 'bilateral_support': bilateral_fraction, 'road_centrality': side_extent_median, 'inside_corridor_fraction': inside, 'normalized_offset': relative_median - 0.5 if np.isfinite(relative_median) else float('inf'), 'lateral_rank_median': relative_median, 'lateral_rank_mad': relative_mad, 'boundary_hugging_fraction': float(np.mean(minimum_side_extent[finite] <= float(p.v37_edge_max_side_extent_fraction))) if np.any(finite) else 1.0, 'coverage': coverage, 'curve': curve, 'paired_boundary_success_ratio': pair_success, 'paired_bilateral_station_fraction': bilateral_fraction, 'paired_side_extent_median': side_extent_median, 'paired_side_road_fraction_median': side_road_median, 'paired_relative_position_median': relative_median, 'paired_relative_position_mad': relative_mad, 'paired_crossing_rejections': int(pairing['crossing_rejections']), 'paired_backtrack_rejections': int(pairing['backtrack_rejections']), 'v37_role_basis': 'monotone_boundary_pairing_and_bilateral_road'})
        legacy_metrics = _v33_track_role_metrics(record, centers, left, right, road_mask, corridor_mask, replace(p, enable_v37_paired_cross_sections=False))
        result.update({'legacy_role': str(legacy_metrics.get('role', 'D')), 'legacy_center_score': float(legacy_metrics.get('center_score', 0.0)), 'legacy_bilateral_support': float(legacy_metrics.get('bilateral_support', 0.0)), 'legacy_road_centrality': float(legacy_metrics.get('road_centrality', 0.0)), 'legacy_normalized_offset': float(legacy_metrics.get('normalized_offset', float('inf')))})
        return result
    if len(curve) > 160:
        take = np.linspace(0, len(curve) - 1, 160).astype(np.int32)
        curve = curve[take]
    nearest, _ = _nearest_station_indices(curve, centers, maximum_samples=max(256, len(curve)))
    widths = np.linalg.norm(right - left, axis=1)
    normals = (right - left) / np.maximum(widths[:, None], 1e-06)
    offsets = np.einsum('ij,ij->i', curve - centers[nearest], normals[nearest])
    normalized = offsets / np.maximum(widths[nearest], 1.0)
    lateral_rank = 0.5 + normalized
    finite_rank = lateral_rank[np.isfinite(lateral_rank)]
    lateral_rank_median = float(np.median(finite_rank)) if len(finite_rank) else float('inf')
    lateral_rank_mad = float(np.median(np.abs(finite_rank - lateral_rank_median))) if len(finite_rank) else float('inf')
    boundary_hugging_fraction = float(np.mean((finite_rank <= 0.1) | (finite_rank >= 0.9))) if len(finite_rank) else 1.0
    inside_flags, road_flags, bilateral_flags, centralities = ([], [], [], [])
    height, image_width = road_mask.shape
    for index, (point, station) in enumerate(zip(curve, nearest)):
        station = int(station)
        tangent_lo = max(0, index - 2)
        tangent_hi = min(len(curve) - 1, index + 2)
        tangent = curve[tangent_hi] - curve[tangent_lo]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-06:
            tangent = np.asarray([normals[station, 1], -normals[station, 0]])
        else:
            tangent = tangent / norm
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        sample_offsets = np.linspace(-0.48 * max(1.0, widths[station]), 0.48 * max(1.0, widths[station]), 41)
        samples = point[None, :] + sample_offsets[:, None] * normal[None, :]
        xs = np.rint(samples[:, 0]).astype(np.int32)
        ys = np.rint(samples[:, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < height)
        road_hits = np.zeros(len(samples), dtype=bool)
        corridor_hits = np.zeros(len(samples), dtype=bool)
        road_hits[valid] = road_mask[ys[valid], xs[valid]]
        corridor_hits[valid] = corridor_mask[ys[valid], xs[valid]]
        negative = sample_offsets[(sample_offsets < 0.0) & road_hits]
        positive = sample_offsets[(sample_offsets > 0.0) & road_hits]
        road_flags.append(bool(np.any(road_hits)))
        bilateral = bool(len(negative) and len(positive))
        bilateral_flags.append(bilateral)
        if bilateral:
            negative_extent = abs(float(negative.min()))
            positive_extent = float(positive.max())
            centralities.append(min(negative_extent, positive_extent) / max(negative_extent + positive_extent, 1e-06))
        x = int(round(point[0]))
        y = int(round(point[1]))
        inside_flags.append(bool(0 <= x < image_width and 0 <= y < height and corridor_mask[y, x]) or bool(np.any(corridor_hits[len(corridor_hits) // 2 - 1:len(corridor_hits) // 2 + 2])))
    observed_stations = np.unique(np.asarray(record.get('station_indices', nearest), dtype=np.int32))
    observed_stations = observed_stations[(observed_stations >= 0) & (observed_stations < len(centers))]
    coverage = float(len(observed_stations) / max(1, len(centers)))
    road_support = float(np.mean(road_flags)) if road_flags else 0.0
    bilateral_support = float(np.mean(bilateral_flags)) if bilateral_flags else 0.0
    centrality = float(np.median(centralities)) if centralities else 0.0
    inside = float(np.mean(inside_flags)) if inside_flags else 0.0
    normalized_offset = float(np.median(normalized)) if len(normalized) else float('inf')
    consistency = float(record.get('angle_consistency', 1.0))
    offset_term = math.exp(-abs(normalized_offset) / max(1e-06, float(p.v33_center_offset_soft_scale))) if np.isfinite(normalized_offset) else 0.0
    score = float(0.3 * bilateral_support + 0.18 * road_support + 0.18 * min(1.0, centrality / 0.45) + 0.12 * inside + 0.08 * min(1.0, coverage / 0.25) + 0.06 * np.clip(consistency, 0.0, 1.0) + 0.08 * offset_term)
    center = bool(coverage >= float(p.v33_center_min_coverage) and bilateral_support >= float(p.v33_center_min_bilateral_support) and (road_support >= float(p.v33_center_min_road_support)) and (score >= float(p.v33_center_min_score)))
    result.update({'role': 'C' if center else 'E' if inside >= 0.35 else 'D', 'center_score': score, 'road_support': road_support, 'bilateral_support': bilateral_support, 'road_centrality': centrality, 'inside_corridor_fraction': inside, 'normalized_offset': normalized_offset, 'lateral_rank_median': lateral_rank_median, 'lateral_rank_mad': lateral_rank_mad, 'boundary_hugging_fraction': boundary_hugging_fraction, 'coverage': coverage, 'curve': curve})
    return result

# 【透視工具】將指向相近消失區域的近直線輔助軌跡分成同一群。
def _v33_assign_vanishing_groups(records: list[dict], shape: tuple[int, int]) -> None:
    models: dict[int, dict] = {}
    for index, record in enumerate(records):
        curve = np.asarray(record.get('v33_curve', []), dtype=np.float64)
        model = _v33_curve_line_model(curve)
        if model is not None and model['span'] >= 20.0:
            models[index] = model
    diagonal = max(1.0, math.hypot(shape[1], shape[0]))
    candidates: list[np.ndarray] = []
    keys = list(models)
    for offset, first_index in enumerate(keys):
        for second_index in keys[offset + 1:]:
            delta = angular_distance_degrees(models[first_index]['angle_deg'], models[second_index]['angle_deg'])
            if delta < 2.0 or delta > 75.0:
                continue
            point = _v33_line_intersection(models[first_index], models[second_index])
            if point is not None and float(np.linalg.norm(point - np.asarray([0.5 * shape[1], 0.5 * shape[0]]))) <= 8.0 * diagonal:
                candidates.append(point)
    if not candidates:
        return
    best_point = None
    best_support: list[int] = []
    for point in candidates:
        support = []
        for index, model in models.items():
            vector = point - model['center']
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-06:
                continue
            angle = float(math.degrees(math.atan2(vector[1], vector[0])) % 180.0)
            if angular_distance_degrees(angle, model['angle_deg']) <= 8.0:
                support.append(index)
        if len(support) > len(best_support):
            best_point, best_support = (point, support)
    if best_point is None or len(best_support) < 2:
        return
    for index in best_support:
        records[index]['v33_vanishing_group'] = 1
        records[index]['v33_vanishing_point'] = best_point.tolist()

# 【角色工具】依多條 Roadline 的共同覆蓋與左右排列提出群組角色。
def _v38_assign_roadline_group_roles(roadlines: list[dict], p: Parameters) -> list[dict]:
    for record in roadlines:
        record['v38_roadline_group'] = None
        record['v38_group_order'] = None
        record['v38_group_role'] = None
        record['v38_group_resolved'] = False
    if not bool(p.enable_v38_roadline_group_roles) or len(roadlines) < 3:
        return []
    station_sets = [set(np.unique(np.asarray(record.get('station_indices', []), dtype=np.int32)).tolist()) for record in roadlines]
    adjacency = [set() for _ in roadlines]
    for first in range(len(roadlines)):
        for second in range(first + 1, len(roadlines)):
            common = station_sets[first] & station_sets[second]
            smaller = max(1, min(len(station_sets[first]), len(station_sets[second])))
            minimum = max(int(p.v38_group_min_common_stations), int(math.ceil(float(p.v38_group_min_common_fraction) * smaller)))
            same_vanishing = bool(roadlines[first].get('v33_vanishing_group') is not None and roadlines[first].get('v33_vanishing_group') == roadlines[second].get('v33_vanishing_group'))
            angle_ok = angular_distance_degrees(float(roadlines[first].get('angle_degrees', 0.0)), float(roadlines[second].get('angle_degrees', 0.0))) <= float(p.v38_group_max_angle_delta_deg)
            first_span = (min(station_sets[first]), max(station_sets[first])) if station_sets[first] else (0, -1)
            second_span = (min(station_sets[second]), max(station_sets[second])) if station_sets[second] else (0, -1)
            span_overlap = max(0, min(first_span[1], second_span[1]) - max(first_span[0], second_span[0]) + 1)
            enough_individual_support = bool(min(len(station_sets[first]), len(station_sets[second])) >= int(p.v38_group_min_common_stations))
            if (len(common) >= minimum or span_overlap >= minimum) and angle_ok or (same_vanishing and enough_individual_support):
                adjacency[first].add(second)
                adjacency[second].add(first)
    components: list[list[int]] = []
    unseen = set(range(len(roadlines)))
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        if len(component) >= 3:
            components.append(component)
    summaries: list[dict] = []
    for group_index, component in enumerate(components, start=1):
        ranked: list[tuple[float, dict]] = []
        for member in (roadlines[index] for index in component):
            metrics = member.get('v33_role_metrics', {})
            rank = float(metrics.get('paired_relative_position_median', float('nan')))
            mad = float(metrics.get('paired_relative_position_mad', float('inf')))
            if not np.isfinite(rank):
                legacy_offset = float(metrics.get('legacy_normalized_offset', float('nan')))
                if np.isfinite(legacy_offset):
                    rank = 0.5 + legacy_offset
                    mad = min(mad, 0.18)
            if np.isfinite(rank) and mad <= float(p.v38_group_max_rank_mad):
                ranked.append((rank, member))
        ranked.sort(key=lambda item: item[0])
        if len(ranked) < 3:
            continue
        gaps = np.diff(np.asarray([item[0] for item in ranked], dtype=np.float64))
        group_name = f'Roadline Group {group_index}'
        for order, (_, member) in enumerate(ranked):
            member['v38_roadline_group'] = group_name
            member['v38_group_order'] = int(order)
            member['v38_group_resolved'] = True
            member['v33_role'] = 'E'
            member['v33_center_group'] = None
            member['v33_role_reason'] = 'v38_group_outer_roadline_direction_only'
        center_members: list[dict]
        group_role = 'single_center'
        if len(ranked) == 3 and len(gaps) == 2:
            small_index = int(np.argmin(gaps))
            large_gap = float(np.max(gaps))
            small_gap = float(np.min(gaps))
            pair = [ranked[small_index][1], ranked[small_index + 1][1]]
            pair_internal = bool(max((float(member.get('v33_role_metrics', {}).get('paired_bilateral_station_fraction', 0.0)) for member in pair)) >= 0.45 and all((float(member.get('v33_role_metrics', {}).get('inside_corridor_fraction', 0.0)) >= 0.7 for member in pair)))
            if large_gap > 1e-06 and small_gap <= float(p.v38_double_line_gap_ratio) * large_gap and pair_internal:
                center_members = pair
                group_role = 'double_center'
            else:
                center_members = [ranked[1][1]]
        else:
            center_members = [ranked[len(ranked) // 2][1]]
        center_key = f'{group_name} Central Entity'
        for member in center_members:
            member['v33_role'] = 'C'
            member['v33_center_group'] = center_key
            member['v38_group_role'] = group_role
            member['v33_role_reason'] = 'v38_double_central_roadline_group' if group_role == 'double_center' else 'v38_middle_roadline_group_center'
        center_member_object_ids = {id(member) for member in center_members}
        for _, member in ranked:
            if id(member) not in center_member_object_ids:
                member['v38_group_role'] = 'edge'
        summaries.append({'group_id': group_name, 'members': [str(member.get('v33_track_id', 'Roadline')) for _, member in ranked], 'role': group_role, 'center_members': [str(member.get('v33_track_id', 'Roadline')) for member in center_members], 'relative_positions': [float(rank) for rank, _ in ranked]})
    return summaries

def _v40_finalize_roadline_roles(records: list[dict], p: Parameters) -> list[dict]:
    # 【主要功能：統一確認 Roadline 的幾何角色】
    # 彙整整條實體軌跡的排列、道路內位置、雙側 Road 支持、覆蓋長度與
    # 品質，產生唯一 C／E／D／X 角色，避免同一條線前後角色跳變。
    roadlines = [record for record in records if str(record.get('source_family', '')) == 'road_line' and len(record.get('v33_curve', [])) >= 2]
    for record in roadlines:
        record['v33_role'] = 'E'
        record['v33_center_group'] = None
        record['v33_adoption'] = 'candidate'
        record['v33_role_reason'] = 'v40_boundary_relative_direction_only'
        record['v38_roadline_group'] = None
        record['v38_group_order'] = None
        record['v38_group_role'] = 'edge'
        record['v38_group_resolved'] = False
    if not roadlines:
        return []
    station_sets = [set(np.unique(np.asarray(record.get('station_indices', []), dtype=np.int32)).tolist()) for record in roadlines]
    adjacency = [set() for _ in roadlines]
    for first in range(len(roadlines)):
        for second in range(first + 1, len(roadlines)):
            common = station_sets[first] & station_sets[second]
            shorter = max(1, min(len(station_sets[first]), len(station_sets[second])))
            required = max(int(p.v38_group_min_common_stations), int(math.ceil(float(p.v38_group_min_common_fraction) * shorter)))
            angle_ok = angular_distance_degrees(float(roadlines[first].get('angle_degrees', 0.0)), float(roadlines[second].get('angle_degrees', 0.0))) <= float(p.v38_group_max_angle_delta_deg)
            if len(common) >= required and angle_ok:
                adjacency[first].add(second)
                adjacency[second].add(first)
    components: list[list[int]] = []
    unseen = set(range(len(roadlines)))
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)

    def role_values(record: dict) -> dict:
        metrics = record.get('v33_role_metrics', {})
        rank = float(metrics.get('paired_relative_position_median', float('nan')))
        mad = float(metrics.get('paired_relative_position_mad', float('inf')))
        pair = float(metrics.get('paired_boundary_success_ratio', 0.0))
        bilateral = float(metrics.get('paired_bilateral_station_fraction', 0.0))
        coverage = float(metrics.get('coverage', 0.0))
        inside = float(metrics.get('inside_corridor_fraction', 0.0))
        side_road = float(metrics.get('paired_side_road_fraction_median', 0.0))
        if not np.isfinite(rank):
            legacy = float(metrics.get('legacy_normalized_offset', float('nan')))
            if np.isfinite(legacy):
                rank = 0.5 + legacy
        centrality = 1.0 - min(1.0, 2.0 * abs(rank - 0.5)) if np.isfinite(rank) else 0.0
        stability = 1.0 - min(1.0, mad / max(1e-06, p.v37_center_max_lateral_trend_mad)) if np.isfinite(mad) else 0.0
        score = float(0.34 * centrality + 0.2 * np.clip(pair, 0.0, 1.0) + 0.18 * np.clip(bilateral, 0.0, 1.0) + 0.1 * stability + 0.1 * min(1.0, coverage / max(1e-06, p.v40_center_min_coverage)) + 0.08 * np.clip(inside, 0.0, 1.0))
        eligible = bool(np.isfinite(rank) and 0.12 <= rank <= 0.88 and np.isfinite(mad) and (mad <= float(p.v37_center_max_lateral_trend_mad)) and (pair >= float(p.v37_center_min_pair_success)) and (bilateral >= float(p.v37_center_min_bilateral_station_fraction)) and (side_road >= float(p.v37_center_min_side_road_fraction)) and (coverage >= float(p.v40_center_min_coverage)) and (inside >= float(p.v37_pair_min_corridor_fraction)))
        return {'rank': rank, 'mad': mad, 'pair': pair, 'bilateral': bilateral, 'coverage': coverage, 'inside': inside, 'side_road': side_road, 'centrality': centrality, 'score': score, 'eligible': eligible}
    summaries: list[dict] = []
    candidates: list[tuple[float, dict, str]] = []
    grouped_ids: set[int] = set()
    group_index = 0
    for component in components:
        if len(component) < 3:
            continue
        common = set(station_sets[component[0]])
        for index in component[1:]:
            common &= station_sets[index]
        if len(common) < int(p.v38_group_min_common_stations):
            continue
        ranked: list[tuple[float, dict, dict]] = []
        for index in component:
            values = role_values(roadlines[index])
            if np.isfinite(values['rank']):
                ranked.append((float(values['rank']), roadlines[index], values))
        ranked.sort(key=lambda item: item[0])
        if len(ranked) < 3:
            continue
        group_index += 1
        group_name = f'Roadline Group {group_index}'
        for order, (_, record, _) in enumerate(ranked):
            grouped_ids.add(id(record))
            record['v38_roadline_group'] = group_name
            record['v38_group_order'] = int(order)
            record['v38_group_resolved'] = True
            record['v38_group_role'] = 'edge'
        middle_index, middle = min(enumerate(ranked), key=lambda indexed: (abs(indexed[0] / max(1, len(ranked) - 1) - 0.5), abs(float(indexed[1][0]) - 0.5)))
        _, middle_record, middle_values = middle
        middle_score = float(min(1.0, middle_values['score'] + 0.12))
        if middle_values['eligible']:
            candidates.append((middle_score, middle_record, 'concurrent_middle'))
        summaries.append({'group_id': group_name, 'members': [str(item[1].get('v33_track_id', 'Roadline')) for item in ranked], 'role': 'single_center_candidate', 'center_members': [str(middle_record.get('v33_track_id', 'Roadline'))], 'relative_positions': [float(item[0]) for item in ranked], 'actual_common_station_count': int(len(common)), 'double_line_special_used': False})
    for record in roadlines:
        if id(record) in grouped_ids:
            continue
        values = role_values(record)
        if values['eligible']:
            candidates.append((float(values['score']), record, 'boundary_relative'))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: dict | None = None
    if candidates:
        best_score, best_record, best_reason = candidates[0]
        margin = best_score - candidates[1][0] if len(candidates) > 1 else best_score
        if best_score >= float(p.v40_center_min_score) and margin >= float(p.v40_center_min_margin):
            selected = best_record
            selected['v33_role'] = 'C'
            selected['v33_role_reason'] = f'v40_{best_reason}_central_roadline'
            selected['v38_group_role'] = 'single_center'
            selected['v40_center_score'] = float(best_score)
            selected['v40_center_margin'] = float(margin)
    for record in roadlines:
        record['v40_final_role'] = str(record.get('v33_role', 'E'))
        record['v40_center_selected'] = bool(record is selected)
    return summaries

def _v33_build_unified_auxiliary_tracks(records: list[dict], centers: np.ndarray, left: np.ndarray, right: np.ndarray, road_mask: np.ndarray, corridor_mask: np.ndarray, p: Parameters, legacy_reference: dict | None=None) -> tuple[list[dict], list[dict]]:
    # 【主要功能：建立可供導航使用的實體輔助軌跡】
    # 將被遮擋或 segmentation 斷裂、但位置與走勢相容的片段合併成單一
    # 平滑曲線，再計算它與主要 Road 的關係。輸出全部軌跡及可採用軌跡。
    centers = np.asarray(centers, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    widths = np.linalg.norm(right - left, axis=1)
    merged = _v33_merge_physical_roadline_records(records, centers, widths, p)
    for record in merged:
        curve = _v34_legacy_smooth_track_curve(record, centers, p)
        if len(curve) >= 2:
            record['curve_points'] = curve.tolist()
        record['v33_curve'] = curve.tolist()
        observed_spans = record.get('v33_observed_station_spans')
        if not observed_spans:
            station_start, station_end = _v33_record_station_span(record)
            observed_spans = [[int(station_start), int(station_end)]]
        smooth_observed_segments: list[list[list[float]]] = []
        if len(curve) >= 2:
            curve_stations, _ = _nearest_station_indices(curve, centers, maximum_samples=max(768, len(curve)))
            for station_start, station_end in observed_spans:
                low, high = sorted((int(station_start), int(station_end)))
                keep = (curve_stations >= low) & (curve_stations <= high)
                indices = np.flatnonzero(keep)
                if len(indices) >= 2:
                    smooth_observed_segments.append(curve[indices[0]:indices[-1] + 1].tolist())
        record['v34_smooth_observed_segments'] = smooth_observed_segments
        record['v33_role'] = 'D'
        record['v33_adoption'] = 'candidate'
        record['v33_role_reason'] = 'direction_candidate'
        if str(record.get('source_family', '')) == 'road_line':
            metrics = _v33_track_role_metrics(record, centers, left, right, road_mask, corridor_mask, p)
            record['v33_role_metrics'] = {key: value for key, value in metrics.items() if key != 'curve'}
            record['v33_role'] = str(metrics['role'])
            record['v33_role_reason'] = 'multi_evidence_central_roadline' if metrics['role'] == 'C' else 'roadline_edge_direction_only' if metrics['role'] == 'E' else 'roadline_direction_only'
        elif str(record.get('source_family', '')) == 'roadside_boundary':
            record['v33_role'] = 'E'
            record['v33_role_reason'] = 'roadside_facility_direction_only'
    if bool(p.enable_v37_paired_cross_sections):
        roadline_records = [record for record in merged if str(record.get('source_family', '')) == 'road_line']
        strong_internal_tracks = [record for record in roadline_records if float(record.get('v33_role_metrics', {}).get('paired_boundary_success_ratio', 0.0)) >= 0.8 and float(record.get('v33_role_metrics', {}).get('paired_bilateral_station_fraction', 0.0)) >= 0.8 and (float(record.get('v33_role_metrics', {}).get('paired_side_extent_median', 0.0)) >= 0.16)]
        perspective_candidates = [record for record in strong_internal_tracks if str(record.get('v33_role', 'D')) != 'C' and str(record.get('v33_role_metrics', {}).get('legacy_role', 'D')) == 'C']
        if len(strong_internal_tracks) >= 2 and perspective_candidates:
            chosen = max(perspective_candidates, key=lambda record: float(record.get('v33_role_metrics', {}).get('legacy_center_score', 0.0)))
            chosen['v33_role'] = 'C'
            chosen['v33_role_reason'] = 'v37_multi_track_perspective_center_recovery'
        for record in roadline_records:
            if str(record.get('v33_role', 'D')) == 'C':
                continue
            metrics = record.get('v33_role_metrics', {})
            if str(metrics.get('legacy_role', 'D')) == 'C' and float(metrics.get('paired_boundary_success_ratio', 0.0)) < 0.15 and (float(metrics.get('coverage', 0.0)) >= 0.3):
                record['v33_role'] = 'C'
                record['v33_role_reason'] = 'v37_unavailable_pair_long_center_recovery'
    if legacy_reference is not None and str(legacy_reference.get('role', '')) == 'confirmed_centerline':
        legacy_curve = np.asarray(legacy_reference.get('curve_points', []), dtype=np.float64)
        if len(legacy_curve) >= 2:
            candidates = [record for record in merged if str(record.get('source_family', '')) == 'road_line' and len(record.get('v33_curve', [])) >= 2]
            if candidates:

                def legacy_distance(record: dict) -> float:
                    curve = np.asarray(record['v33_curve'], dtype=np.float64)
                    take = np.linspace(0, len(curve) - 1, min(80, len(curve))).astype(np.int32)
                    reference_take = np.linspace(0, len(legacy_curve) - 1, min(80, len(legacy_curve))).astype(np.int32)
                    first = curve[take]
                    second = legacy_curve[reference_take]
                    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
                    return float(np.median(np.min(distances, axis=1)))
                chosen = min(candidates, key=legacy_distance)
                local_width = max(1.0, float(np.median(widths)))
                if legacy_distance(chosen) <= 0.35 * local_width:
                    chosen['v33_legacy_center_supported'] = True
                    metrics = chosen.get('v33_role_metrics', {})
                    pair_success = float(metrics.get('paired_boundary_success_ratio', 0.0))
                    bilateral = float(metrics.get('paired_bilateral_station_fraction', 0.0))
                    side_extent = float(metrics.get('paired_side_extent_median', 0.0))
                    coverage = float(metrics.get('coverage', 0.0))
                    perspective_recovery = bool(pair_success >= 0.8 and bilateral >= 0.8 and (side_extent >= 0.16))
                    unavailable_pair_recovery = bool(pair_success < 0.15 and coverage >= 0.3)
                    if bool(p.enable_v37_paired_cross_sections) and str(chosen.get('v33_role', 'D')) != 'C' and (perspective_recovery or unavailable_pair_recovery):
                        chosen['v33_role'] = 'C'
                        chosen['v33_role_reason'] = 'v37_legacy_agreement_recovers_perspective_center' if perspective_recovery else 'v37_legacy_agreement_recovers_unavailable_pair_center'
                    if str(chosen.get('v33_role', 'D')) == 'C':
                        if not str(chosen.get('v33_role_reason', '')).startswith('v37_legacy_agreement_recovers_'):
                            chosen['v33_role_reason'] = 'v37_paired_support_with_legacy_agreement'
    _v33_assign_vanishing_groups(merged, road_mask.shape)
    roadlines = [record for record in merged if str(record.get('source_family', '')) == 'road_line' and len(record.get('v33_curve', [])) >= 2]
    v38_roadline_groups = _v38_assign_roadline_group_roles(roadlines, p)
    for record in roadlines:
        record['v38_roadline_group_summaries'] = v38_roadline_groups
    if bool(p.enable_v35_native_roadline_center):
        roadline_count = len(roadlines)
        for record in roadlines:
            if str(record.get('v33_role', 'D')) != 'C':
                continue
            metrics = record.get('v33_role_metrics', {})
            if bool(p.enable_v38_roadline_group_roles and record.get('v38_group_resolved', False)):
                record['v35_center_role_eligible'] = True
                continue
            if bool(p.enable_v37_paired_cross_sections):
                record['v35_center_role_eligible'] = True
                record['v33_role_reason'] = 'v37_paired_boundaries_confirmed_center'
                continue
            centrality = float(metrics.get('road_centrality', 0.0))
            coverage = float(metrics.get('coverage', 0.0))
            offset = abs(float(metrics.get('normalized_offset', float('inf'))))
            rank_mad = float(metrics.get('lateral_rank_mad', float('inf')))
            hugging = float(metrics.get('boundary_hugging_fraction', 1.0))
            stable_layout = bool(rank_mad <= float(p.v35_max_lateral_rank_mad) and hugging <= float(p.v35_max_boundary_hugging_fraction))
            if roadline_count <= 1:
                central_layout = bool(centrality >= float(p.v35_single_track_min_centrality) or offset <= float(p.v35_single_track_center_offset_fraction) or coverage >= float(p.v35_single_track_long_coverage))
            else:
                central_layout = bool(centrality >= float(p.v35_multi_track_min_centrality))
            record['v35_center_role_eligible'] = bool(stable_layout and central_layout)
            if not record['v35_center_role_eligible']:
                record['v33_role'] = 'E'
                record['v33_center_group'] = None
                record['v33_role_reason'] = 'v35_ambiguous_or_boundary_roadline_direction_only'
    center_group_counter = 0
    for first_index, first in enumerate(roadlines):
        first_stations = np.asarray(first.get('station_indices', []), dtype=np.int32)
        for second in roadlines[first_index + 1:]:
            if bool(p.enable_v38_roadline_group_roles and (first.get('v38_group_resolved', False) or second.get('v38_group_resolved', False))):
                continue
            second_stations = np.asarray(second.get('station_indices', []), dtype=np.int32)
            common = np.intersect1d(np.unique(first_stations), np.unique(second_stations))
            if len(common) < 5:
                continue
            if angular_distance_degrees(float(first.get('angle_degrees', 0.0)), float(second.get('angle_degrees', 0.0))) > 18.0 and (not (first.get('v33_vanishing_group') is not None and first.get('v33_vanishing_group') == second.get('v33_vanishing_group'))):
                continue
            separations = []
            first_points = np.asarray(first.get('points', []), dtype=np.float64)
            second_points = np.asarray(second.get('points', []), dtype=np.float64)
            for station in common:
                a = first_points[first_stations == station]
                b = second_points[second_stations == station]
                if len(a) and len(b):
                    separations.append(float(np.linalg.norm(np.median(a, axis=0) - np.median(b, axis=0))) / max(1.0, widths[int(station)]))
            if not separations or float(np.median(separations)) > float(p.v33_double_line_max_separation_width_fraction):
                continue
            first_score = float(first.get('v33_role_metrics', {}).get('center_score', 0.0))
            second_score = float(second.get('v33_role_metrics', {}).get('center_score', 0.0))
            if bool(p.enable_v37_paired_cross_sections):
                credible_member = bool(first['v33_role'] == 'C' or second['v33_role'] == 'C')
            else:
                credible_member = bool(max(first_score, second_score) >= 0.42 or first['v33_role'] == 'C' or second['v33_role'] == 'C')
            if not credible_member:
                continue
            center_group_counter += 1
            group_name = f'Central Roadline Group {center_group_counter}'
            for record in (first, second):
                record['v33_role'] = 'C'
                record['v33_center_group'] = group_name
                record['v33_role_reason'] = 'double_central_roadline_group'
    center_entities: dict[str, list[dict]] = {}
    for record in roadlines:
        if str(record.get('v33_role', 'D')) != 'C':
            continue
        key = str(record.get('v33_center_group', record.get('v33_track_id', 'Roadline')))
        center_entities.setdefault(key, []).append(record)
    if center_entities:

        def center_entity_score(members: list[dict]) -> float:
            scores = [float(member.get('v33_role_metrics', {}).get('center_score', 0.0)) for member in members]
            coverages = [float(member.get('v33_role_metrics', {}).get('coverage', 0.0)) for member in members]
            legacy_bonus = 0.25 if any((bool(member.get('v33_legacy_center_supported', False)) for member in members)) else 0.0
            pair_bonus = 0.08 if len(members) >= 2 else 0.0
            return max(scores, default=0.0) + 0.1 * sum(coverages) + legacy_bonus + pair_bonus
        ranked_center_entities = sorted(center_entities, key=lambda key: center_entity_score(center_entities[key]), reverse=True)
        selected_center_key = ranked_center_entities[0]
        if bool(p.enable_v35_native_roadline_center) and len(ranked_center_entities) > 1 and (center_entity_score(center_entities[selected_center_key]) - center_entity_score(center_entities[ranked_center_entities[1]]) < float(p.v35_center_uniqueness_margin)):
            selected_center_key = ''
        for key, members in center_entities.items():
            if key == selected_center_key:
                continue
            for record in members:
                record['v33_role'] = 'E'
                record['v33_center_group'] = None
                record['v33_role_reason'] = 'v35_ambiguous_center_direction_only' if not selected_center_key else 'non_selected_internal_roadline_direction_only'
    if bool(p.enable_v40_v38_roadline_center_sections):
        v40_groups = _v40_finalize_roadline_roles(merged, p)
        for record in merged:
            if str(record.get('source_family', '')) == 'road_line':
                record['v38_roadline_group_summaries'] = v40_groups
    serializable: list[dict] = []
    for record_index, record in enumerate(merged):
        record['v33_record_index'] = int(record_index)
        serializable.append({'record_index': int(record_index), 'track_id': str(record.get('v33_track_id', f'Auxiliary Track {record_index + 1}')), 'source_family': str(record.get('source_family', 'auxiliary')), 'source_category': str(record.get('source_category', 'auxiliary')), 'role': str(record.get('v33_role', 'D')), 'role_reason': str(record.get('v33_role_reason', 'direction_candidate')), 'adoption': str(record.get('v33_adoption', 'candidate')), 'fragment_count': int(record.get('fragment_count', 1)), 'physical_component_count': int(record.get('v33_physical_component_count', 1)), 'angle_degrees': float(record.get('angle_degrees', 0.0)), 'station_span': list(_v33_record_station_span(record)), 'station_count': int(len(np.unique(np.asarray(record.get('station_indices', []), dtype=np.int32)))), 'observed_segments': record.get('v34_smooth_observed_segments', [record.get('v33_curve', [])]), 'curve': record.get('v33_curve', []), 'curve_geometry': 'legacy_smooth_single_curve', 'center_group': record.get('v33_center_group'), 'roadline_group': record.get('v38_roadline_group'), 'roadline_group_order': record.get('v38_group_order'), 'roadline_group_role': record.get('v38_group_role'), 'roadline_group_resolved': bool(record.get('v38_group_resolved', False)), 'roadline_group_summaries': record.get('v38_roadline_group_summaries', []), 'legacy_center_supported': bool(record.get('v33_legacy_center_supported', False)), 'vanishing_group': record.get('v33_vanishing_group'), 'vanishing_point': record.get('v33_vanishing_point'), 'metrics': record.get('v33_role_metrics', {})})
    return (merged, serializable)

# =============================================================================
# 11. 中央 Roadline 原生站序、權威中心與過渡曲線
# =============================================================================
# 確認中央 Roadline 後，以其弧長建立站序並直接控制覆蓋區的中心位置與
# 切線；覆蓋區外才以平順過渡接回 V12 基礎中心。
# 【幾何工具】將中央 Roadline 等距取樣並計算平滑切線。
def _v35_uniform_curve_and_tangents(curve: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    sampled = resample_polyline(np.asarray(curve, dtype=np.float64), spacing)
    if len(sampled) < 2:
        return (sampled, np.empty((0, 2), dtype=np.float64))
    tangents = np.zeros_like(sampled)
    for index in range(len(sampled)):
        lo = max(0, index - 2)
        hi = min(len(sampled) - 1, index + 2)
        vector = sampled[hi] - sampled[lo]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-06:
            vector = sampled[min(len(sampled) - 1, index + 1)] - sampled[max(0, index - 1)]
            norm = float(np.linalg.norm(vector))
        tangents[index] = vector / max(norm, 1e-06)
    for index in range(1, len(tangents)):
        if float(tangents[index] @ tangents[index - 1]) < 0.0:
            tangents[index] *= -1.0
    return (sampled, tangents)

# 【中心工具】對單條或成對中央 Roadline 建立唯一中心參考曲線。
def _v35_midline_for_center_records(records: list[dict]) -> np.ndarray:
    curves = [np.asarray(record.get('v33_curve', []), dtype=np.float64) for record in records if len(record.get('v33_curve', [])) >= 2]
    if not curves:
        return np.empty((0, 2), dtype=np.float64)
    if len(curves) == 1:
        return curves[0]
    reference = curves[0]
    aligned = [reference]
    for curve in curves[1:]:
        direct = float(np.linalg.norm(reference[0] - curve[0]) + np.linalg.norm(reference[-1] - curve[-1]))
        reverse = float(np.linalg.norm(reference[0] - curve[-1]) + np.linalg.norm(reference[-1] - curve[0]))
        aligned.append(curve[::-1] if reverse < direct else curve)
    sample_count = max(2, max((len(curve) for curve in aligned)))
    normalized = np.linspace(0.0, 1.0, sample_count)
    resampled = []
    for curve in aligned:
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        if cumulative[-1] <= 1e-06:
            continue
        cumulative /= cumulative[-1]
        resampled.append(np.column_stack((np.interp(normalized, cumulative, curve[:, 0]), np.interp(normalized, cumulative, curve[:, 1]))))
    return np.mean(np.stack(resampled), axis=0) if resampled else np.empty((0, 2), dtype=np.float64)

# 【邊界工具】從中心與切線出發，在走廊遮罩中搜尋一組左右交點。
def _v35_mask_boundary_pair(center: np.ndarray, tangent: np.ndarray, corridor_mask: np.ndarray, minimum_width: float) -> tuple[np.ndarray, np.ndarray] | None:
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-06)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    height, width = corridor_mask.shape
    maximum = int(math.ceil(math.hypot(height, width)))
    offsets = np.arange(-maximum, maximum + 1, dtype=np.float64)
    xs = np.rint(float(center[0]) + offsets * normal[0]).astype(np.int32)
    ys = np.rint(float(center[1]) + offsets * normal[1]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, offsets = (xs[valid], ys[valid], offsets[valid])
    if not len(xs):
        return None
    keep = np.ones(len(xs), dtype=bool)
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    xs, ys, offsets = (xs[keep], ys[keep], offsets[keep])
    hits = np.asarray(corridor_mask[ys, xs], dtype=bool)
    hit_indices = np.flatnonzero(hits)
    if not len(hit_indices):
        return None
    center_index = int(np.argmin(np.abs(offsets)))
    anchor = int(hit_indices[np.argmin(np.abs(hit_indices - center_index))])
    if abs(anchor - center_index) > 3:
        return None
    lo = anchor
    while lo > 0 and hits[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(hits) and hits[hi + 1]:
        hi += 1
    if float(np.linalg.norm([xs[hi] - xs[lo], ys[hi] - ys[lo]])) < minimum_width:
        return None
    return (np.asarray([xs[lo], ys[lo]], dtype=np.float64), np.asarray([xs[hi], ys[hi]], dtype=np.float64))

# 【導航函式】以中央 Roadline 原生站序取代對應基礎中心區段。
def _v35_apply_native_central_roadline(base_centers: np.ndarray, base_left: np.ndarray, base_right: np.ndarray, base_sources: list[str], records: list[dict], corridor_mask: np.ndarray, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, dict]:
    centers = np.asarray(base_centers, dtype=np.float64)
    left = np.asarray(base_left, dtype=np.float64)
    right = np.asarray(base_right, dtype=np.float64)
    count = min(len(centers), len(left), len(right), len(base_sources))
    fallback_tangents = _v31_long_window_road_tangents(centers[:count], int(p.v32_road_fallback_window_bands))
    empty_authority = np.zeros(count, dtype=bool)
    summary = {'applied': False, 'reason': 'no_unique_confirmed_central_roadline', 'native_station_count': 0}
    if count < 3 or not bool(p.enable_v35_native_roadline_center):
        return (centers[:count], left[:count], right[:count], list(base_sources[:count]), fallback_tangents, empty_authority, summary)
    central = [record for record in records if str(record.get('source_family', '')) == 'road_line' and str(record.get('v33_role', 'D')) == 'C' and (str(record.get('v33_adoption', 'candidate')) == 'used')]
    if not central:
        return (centers[:count], left[:count], right[:count], list(base_sources[:count]), fallback_tangents, empty_authority, summary)
    entity_keys = {str(record.get('v33_center_group') or record.get('v33_track_id')) for record in central}
    if len(entity_keys) != 1:
        summary['reason'] = 'ambiguous_multiple_center_entities'
        return (centers[:count], left[:count], right[:count], list(base_sources[:count]), fallback_tangents, empty_authority, summary)
    center_curve = _v35_midline_for_center_records(central)
    if len(center_curve) < 2:
        summary['reason'] = 'central_curve_too_short'
        return (centers[:count], left[:count], right[:count], list(base_sources[:count]), fallback_tangents, empty_authority, summary)
    base_steps = np.linalg.norm(np.diff(centers[:count], axis=0), axis=1)
    positive = base_steps[base_steps > 1e-06]
    native_spacing = float(np.median(positive)) if len(positive) else 6.0
    native_spacing = max(2.0, native_spacing)
    native_centers, native_tangents = _v35_uniform_curve_and_tangents(center_curve, native_spacing)
    if len(native_centers) < 2:
        summary['reason'] = 'native_arc_length_sampling_failed'
        return (centers[:count], left[:count], right[:count], list(base_sources[:count]), fallback_tangents, empty_authority, summary)
    start_index = int(np.argmin(np.linalg.norm(centers[:count] - native_centers[0], axis=1)))
    end_index = int(np.argmin(np.linalg.norm(centers[:count] - native_centers[-1], axis=1)))
    if start_index > end_index:
        native_centers = native_centers[::-1].copy()
        native_tangents = -native_tangents[::-1].copy()
        start_index, end_index = (end_index, start_index)
    elif start_index == end_index:
        progress = centers[count - 1] - centers[0]
        if float((native_centers[-1] - native_centers[0]) @ progress) < 0.0:
            native_centers = native_centers[::-1].copy()
            native_tangents = -native_tangents[::-1].copy()
    required_span = min(count, max(2, len(native_centers)))
    if end_index - start_index + 1 < max(2, required_span // 2):
        nearest, _ = _nearest_station_indices(native_centers, centers[:count], maximum_samples=max(512, len(native_centers)))
        midpoint = int(np.median(nearest)) if len(nearest) else start_index
        start_index = max(0, midpoint - required_span // 2)
        end_index = min(count - 1, start_index + required_span - 1)
        start_index = max(0, end_index - required_span + 1)
    native_left: list[np.ndarray] = []
    native_right: list[np.ndarray] = []
    native_sources: list[str] = []
    for local_index, (point, tangent) in enumerate(zip(native_centers, native_tangents)):
        pair = _v35_mask_boundary_pair(point, tangent, corridor_mask, float(p.min_road_width))
        fraction = local_index / max(1, len(native_centers) - 1)
        mapped = int(round((1.0 - fraction) * start_index + fraction * end_index))
        mapped = int(np.clip(mapped, 0, count - 1))
        if pair is None:
            pair = (left[mapped].copy(), right[mapped].copy())
        first, second = pair
        direct_cost = float(np.linalg.norm(first - left[mapped]) + np.linalg.norm(second - right[mapped]))
        swapped_cost = float(np.linalg.norm(second - left[mapped]) + np.linalg.norm(first - right[mapped]))
        if swapped_cost < direct_cost:
            first, second = (second, first)
        native_left.append(first)
        native_right.append(second)
        native_sources.append(str(base_sources[mapped]))
    combined_centers = np.vstack((centers[:start_index], native_centers, centers[end_index + 1:count]))
    combined_left = np.vstack((left[:start_index], np.asarray(native_left), left[end_index + 1:count]))
    combined_right = np.vstack((right[:start_index], np.asarray(native_right), right[end_index + 1:count]))
    combined_sources = list(base_sources[:start_index]) + native_sources + list(base_sources[end_index + 1:count])
    combined_tangents = _v31_long_window_road_tangents(combined_centers, int(p.v32_road_fallback_window_bands))
    native_start = start_index
    native_end = native_start + len(native_centers)
    combined_tangents[native_start:native_end] = native_tangents
    authority = np.zeros(len(combined_centers), dtype=bool)
    authority[native_start:native_end] = True
    summary.update({'applied': True, 'reason': 'confirmed_center_uses_native_arc_length', 'center_entity': next(iter(entity_keys)), 'native_station_count': int(len(native_centers)), 'replaced_v12_station_count': int(end_index - start_index + 1), 'native_start_index': int(native_start), 'native_end_index_exclusive': int(native_end), 'native_curve_length_px': float(polyline_length(native_centers)), 'native_spacing_px': float(native_spacing)})
    return (combined_centers, combined_left, combined_right, combined_sources, combined_tangents, authority, summary)

# 【幾何工具】以位置與端點切線建立平滑 Hermite 過渡曲線。
def _v36_hermite_bridge(start: np.ndarray, start_tangent: np.ndarray, end: np.ndarray, end_tangent: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    chord = end - start
    distance = float(np.linalg.norm(chord))
    if distance <= 1e-06:
        return (np.vstack((start, end)), np.vstack((start_tangent, end_tangent)))
    chord_direction = chord / distance
    tangent0 = np.asarray(start_tangent, dtype=np.float64).copy()
    tangent1 = np.asarray(end_tangent, dtype=np.float64).copy()
    tangent0 /= max(float(np.linalg.norm(tangent0)), 1e-06)
    tangent1 /= max(float(np.linalg.norm(tangent1)), 1e-06)
    if float(tangent0 @ chord_direction) < 0.0:
        tangent0 *= -1.0
    if float(tangent1 @ chord_direction) < 0.0:
        tangent1 *= -1.0
    sample_count = max(2, int(math.ceil(distance / max(2.0, spacing))) + 1)
    values = np.linspace(0.0, 1.0, sample_count)
    u = values[:, None]
    h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
    h10 = u ** 3 - 2.0 * u ** 2 + u
    h01 = -2.0 * u ** 3 + 3.0 * u ** 2
    h11 = u ** 3 - u ** 2
    dh00 = 6.0 * u ** 2 - 6.0 * u
    dh10 = 3.0 * u ** 2 - 4.0 * u + 1.0
    dh01 = -6.0 * u ** 2 + 6.0 * u
    dh11 = 3.0 * u ** 2 - 2.0 * u

    def evaluate(handle_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        derivative0 = handle_fraction * distance * tangent0
        derivative1 = handle_fraction * distance * tangent1
        candidate_points = h00 * start + h10 * derivative0 + h01 * end + h11 * derivative1
        candidate_derivatives = dh00 * start + dh10 * derivative0 + dh01 * end + dh11 * derivative1
        return (candidate_points, candidate_derivatives)
    points, derivatives = evaluate(0.3)
    for handle_fraction in (0.24, 0.18, 0.12, 0.08):
        progress = (points - start) @ chord_direction
        lateral_vectors = points - start - progress[:, None] * chord_direction
        lateral = np.linalg.norm(lateral_vectors, axis=1)
        monotonic = bool(np.all(np.diff(progress) >= -0.02 * max(2.0, spacing)))
        local_bow = bool((float(np.max(lateral)) if len(lateral) else 0.0) <= 0.3 * distance)
        if monotonic and local_bow:
            break
        points, derivatives = evaluate(handle_fraction)
    norms = np.linalg.norm(derivatives, axis=1)
    tangents = derivatives / np.maximum(norms[:, None], 1e-06)
    for index in range(1, len(tangents)):
        if float(tangents[index] @ tangents[index - 1]) < 0.0:
            tangents[index] *= -1.0
    return (points, tangents)

# 【導航工具】依角度差尋找足夠遠且穩定的 V12 過渡錨點。
def _v36_transition_anchor(centers: np.ndarray, tangents: np.ndarray, boundary_index: int, native_tangent: np.ndarray, direction: int, p: Parameters) -> int:
    boundary_index = int(np.clip(boundary_index, 0, len(centers) - 1))
    base_tangent = np.asarray(tangents[boundary_index], dtype=np.float64).copy()
    native_tangent = np.asarray(native_tangent, dtype=np.float64).copy()
    base_tangent /= max(float(np.linalg.norm(base_tangent)), 1e-06)
    native_tangent /= max(float(np.linalg.norm(native_tangent)), 1e-06)
    angle = math.acos(float(np.clip(abs(base_tangent @ native_tangent), 0.0, 1.0)))
    fraction = angle / (0.5 * math.pi)
    step_count = int(round(float(p.v36_transition_min_stations) + fraction * (float(p.v36_transition_max_stations) - float(p.v36_transition_min_stations))))
    step_count = int(np.clip(step_count, int(p.v36_transition_min_stations), int(p.v36_transition_max_stations)))
    return int(np.clip(boundary_index + direction * step_count, 0, len(centers) - 1))

def _v36_apply_geometric_central_roadline(base_centers: np.ndarray, base_left: np.ndarray, base_right: np.ndarray, base_sources: list[str], records: list[dict], corridor_mask: np.ndarray, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str], dict]:
    # 【主要功能：讓確認的中央 Roadline 接管中心與前進方向】
    # Roadline 可靠覆蓋區使用其原生弧長站序、位置與切線；區段外以平順
    # 過渡接回基礎中心。道路寬度及左右外界仍由正式道路走廊限制。
    centers = np.asarray(base_centers, dtype=np.float64)
    left = np.asarray(base_left, dtype=np.float64)
    right = np.asarray(base_right, dtype=np.float64)
    count = min(len(centers), len(left), len(right), len(base_sources))
    centers, left, right = (centers[:count], left[:count], right[:count])
    sources = list(base_sources[:count])
    fallback_tangents = _v31_long_window_road_tangents(centers, int(p.v32_road_fallback_window_bands))
    summary = {'applied': False, 'reason': 'no_unique_confirmed_central_roadline', 'native_station_count': 0, 'transition_station_count': 0, 'geometric_authority': False}
    empty = np.zeros(count, dtype=bool)
    labels = ['v12'] * count
    if count < 3 or not bool(p.enable_v36_geometric_roadline_authority):
        return (centers, left, right, sources, fallback_tangents, empty, labels, summary)
    central = [record for record in records if str(record.get('source_family', '')) == 'road_line' and str(record.get('v33_role', 'D')) == 'C' and (str(record.get('v33_adoption', 'candidate')) == 'used')]
    entity_keys = {str(record.get('v33_center_group') or record.get('v33_track_id')) for record in central}
    if not central:
        summary['reason'] = 'no_confirmed_center_selected'
        return (centers, left, right, sources, fallback_tangents, empty, labels, summary)
    if len(entity_keys) != 1:
        summary['reason'] = 'ambiguous_multiple_center_entities'
        return (centers, left, right, sources, fallback_tangents, empty, labels, summary)
    center_curve = _v35_midline_for_center_records(central)
    if len(center_curve) < 2:
        summary['reason'] = 'central_curve_too_short'
        return (centers, left, right, sources, fallback_tangents, empty, labels, summary)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive = steps[steps > 1e-06]
    spacing = max(2.0, float(np.median(positive)) if len(positive) else 6.0)
    native_centers, native_tangents = _v35_uniform_curve_and_tangents(center_curve, spacing)
    if len(native_centers) < 2:
        summary['reason'] = 'native_arc_length_sampling_failed'
        return (centers, left, right, sources, fallback_tangents, empty, labels, summary)
    nearest_start = int(np.argmin(np.linalg.norm(centers - native_centers[0], axis=1)))
    nearest_end = int(np.argmin(np.linalg.norm(centers - native_centers[-1], axis=1)))
    if nearest_start > nearest_end:
        native_centers = native_centers[::-1].copy()
        native_tangents = -native_tangents[::-1].copy()
        nearest_start, nearest_end = (nearest_end, nearest_start)
    elif nearest_start == nearest_end:
        progress = centers[-1] - centers[0]
        if float((native_centers[-1] - native_centers[0]) @ progress) < 0.0:
            native_centers = native_centers[::-1].copy()
            native_tangents = -native_tangents[::-1].copy()
    trim_limit = max(1, len(native_centers) // 4)
    trimmed_front = 0
    trimmed_back = 0
    while len(native_centers) >= 4 and trimmed_front < trim_limit:
        chord = native_centers[-1] - native_centers[0]
        if float((native_centers[1] - native_centers[0]) @ chord) > 0.0:
            break
        native_centers = native_centers[1:]
        trimmed_front += 1
    while len(native_centers) >= 4 and trimmed_back < trim_limit:
        chord = native_centers[-1] - native_centers[0]
        if float((native_centers[-1] - native_centers[-2]) @ chord) > 0.0:
            break
        native_centers = native_centers[:-1]
        trimmed_back += 1
    native_centers, native_tangents = _v35_uniform_curve_and_tangents(native_centers, spacing)
    ordered_tangents = np.zeros_like(native_centers)
    for index in range(len(native_centers)):
        lo = max(0, index - 1)
        hi = min(len(native_centers) - 1, index + 1)
        local_forward = native_centers[hi] - native_centers[lo]
        norm = float(np.linalg.norm(local_forward))
        ordered_tangents[index] = local_forward / max(norm, 1e-06)
    native_tangents = ordered_tangents
    start_point, end_point = (native_centers[0], native_centers[-1])
    start_tangent, end_tangent = (native_tangents[0], native_tangents[-1])
    gate_tolerance = 0.5 * spacing
    start_signed = (centers - start_point) @ start_tangent
    end_signed = (centers - end_point) @ end_tangent
    inside_gate = (start_signed >= -gate_tolerance) & (end_signed <= gate_tolerance)
    gate_indices = np.flatnonzero(inside_gate)
    replace_start = nearest_start
    replace_end = nearest_end
    if len(gate_indices):
        replace_start = min(replace_start, int(gate_indices.min()))
        replace_end = max(replace_end, int(gate_indices.max()))
    required_span = min(count, max(2, len(native_centers)))
    if replace_end - replace_start + 1 < max(2, required_span // 2):
        nearest, _ = _nearest_station_indices(native_centers, centers, maximum_samples=max(512, len(native_centers)))
        midpoint = int(np.median(nearest)) if len(nearest) else replace_start
        expanded_start = max(0, midpoint - required_span // 2)
        expanded_end = min(count - 1, expanded_start + required_span - 1)
        expanded_start = max(0, expanded_end - required_span + 1)
        replace_start = min(replace_start, expanded_start)
        replace_end = max(replace_end, expanded_end)
    pre_anchor = _v36_transition_anchor(centers, fallback_tangents, max(0, replace_start - 1), start_tangent, -1, p) if replace_start > 0 else -1
    post_anchor = _v36_transition_anchor(centers, fallback_tangents, min(count - 1, replace_end + 1), end_tangent, 1, p) if replace_end + 1 < count else count
    if pre_anchor >= 0 and float(start_signed[pre_anchor]) >= -gate_tolerance:
        exterior_pre = np.flatnonzero(start_signed[:replace_start] < -gate_tolerance)
        pre_anchor = int(exterior_pre[np.argmin(np.abs(exterior_pre - pre_anchor))]) if len(exterior_pre) else -1
    if post_anchor < count and float(end_signed[post_anchor]) <= gate_tolerance:
        exterior_post = np.flatnonzero(end_signed[replace_end + 1:] > gate_tolerance) + replace_end + 1
        post_anchor = int(exterior_post[np.argmin(np.abs(exterior_post - post_anchor))]) if len(exterior_post) else count
    chunks: list[np.ndarray] = []
    tangent_chunks: list[np.ndarray] = []
    label_chunks: list[list[str]] = []
    source_chunks: list[list[str]] = []
    boundary_left_chunks: list[np.ndarray] = []
    boundary_right_chunks: list[np.ndarray] = []

    def add_existing(lo: int, hi: int):
        if hi <= lo:
            return
        chunks.append(centers[lo:hi])
        tangent_chunks.append(fallback_tangents[lo:hi])
        label_chunks.append(['v12'] * (hi - lo))
        source_chunks.append(sources[lo:hi])
        boundary_left_chunks.append(left[lo:hi])
        boundary_right_chunks.append(right[lo:hi])

    def add_generated(points: np.ndarray, tangents: np.ndarray, label: str, fallback_indices: np.ndarray):
        if not len(points):
            return
        generated_left: list[np.ndarray] = []
        generated_right: list[np.ndarray] = []
        generated_sources: list[str] = []
        for point, tangent, mapped in zip(points, tangents, fallback_indices):
            mapped = int(np.clip(mapped, 0, count - 1))
            pair = _v35_mask_boundary_pair(point, tangent, corridor_mask, float(p.min_road_width))
            if pair is None:
                pair = (left[mapped].copy(), right[mapped].copy())
            first, second = pair
            direct_cost = float(np.linalg.norm(first - left[mapped]) + np.linalg.norm(second - right[mapped]))
            swapped_cost = float(np.linalg.norm(second - left[mapped]) + np.linalg.norm(first - right[mapped]))
            if swapped_cost < direct_cost:
                first, second = (second, first)
            generated_left.append(first)
            generated_right.append(second)
            generated_sources.append(sources[mapped])
        chunks.append(points)
        tangent_chunks.append(tangents)
        label_chunks.append([label] * len(points))
        source_chunks.append(generated_sources)
        boundary_left_chunks.append(np.asarray(generated_left))
        boundary_right_chunks.append(np.asarray(generated_right))
    if pre_anchor >= 0:
        add_existing(0, pre_anchor + 1)
        bridge, bridge_tangents = _v36_hermite_bridge(centers[pre_anchor], fallback_tangents[pre_anchor], native_centers[0], native_tangents[0], spacing)
        if len(bridge) > 2:
            transition_points = bridge[1:-1]
            transition_tangents = bridge_tangents[1:-1]
            mapping = np.rint(np.linspace(pre_anchor, replace_start, len(bridge) - 2)).astype(np.int32)
            exterior = (transition_points - start_point) @ start_tangent < -1e-06
            add_generated(transition_points[exterior], transition_tangents[exterior], 'transition', mapping[exterior])
    native_mapping = np.rint(np.linspace(replace_start, replace_end, len(native_centers))).astype(np.int32)
    add_generated(native_centers, native_tangents, 'roadline_center', native_mapping)
    if post_anchor < count:
        bridge, bridge_tangents = _v36_hermite_bridge(native_centers[-1], native_tangents[-1], centers[post_anchor], fallback_tangents[post_anchor], spacing)
        if len(bridge) > 2:
            transition_points = bridge[1:-1]
            transition_tangents = bridge_tangents[1:-1]
            mapping = np.rint(np.linspace(replace_end, post_anchor, len(bridge) - 2)).astype(np.int32)
            exterior = (transition_points - end_point) @ end_tangent > 1e-06
            add_generated(transition_points[exterior], transition_tangents[exterior], 'transition', mapping[exterior])
        add_existing(post_anchor, count)
    combined_centers = np.vstack(chunks)
    combined_left = np.vstack(boundary_left_chunks)
    combined_right = np.vstack(boundary_right_chunks)
    combined_tangents = np.vstack(tangent_chunks)
    combined_labels = [value for chunk in label_chunks for value in chunk]
    combined_sources = [value for chunk in source_chunks for value in chunk]
    authority = np.asarray([value == 'roadline_center' for value in combined_labels], dtype=bool)
    transition_count = int(sum((value == 'transition' for value in combined_labels)))
    summary.update({'applied': True, 'reason': 'confirmed_center_uses_geometric_authority_and_g1_bridges', 'center_entity': next(iter(entity_keys)), 'geometric_authority': True, 'native_station_count': int(authority.sum()), 'transition_station_count': transition_count, 'replaced_v12_station_count': int(replace_end - replace_start + 1), 'geometric_gate_v12_station_count': int(inside_gate.sum()), 'authority_start_point': start_point.tolist(), 'authority_start_tangent': start_tangent.tolist(), 'authority_end_point': end_point.tolist(), 'authority_end_tangent': end_tangent.tolist(), 'native_curve_length_px': float(polyline_length(native_centers)), 'native_spacing_px': float(spacing), 'trimmed_front_native_stations': int(trimmed_front), 'trimmed_back_native_stations': int(trimmed_back), 'pre_transition_anchor_index': int(pre_anchor), 'post_transition_anchor_index': int(post_anchor)})
    return (combined_centers, combined_left, combined_right, combined_sources, combined_tangents, authority, combined_labels, summary)

# =============================================================================
# 12. 輔助物支持的道路包絡與歷史相容重建
# =============================================================================
# 建立 Road 與可信側邊設施之間的受限推測面，並保留早期版本診斷欄位；
# 新增區域不可反向成為 direct Road，也不可無限制改變主要道路身份。
# 【量測工具】沿左右端點線段產生整數像素取樣座標。
def _v13_section_samples(left: np.ndarray, right: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
    count = max(2, int(math.ceil(distance)) + 1)
    xs = np.rint(np.linspace(left[0], right[0], count)).astype(np.int32)
    ys = np.rint(np.linspace(left[1], right[1], count)).astype(np.int32)
    valid = (xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])
    xs, ys = (xs[valid], ys[valid])
    if len(xs) > 1:
        keep = np.ones(len(xs), dtype=bool)
        keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        xs, ys = (xs[keep], ys[keep])
    return (xs, ys)

# 【序列工具】依最大允許缺口將輔助站點拆成連續區段。
def _v19_split_station_runs(stations: list[int], maximum_gap: int) -> list[list[int]]:
    if not stations:
        return []
    ordered = sorted(set((int(value) for value in stations)))
    runs: list[list[int]] = [[ordered[0]]]
    for station in ordered[1:]:
        if station - runs[-1][-1] <= max(1, int(maximum_gap)):
            runs[-1].append(station)
        else:
            runs.append([station])
    return runs

# 【邊界工具】取得輔助 segmentation 面向道路及外側的實際邊緣。
def _v23_auxiliary_segmentation_edges(record: dict, centers: np.ndarray, left: np.ndarray, right: np.ndarray, p: Parameters) -> dict:
    count = min(len(centers), len(left), len(right))
    result = {'targets': {}, 'side': 0, 'normalized_offset': 0.0, 'center_like': False, 'side_like': False}
    if count < 2:
        return result
    raw = np.asarray(record.get('raw_points', record.get('points', [])), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2 or len(raw) < 2:
        return result
    raw = raw[:, :2]
    if len(raw) > 8192:
        take = np.linspace(0, len(raw) - 1, 8192).astype(np.int32)
        raw = raw[take]
    centers = np.asarray(centers[:count], dtype=np.float64)
    left = np.asarray(left[:count], dtype=np.float64)
    right = np.asarray(right[:count], dtype=np.float64)
    widths = np.linalg.norm(right - left, axis=1)
    normals = (right - left) / np.maximum(widths[:, None], 1e-06)
    nearest, distances = _nearest_station_indices(raw, centers, maximum_samples=max(8192, len(raw)))
    keep = distances <= widths[nearest] * float(p.auxiliary_association_width_factor)
    raw, nearest = (raw[keep], nearest[keep])
    if len(raw) < 2:
        return result
    offsets = np.einsum('ij,ij->i', raw - centers[nearest], normals[nearest])
    normalized = offsets / np.maximum(widths[nearest], 1e-06)
    median_normalized = float(np.median(normalized))
    side = 1 if median_normalized > 0.02 else -1 if median_normalized < -0.02 else 0
    quantile = float(np.clip(p.v23_edge_quantile, 0.01, 0.49))
    targets: dict[int, np.ndarray] = {}
    for station in np.unique(nearest):
        station = int(station)
        station_offsets = offsets[nearest == station]
        if not len(station_offsets):
            continue
        if side > 0:
            same_side = station_offsets[station_offsets >= 0.0]
            if not len(same_side):
                continue
            edge_offset = float(np.quantile(same_side, quantile))
        elif side < 0:
            same_side = station_offsets[station_offsets <= 0.0]
            if not len(same_side):
                continue
            edge_offset = float(np.quantile(same_side, 1.0 - quantile))
        else:
            edge_offset = float(station_offsets[np.argmin(np.abs(station_offsets))])
        targets[station] = centers[station] + normals[station] * edge_offset
    family = str(record.get('source_family', ''))
    result.update({'targets': targets, 'side': int(side), 'normalized_offset': median_normalized, 'center_like': bool(family == 'road_line' and abs(median_normalized) <= float(p.v23_center_roadline_max_offset_fraction)), 'side_like': bool(abs(median_normalized) >= float(p.v23_side_aux_min_offset_fraction))})
    return result

# 【遮罩工具】判斷線段是否穿過指定遮罩。
def _v23_segment_hits_mask(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> bool:
    length = float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
    samples = max(3, int(math.ceil(length)) + 1)
    xs = np.rint(np.linspace(first[0], second[0], samples)).astype(np.int32)
    ys = np.rint(np.linspace(first[1], second[1], samples)).astype(np.int32)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    return bool(np.any(mask[ys[valid], xs[valid]])) if np.any(valid) else False

# 【重建函式】建立 Road 與可信輔助物外緣間的受限推測道路面。
def _v19_auxiliary_road_envelope(road: np.ndarray, obstacles: np.ndarray, centers: np.ndarray, left: np.ndarray, right: np.ndarray, records: list[dict], p: Parameters, confirmed_centerline_available: bool=False) -> tuple[np.ndarray, list[list[list[float]]], dict]:
    empty = np.zeros_like(road, dtype=bool)
    empty_summary = {'enabled': bool(p.enable_v19_auxiliary_road_envelope), 'accepted_component_count': 0, 'accepted_run_count': 0, 'accepted_station_count': 0, 'added_pixels': 0, 'blank_pixels': 0, 'obstacle_pixels': 0, 'source_counts': {}, 'single_outline_pixels': 0, 'paired_outline_pixels': 0, 'accepted_pair_count': 0, 'rejected_pair_counts': {}}
    if not p.enable_v19_auxiliary_road_envelope or len(centers) < 2 or (not records):
        return (empty, [], empty_summary)
    height, width = road.shape
    single_canvas = Image.new('1', (width, height), 0)
    single_draw = ImageDraw.Draw(single_canvas)
    paired_canvas = Image.new('1', (width, height), 0)
    paired_draw = ImageDraw.Draw(paired_canvas)
    polygons: list[list[list[float]]] = []
    accepted_components = 0
    accepted_runs = 0
    accepted_stations: set[int] = set()
    source_counts: dict[str, int] = {}
    pair_candidate_records: list[tuple[dict, dict]] = []
    for record in records:
        family = str(record.get('source_family', ''))
        if family not in {'road_line', 'roadside_boundary'}:
            continue
        if float(record.get('associated_fraction', 0.0)) < float(p.v19_envelope_min_associated_fraction):
            continue
        if float(record.get('angle_consistency', 0.0)) < float(p.v19_envelope_min_angle_consistency):
            continue
        edge = _v23_auxiliary_segmentation_edges(record, centers, left, right, p)
        targets = edge['targets']
        if len(targets) < int(p.v19_envelope_min_station_observations):
            continue
        pair_candidate_records.append((record, edge))
        component_used = False
        for run in _v19_split_station_runs(list(targets), int(p.v19_envelope_max_station_gap_bands)):
            if len(run) < int(p.v19_envelope_min_station_observations) or run[-1] - run[0] + 1 < int(p.v19_envelope_min_span_bands):
                continue
            anchors: list[tuple[int, np.ndarray, np.ndarray]] = []
            for station in run:
                center = np.asarray(centers[station], dtype=np.float64)
                pair = np.asarray(right[station] - left[station], dtype=np.float64)
                road_width = float(np.linalg.norm(pair))
                if road_width < float(p.min_road_width):
                    continue
                normal = pair / road_width
                target = np.asarray(targets[station], dtype=np.float64)
                signed_offset = float((target - center) @ normal)
                if abs(signed_offset) > road_width * float(p.v19_envelope_max_offset_width_fraction):
                    continue
                sample_count = max(3, int(math.ceil(abs(signed_offset))) + 1)
                sample_points = center[None, :] + np.linspace(0.0, 1.0, sample_count)[:, None] * (target - center)[None, :]
                xs = np.rint(sample_points[:, 0]).astype(np.int32)
                ys = np.rint(sample_points[:, 1]).astype(np.int32)
                valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                if not np.any(valid):
                    continue
                sample_points, xs, ys = (sample_points[valid], xs[valid], ys[valid])
                road_hits = np.flatnonzero(road[ys, xs])
                if not len(road_hits):
                    continue
                road_anchor = sample_points[int(road_hits[-1])]
                if float(np.linalg.norm(target - road_anchor)) < 1.0:
                    continue
                anchors.append((station, road_anchor, target))
            if len(anchors) < int(p.v19_envelope_min_station_observations):
                continue
            accepted_runs += 1
            component_used = True
            source_counts[family] = source_counts.get(family, 0) + 1
            bridge_width = max(2, int(p.v19_envelope_bridge_width_px))
            for _, road_anchor, target in anchors:
                single_draw.line([tuple(road_anchor), tuple(target)], fill=1, width=bridge_width)
            for first, second in zip(anchors, anchors[1:]):
                first_station, first_road, first_target = first
                second_station, second_road, second_target = second
                if second_station - first_station > int(p.v19_envelope_max_station_gap_bands):
                    continue
                polygon = [first_road.tolist(), first_target.tolist(), second_target.tolist(), second_road.tolist()]
                single_draw.polygon([tuple(point) for point in polygon], fill=1)
                polygons.append(polygon)
            accepted_stations.update((item[0] for item in anchors))
        if component_used:
            accepted_components += 1
    rejected_pair_counts: dict[str, int] = {}
    accepted_pair_count = 0
    v12_support = corridor_polygon_mask(road.shape, left, right) | road

    def reject(reason: str):
        rejected_pair_counts[reason] = rejected_pair_counts.get(reason, 0) + 1
    if p.enable_v23_auxiliary_segmentation_envelope:
        for first_index in range(len(pair_candidate_records)):
            first_record, first_edge = pair_candidate_records[first_index]
            for second_index in range(first_index + 1, len(pair_candidate_records)):
                second_record, second_edge = pair_candidate_records[second_index]
                if angular_distance_degrees(float(first_record.get('angle_degrees', 0.0)), float(second_record.get('angle_degrees', 0.0))) > float(p.v23_pair_max_angle_delta_deg):
                    reject('direction_mismatch')
                    continue
                opposite_sides = bool(first_edge['side'] * second_edge['side'] < 0 and first_edge['side_like'] and second_edge['side_like'])
                center_and_side = bool(confirmed_centerline_available and (first_edge['center_like'] and second_edge['side_like'] or (second_edge['center_like'] and first_edge['side_like'])))
                if not (opposite_sides or center_and_side):
                    reject('same_side_or_unresolved_roles')
                    continue
                common = sorted(set(first_edge['targets']) & set(second_edge['targets']))
                pair_used = False
                for run in _v19_split_station_runs(common, int(p.v19_envelope_max_station_gap_bands)):
                    if len(run) < int(p.v23_pair_min_overlap_stations) or run[-1] - run[0] + 1 < int(p.v23_pair_min_overlap_span_bands):
                        continue
                    paired_targets = [(station, np.asarray(first_edge['targets'][station]), np.asarray(second_edge['targets'][station])) for station in run]
                    support = np.asarray([_v23_segment_hits_mask(a, b, v12_support) for _, a, b in paired_targets], dtype=bool)
                    if float(support.mean()) < float(p.v23_pair_min_v12_support_fraction):
                        reject('missing_v12_support')
                        continue
                    paired_targets = [item for item, supported in zip(paired_targets, support) if supported]
                    if len(paired_targets) < int(p.v23_pair_min_overlap_stations):
                        continue
                    for _, a, b in paired_targets:
                        paired_draw.line([tuple(a), tuple(b)], fill=1, width=max(2, int(p.v19_envelope_bridge_width_px)))
                    for first, second in zip(paired_targets, paired_targets[1:]):
                        if second[0] - first[0] > int(p.v19_envelope_max_station_gap_bands):
                            continue
                        polygon = [first[1].tolist(), first[2].tolist(), second[2].tolist(), second[1].tolist()]
                        paired_draw.polygon([tuple(point) for point in polygon], fill=1)
                        polygons.append(polygon)
                    accepted_stations.update((item[0] for item in paired_targets))
                    accepted_runs += 1
                    pair_used = True
                if pair_used:
                    accepted_pair_count += 1
    single_mask = np.asarray(single_canvas, dtype=bool) & ~road
    paired_mask = np.asarray(paired_canvas, dtype=bool) & ~road
    envelope = single_mask | paired_mask
    summary = {'enabled': True, 'accepted_component_count': int(accepted_components), 'accepted_run_count': int(accepted_runs), 'accepted_station_count': int(len(accepted_stations)), 'added_pixels': int(envelope.sum()), 'blank_pixels': int((envelope & ~obstacles).sum()), 'obstacle_pixels': int((envelope & obstacles).sum()), 'source_counts': source_counts, 'segmentation_outline_mode': 'road_facing_actual_mask_contour', 'single_outline_pixels': int(single_mask.sum()), 'paired_outline_pixels': int((paired_mask & ~single_mask).sum()), 'accepted_pair_count': int(accepted_pair_count), 'rejected_pair_counts': rejected_pair_counts}
    return (envelope, polygons, summary)

# 【重建函式】必要時只向輔助包絡所在側擴張道路邊界並局部平滑。
def _v19_expand_boundaries_to_envelope(road: np.ndarray, envelope: np.ndarray, centers: np.ndarray, left: np.ndarray, right: np.ndarray, sources: list[str], p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    centers = np.asarray(centers, dtype=np.float64).copy()
    left = np.asarray(left, dtype=np.float64).copy()
    right = np.asarray(right, dtype=np.float64).copy()
    sources = list(sources)
    base_centers = centers.copy()
    base_left = left.copy()
    base_right = right.copy()
    base_widths = np.linalg.norm(base_right - base_left, axis=1)
    base_normals = (base_right - base_left) / np.maximum(base_widths[:, None], 1e-06)
    left_extensions = np.zeros(len(centers), dtype=np.float64)
    right_extensions = np.zeros(len(centers), dtype=np.float64)
    summary = {'enabled': bool(p.v19_envelope_expand_geometry), 'modified_station_count': 0, 'left_expansion_count': 0, 'right_expansion_count': 0, 'maximum_width_growth_ratio': 1.0}
    if not p.v19_envelope_expand_geometry or not np.any(envelope):
        return (centers, left, right, sources, summary)
    height, width = road.shape
    support = road | envelope
    modified = 0
    left_count = 0
    right_count = 0
    max_growth = 1.0
    for index in range(min(len(centers), len(left), len(right), len(sources))):
        center = centers[index]
        pair = right[index] - left[index]
        old_width = float(np.linalg.norm(pair))
        if old_width < float(p.min_road_width):
            continue
        normal = pair / old_width
        extent = old_width * float(p.v19_envelope_max_offset_width_fraction)
        sample_count = max(5, int(math.ceil(2.0 * extent)) + 1)
        offsets = np.linspace(-extent, extent, sample_count)
        points = center[None, :] + offsets[:, None] * normal[None, :]
        xs = np.rint(points[:, 0]).astype(np.int32)
        ys = np.rint(points[:, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not np.any(valid):
            continue
        offsets, points, xs, ys = (offsets[valid], points[valid], xs[valid], ys[valid])
        envelope_hits = envelope[ys, xs]
        if not np.any(envelope_hits):
            continue
        support_hits = support[ys, xs]
        old_half = old_width / 2.0
        target_left = -old_half
        target_right = old_half
        if np.any(envelope_hits & (offsets < -old_half)):
            side_support = offsets[support_hits & (offsets < 0.0)]
            if len(side_support):
                target_left = min(target_left, float(side_support.min()))
        if np.any(envelope_hits & (offsets > old_half)):
            side_support = offsets[support_hits & (offsets > 0.0)]
            if len(side_support):
                target_right = max(target_right, float(side_support.max()))
        if target_left >= -old_half - 0.5 and target_right <= old_half + 0.5:
            continue
        max_width = old_width * float(p.v19_envelope_max_width_growth)
        requested_width = target_right - target_left
        if requested_width > max_width:
            overflow = requested_width - max_width
            left_extension = max(0.0, -old_half - target_left)
            right_extension = max(0.0, target_right - old_half)
            total_extension = max(left_extension + right_extension, 1e-06)
            target_left += overflow * left_extension / total_extension
            target_right -= overflow * right_extension / total_extension
        blend = float(np.clip(p.v19_envelope_boundary_blend, 0.0, 1.0))
        new_left_offset = (1.0 - blend) * -old_half + blend * target_left
        new_right_offset = (1.0 - blend) * old_half + blend * target_right
        left_extensions[index] = max(0.0, -new_left_offset - old_half)
        right_extensions[index] = max(0.0, new_right_offset - old_half)
        if target_left < -old_half - 0.5:
            left_count += 1
        if target_right > old_half + 0.5:
            right_count += 1
        modified += 1
        max_growth = max(max_growth, (old_width + left_extensions[index] + right_extensions[index]) / old_width)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= float(kernel.sum())
    smooth_left = np.convolve(left_extensions, kernel, mode='same')
    smooth_right = np.convolve(right_extensions, kernel, mode='same')
    for index in range(len(centers)):
        if smooth_left[index] <= 0.25 and smooth_right[index] <= 0.25:
            continue
        half = base_widths[index] / 2.0
        normal = base_normals[index]
        left[index] = base_centers[index] - normal * (half + smooth_left[index])
        right[index] = base_centers[index] + normal * (half + smooth_right[index])
        if sources[index] not in {'extrapolated', 'auxiliary_supported_extrapolated'}:
            sources[index] = 'auxiliary_boundary_inferred'
    summary.update({'modified_station_count': int(modified), 'left_expansion_count': int(left_count), 'right_expansion_count': int(right_count), 'maximum_width_growth_ratio': float(max_growth)})
    return (base_centers, left, right, sources, summary)

# 【歷史相容】以早期中央 Roadline 參考修正中心位置與局部方向。
def _v17_apply_central_roadline_reference(centers: np.ndarray, left: np.ndarray, right: np.ndarray, reference: dict | None, global_angle_degrees: float, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int], np.ndarray]:
    guided_indices: set[int] = set()
    guided_mask = np.zeros(len(centers), dtype=bool)
    if reference is None or not p.enable_v17_auxiliary_guided_geometry:
        return (centers, left, right, guided_indices, guided_mask)
    radians = math.radians(float(global_angle_degrees))
    axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    curve = np.asarray(reference['curve_points'], dtype=np.float64)
    curve_u = np.asarray(reference['u_values'], dtype=np.float64)
    if len(curve) < 2:
        return (centers, left, right, guided_indices, guided_mask)
    curve_stations = np.asarray(reference.get('station_positions', []), dtype=np.float64)
    if len(curve_stations) == len(curve) and len(curve_stations) >= 2:
        full_core = np.arange(max(0, int(math.ceil(float(curve_stations.min())))), min(len(centers) - 1, int(math.floor(float(curve_stations.max())))) + 1, dtype=np.int32)
        if len(full_core) < 2:
            return (centers, left, right, guided_indices, guided_mask)
        observed_stations = np.asarray(reference.get('observed_station_positions', []), dtype=np.float64)
        if len(observed_stations):
            observed_stations = observed_stations[np.isfinite(observed_stations)]
        if len(observed_stations) >= 2:
            support_radius = max(0, int(p.v18_curve_support_transition_bands))
            distances = np.min(np.abs(full_core[:, None] - observed_stations[None, :]), axis=1)
            core = full_core[distances <= float(support_radius)]
        else:
            core = full_core
        if len(core) < 2:
            return (centers, left, right, guided_indices, guided_mask)
        targets = np.column_stack((np.interp(core, curve_stations, curve[:, 0]), np.interp(core, curve_stations, curve[:, 1])))
    else:
        station_u = np.asarray(centers, dtype=np.float64) @ axis
        order = np.argsort(curve_u)
        curve_u = curve_u[order]
        curve = curve[order]
        core = np.flatnonzero((station_u >= float(curve_u.min())) & (station_u <= float(curve_u.max())))
        if len(core) < 2:
            return (centers, left, right, guided_indices, guided_mask)
        targets = np.column_stack((np.interp(station_u[core], curve_u, curve[:, 0]), np.interp(station_u[core], curve_u, curve[:, 1])))
    widths = np.linalg.norm(right - left, axis=1)
    role = str(reference.get('role', 'confirmed_centerline'))
    if role == 'shape_shifted':
        targets = centers[core].copy()
        blend = 0.0
        maximum_fraction = float(p.v18_shape_max_shift_width_fraction)
    else:
        blend = float(np.clip(p.v17_roadline_center_blend, 0.0, 1.0))
        maximum_fraction = float(p.v17_center_max_shift_width_fraction)
    for index, target in zip(core, targets):
        delta = np.asarray(target, dtype=np.float64) - centers[index]
        maximum_shift = max(float(p.min_road_width), widths[index]) * maximum_fraction
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > maximum_shift:
            delta *= maximum_shift / max(delta_norm, 1e-06)
        delta *= blend
        centers[index] += delta
        left[index] += delta
        right[index] += delta
        guided_indices.add(int(index))
        guided_mask[index] = True
    return (centers, left, right, guided_indices, guided_mask)

# 【歷史相容】從關聯輔助曲線中選出一條中心或方向參考。
def _v18_select_auxiliary_curve_reference(records: list[dict], road: np.ndarray, centers: np.ndarray, widths: np.ndarray, global_angle_degrees: float, p: Parameters, joint_auxiliary_score: float=0.0) -> dict | None:
    if not p.enable_v18_auxiliary_curve_roles or len(centers) < 2:
        return None
    height, image_width = road.shape
    radians = math.radians(float(global_angle_degrees))
    axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center_u = centers @ axis
    road_u_span = max(1.0, float(center_u.max() - center_u.min()))
    median_width = max(float(p.min_road_width), float(np.median(widths)))
    offsets_to_sample = np.linspace(-0.72 * median_width, 0.72 * median_width, 51)
    candidates: list[dict] = []
    for record_index, record in enumerate(records):
        raw_points = np.asarray(record.get('points', []), dtype=np.float64)
        raw_stations = np.asarray(record.get('station_indices', []), dtype=np.int32)
        valid_station_points = raw_points.ndim == 2 and raw_points.shape[1] >= 2 and (len(raw_points) == len(raw_stations))
        if not valid_station_points:
            continue
        valid = (raw_stations >= 0) & (raw_stations < len(centers)) & np.all(np.isfinite(raw_points[:, :2]), axis=1)
        raw_points = raw_points[valid, :2]
        raw_stations = raw_stations[valid]
        unique_stations = np.unique(raw_stations)
        if len(unique_stations) < 4:
            continue
        observed_curve = np.asarray([np.median(raw_points[raw_stations == station], axis=0) for station in unique_stations], dtype=np.float64)
        station_positions = np.arange(int(unique_stations[0]), int(unique_stations[-1]) + 1, dtype=np.int32)
        observation_weights = np.zeros(len(station_positions), dtype=np.float64)
        observation_weights[unique_stations - station_positions[0]] = 1.0
        curve = np.column_stack([regularized_curve_fit(np.interp(station_positions, unique_stations, observed_curve[:, coordinate]), observation_weights, max(3.0, float(p.v16_curve_smoothness)), int(p.v16_curve_robust_iterations)) for coordinate in range(2)])
        intrinsic = np.asarray(record.get('curve_points', []), dtype=np.float64)
        if intrinsic.ndim == 2 and intrinsic.shape[1] >= 2 and (len(intrinsic) >= 4):
            intrinsic = intrinsic[:, :2]
            finite = np.all(np.isfinite(intrinsic), axis=1)
            intrinsic = intrinsic[finite]
        if len(intrinsic) >= 4:
            lo_station = int(station_positions[0])
            hi_station = int(station_positions[-1])
            forward_cost = float(np.linalg.norm(intrinsic[0] - centers[lo_station]) + np.linalg.norm(intrinsic[-1] - centers[hi_station]))
            reverse_cost = float(np.linalg.norm(intrinsic[-1] - centers[lo_station]) + np.linalg.norm(intrinsic[0] - centers[hi_station]))
            if reverse_cost < forward_cost:
                intrinsic = intrinsic[::-1]
            intrinsic_steps = np.linalg.norm(np.diff(intrinsic, axis=0), axis=1)
            intrinsic_s = np.concatenate(([0.0], np.cumsum(intrinsic_steps)))
            if intrinsic_s[-1] > 1e-06:
                sample_s = np.linspace(0.0, float(intrinsic_s[-1]), len(station_positions))
                intrinsic_curve = np.column_stack((np.interp(sample_s, intrinsic_s, intrinsic[:, 0]), np.interp(sample_s, intrinsic_s, intrinsic[:, 1])))

                def curve_turn_p95(points: np.ndarray) -> float:
                    segments = np.diff(points, axis=0)
                    norms = np.linalg.norm(segments, axis=1)
                    valid_segments = norms > 1e-06
                    angles = np.degrees(np.arctan2(segments[valid_segments, 1], segments[valid_segments, 0])) % 180.0
                    turns = [angular_distance_degrees(first, second) for first, second in zip(angles[:-1], angles[1:])]
                    return float(np.percentile(turns, 95)) if turns else 0.0
                station_turn = curve_turn_p95(curve)
                intrinsic_turn = curve_turn_p95(intrinsic_curve)
                endpoint_limit = max(2.0 * median_width, 0.12 * math.hypot(image_width, height))
                if min(forward_cost, reverse_cost) <= 2.0 * endpoint_limit and intrinsic_turn + 1.0 < station_turn:
                    curve = intrinsic_curve
        u = curve @ axis
        span = float(polyline_length(curve))
        coverage = float((station_positions[-1] - station_positions[0] + 1) / max(1, len(centers)))
        segments = np.diff(curve, axis=0)
        segment_norms = np.linalg.norm(segments, axis=1)
        valid_segments = segment_norms > 1e-06
        turn_p95 = 0.0
        if int(valid_segments.sum()) >= 2:
            angles = np.degrees(np.arctan2(segments[valid_segments, 1], segments[valid_segments, 0])) % 180.0
            turns = [angular_distance_degrees(first, second) for first, second in zip(angles[:-1], angles[1:])]
            if turns:
                turn_p95 = float(np.percentile(turns, 95))
        station_indices = np.asarray(record.get('station_indices', []), dtype=np.int32)
        signed_offsets = np.asarray(record.get('signed_offsets', []), dtype=np.float64)
        valid_offsets = (station_indices >= 0) & (station_indices < len(widths)) & np.isfinite(signed_offsets)
        if np.any(valid_offsets):
            fractions = signed_offsets[valid_offsets] / np.maximum(widths[station_indices[valid_offsets]], 1.0)
            signed_fraction = float(np.median(fractions))
            offset_fraction = abs(signed_fraction)
            offset_mad = float(np.median(np.abs(fractions - signed_fraction)))
        else:
            signed_fraction = 0.0
            offset_fraction = float('inf')
            offset_mad = float('inf')
        any_road: list[bool] = []
        bilateral: list[bool] = []
        centralities: list[float] = []
        for index, point in enumerate(curve):
            lo, hi = (max(0, index - 2), min(len(curve) - 1, index + 2))
            tangent = curve[hi] - curve[lo]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-06)
            if float(tangent @ axis) < 0:
                tangent *= -1.0
            local_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            samples = point[None, :] + offsets_to_sample[:, None] * local_normal
            xs = np.rint(samples[:, 0]).astype(np.int32)
            ys = np.rint(samples[:, 1]).astype(np.int32)
            valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < height)
            hits = np.zeros(len(offsets_to_sample), dtype=bool)
            hits[valid] = road[ys[valid], xs[valid]]
            negative = offsets_to_sample[(offsets_to_sample < 0) & hits]
            positive = offsets_to_sample[(offsets_to_sample > 0) & hits]
            any_road.append(bool(np.any(hits)))
            has_both = bool(len(negative) and len(positive))
            bilateral.append(has_both)
            if has_both:
                negative_extent = abs(float(negative.min()))
                positive_extent = float(positive.max())
                centralities.append(min(negative_extent, positive_extent) / max(negative_extent + positive_extent, 1e-06))
        road_support = float(np.mean(any_road)) if any_road else 0.0
        bilateral_support = float(np.mean(bilateral)) if bilateral else 0.0
        centrality = float(np.median(centralities)) if centralities else 0.0
        associated_fraction = float(record.get('associated_fraction', 0.0))
        angle_consistency = float(record.get('angle_consistency', 0.0))
        angle_delta = angular_distance_degrees(float(record.get('angle_degrees', global_angle_degrees)), global_angle_degrees)
        reliable_shape = bool(coverage >= float(p.v18_shape_min_coverage) and associated_fraction >= float(p.v18_shape_min_associated_fraction) and (angle_consistency >= 0.86) and (angle_delta <= float(p.v18_local_auxiliary_max_global_delta_deg if p.enable_v18_auxiliary_curve_roles else p.v13_auxiliary_component_angle_delta_deg)) and (turn_p95 <= float(p.v18_shape_max_turn_p95_deg)) and (road_support >= 0.35))
        if p.enable_v30_auxiliary_navigation_fusion:
            reliable_shape = bool(reliable_shape or (coverage >= 0.06 and associated_fraction >= 0.35 and (angle_consistency >= 0.8) and (angle_delta <= float(p.v18_local_auxiliary_max_global_delta_deg)) and (turn_p95 <= max(30.0, float(p.v18_shape_max_turn_p95_deg))) and (road_support >= (0.15 if float(joint_auxiliary_score) >= float(p.v30_strong_joint_direction_score) else 0.25))))
        family = str(record.get('source_family', ''))
        family_weight = 1.0 if family == 'road_line' else 0.72
        shape_score = float(family_weight * coverage * associated_fraction * max(0.25, angle_consistency) * max(0.2, 1.0 - turn_p95 / max(1.0, p.v18_shape_max_turn_p95_deg)))
        candidates.append({'record_index': int(record_index), 'source_family': family, 'source_category': str(record.get('source_category', family)), 'curve_points': curve, 'u_values': u, 'station_positions': station_positions, 'observed_station_positions': unique_stations, 'span_px': span, 'coverage': coverage, 'turn_p95_deg': turn_p95, 'associated_fraction': associated_fraction, 'angle_consistency': angle_consistency, 'angle_delta_deg': float(angle_delta), 'angle_degrees': float(record.get('angle_degrees', global_angle_degrees)), 'road_support': road_support, 'bilateral_support': bilateral_support, 'road_centrality': centrality, 'signed_offset_fraction': signed_fraction, 'offset_fraction': offset_fraction, 'offset_fraction_mad': offset_mad, 'reliable_shape': reliable_shape, 'shape_score': shape_score})
    if p.enable_v30_auxiliary_navigation_fusion and candidates:
        for candidate in candidates:
            first_stations = np.asarray(candidate.get('station_positions', []), dtype=np.float64)
            if not len(first_stations):
                candidate['v30_consensus_track_count'] = 1
                candidate['v30_consensus_union_coverage'] = float(candidate.get('coverage', 0.0))
                candidate['v30_consensus_score'] = float(candidate.get('shape_score', 0.0))
                continue
            first_lo = float(first_stations.min())
            first_hi = float(first_stations.max())
            peers: list[dict] = []
            for other in candidates:
                if angular_distance_degrees(float(candidate.get('angle_degrees', global_angle_degrees)), float(other.get('angle_degrees', global_angle_degrees))) > float(p.v30_auxiliary_consensus_angle_deg):
                    continue
                second_stations = np.asarray(other.get('station_positions', []), dtype=np.float64)
                if not len(second_stations):
                    continue
                second_lo = float(second_stations.min())
                second_hi = float(second_stations.max())
                station_gap = max(0.0, max(first_lo, second_lo) - min(first_hi, second_hi))
                if station_gap > float(p.v30_auxiliary_consensus_station_gap):
                    continue
                peers.append(other)
            covered_stations: set[int] = set()
            for peer in peers:
                peer_stations = np.asarray(peer.get('observed_station_positions', []), dtype=np.float64)
                covered_stations.update((int(round(value)) for value in peer_stations if np.isfinite(value)))
            union_coverage = float(len(covered_stations) / max(1, len(centers)))
            source_diversity = len({str(peer.get('source_category', peer.get('source_family', ''))) for peer in peers})
            peer_score = float(sum((max(0.0, float(peer.get('shape_score', 0.0))) for peer in peers)))
            consensus_multiplier = float(1.0 + float(p.v30_auxiliary_consensus_score_boost) * max(0, len(peers) - 1) + 0.12 * max(0, source_diversity - 1))
            candidate['v30_consensus_track_count'] = int(len(peers))
            candidate['v30_consensus_source_diversity'] = int(source_diversity)
            candidate['v30_consensus_union_coverage'] = union_coverage
            candidate['v30_consensus_score'] = float(max(float(candidate.get('shape_score', 0.0)), peer_score) * consensus_multiplier)
            consensus_reliable = bool(len(peers) >= int(p.v30_auxiliary_consensus_min_tracks) and union_coverage >= float(p.v30_auxiliary_consensus_min_union_coverage) and (float(candidate.get('angle_consistency', 0.0)) >= 0.82) and (float(candidate.get('turn_p95_deg', 180.0)) <= max(28.0, float(p.v18_shape_max_turn_p95_deg))) and (float(candidate.get('associated_fraction', 0.0)) >= 0.35) and (float(candidate.get('road_support', 0.0)) >= 0.25))
            if consensus_reliable:
                candidate['reliable_shape'] = True
                candidate['v30_consensus_promoted'] = True
            else:
                candidate['v30_consensus_promoted'] = False
    else:
        for candidate in candidates:
            candidate['v30_consensus_track_count'] = 1
            candidate['v30_consensus_source_diversity'] = 1
            candidate['v30_consensus_union_coverage'] = float(candidate.get('coverage', 0.0))
            candidate['v30_consensus_score'] = float(candidate.get('shape_score', 0.0))
    roadline_candidates = [candidate for candidate in candidates if candidate['source_family'] == 'road_line' and candidate['reliable_shape'] and (candidate['coverage'] >= float(p.v18_direct_center_min_coverage)) and (candidate['bilateral_support'] >= 0.55) and (candidate['road_centrality'] >= float(p.v17_roadline_min_centrality)) and (candidate['offset_fraction_mad'] <= 0.2)]
    ranked_candidate: dict | None = None
    if len(roadline_candidates) >= 2:
        ranked = sorted(roadline_candidates, key=lambda item: item['offset_fraction'])
        if ranked[0]['offset_fraction'] <= float(p.v18_ranked_center_max_offset_fraction) and ranked[1]['offset_fraction'] - ranked[0]['offset_fraction'] >= float(p.v18_ranked_center_margin_fraction):
            ranked_candidate = ranked[0]
    direct_candidates = [candidate for candidate in roadline_candidates if candidate['offset_fraction'] <= float(p.v18_direct_center_max_offset_fraction)]
    if ranked_candidate is not None and ranked_candidate not in direct_candidates:
        direct_candidates.append(ranked_candidate)
    if direct_candidates:
        chosen = max(direct_candidates, key=lambda item: (item['coverage'] * item['bilateral_support'] * (0.4 + item['road_centrality']), -item['offset_fraction']))
        ranked_selected = bool(ranked_candidate is not None and chosen['record_index'] == ranked_candidate['record_index'] and (chosen['offset_fraction'] > float(p.v18_direct_center_max_offset_fraction)))
        selected = chosen.copy()
        selected['role'] = 'confirmed_centerline'
        selected['selection_reason'] = 'closest_of_multiple_roadlines' if ranked_selected else 'near_predicted_road_center'
        return selected
    shape_candidates = [candidate for candidate in candidates if candidate['reliable_shape']]
    if not shape_candidates:
        return None
    selected = max(shape_candidates, key=lambda item: (float(item.get('v30_consensus_score', item['shape_score'])), float(item['shape_score']), float(item['coverage']))).copy()
    selected['role'] = 'shape_shifted'
    selected['selection_reason'] = 'roadline_not_confirmed_as_center' if selected['source_family'] == 'road_line' else 'roadside_curve_matches_road_direction'
    return selected

# 【品質工具】計算中心對 Roadline 距離、邊界連續性及缺口恢復面積。
def corridor_geometry_metrics(centers: np.ndarray, left: np.ndarray, right: np.ndarray, central_roadline_mask: np.ndarray | None, inferred_internal_gap_mask: np.ndarray, analysis_corridor: np.ndarray) -> dict:
    centers = np.asarray(centers, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    count = min(len(centers), len(left), len(right))
    centers, left, right = (centers[:count], left[:count], right[:count])
    distances: list[float] = []
    if central_roadline_mask is not None and np.any(central_roadline_mask):
        mask = np.asarray(central_roadline_mask, dtype=bool)
        for center, left_point, right_point in zip(centers, left, right):
            xs, ys = _v13_section_samples(left_point, right_point, mask.shape)
            if not len(xs):
                continue
            hit = mask[ys, xs]
            if not np.any(hit):
                continue
            points = np.column_stack((xs[hit], ys[hit])).astype(np.float64)
            distances.append(float(np.min(np.linalg.norm(points - center, axis=1))))

    def boundary_turn_angles(points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return np.asarray([], dtype=np.float64)
        steps = np.diff(points, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        valid = (norms[:-1] > 1e-06) & (norms[1:] > 1e-06)
        if not np.any(valid):
            return np.asarray([], dtype=np.float64)
        first = steps[:-1][valid] / norms[:-1][valid, None]
        second = steps[1:][valid] / norms[1:][valid, None]
        cosine = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(cosine))
    turns = np.concatenate((boundary_turn_angles(left), boundary_turn_angles(right)))
    turn_p95 = float(np.percentile(turns, 95)) if len(turns) else 0.0
    continuity_score = float(np.clip(1.0 - turn_p95 / 45.0, 0.0, 1.0))
    recovered_pixels = int(np.asarray(inferred_internal_gap_mask, dtype=bool).sum())
    corridor_pixels = int(np.asarray(analysis_corridor, dtype=bool).sum())
    return {'centerline_roadline_distance_median_px': float(np.median(distances)) if distances else None, 'centerline_roadline_distance_p90_px': float(np.percentile(distances, 90)) if distances else None, 'centerline_roadline_sample_count': int(len(distances)), 'boundary_turn_p95_deg': turn_p95, 'boundary_continuity_score': continuity_score, 'internal_gap_recovered_pixels': recovered_pixels, 'internal_gap_recovered_ratio_of_corridor': float(recovered_pixels / max(1, corridor_pixels))}

# 【歷史重建】依可信輔助證據有限延長道路末端。
def _v13_extend_to_auxiliary_evidence(centers: np.ndarray, left: np.ndarray, right: np.ndarray, sources: list[str], records: list[dict], p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    if len(centers) < 3 or not records:
        return (centers, left, right, sources, [])
    extensions: list[dict] = []
    all_records = [record for record in records if record['source_family'] in {'road_line', 'roadside_boundary'}]
    if not all_records:
        return (centers, left, right, sources, [])
    step_px = max(2.0, float(p.row_band_height))

    def build(at_start: bool):
        endpoint_index = 0 if at_start else len(centers) - 1
        neighbor_index = 1 if at_start else len(centers) - 2
        endpoint = centers[endpoint_index]
        tangent = endpoint - centers[neighbor_index]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-06:
            return None
        tangent = tangent / norm
        pair = right[endpoint_index] - left[endpoint_index]
        width = float(np.linalg.norm(pair))
        if width < p.min_road_width:
            return None
        normal = pair / width
        candidates: list[tuple[float, float, str]] = []
        roadside_sides: set[int] = set()
        families: set[str] = set()
        for record in all_records:
            points = np.asarray(record['points'], dtype=np.float64)
            delta = points - endpoint
            longitudinal = delta @ tangent
            lateral = delta @ normal
            keep = (longitudinal > step_px * 1.5) & (longitudinal <= step_px * p.v13_auxiliary_terminal_max_bands) & (np.abs(lateral) <= width * 0.95)
            for forward, side in zip(longitudinal[keep], lateral[keep]):
                candidates.append((float(forward), float(side), record['source_family']))
                families.add(record['source_family'])
                if record['source_family'] == 'roadside_boundary':
                    roadside_sides.add(-1 if side < 0 else 1)
        if not candidates:
            return None
        if 'road_line' not in families and len(roadside_sides) < 2:
            return None
        maximum_forward = float(np.percentile([item[0] for item in candidates], 95))
        band_count = min(int(p.v13_auxiliary_terminal_max_bands), int(math.floor(maximum_forward / step_px)))
        if band_count < int(p.v13_auxiliary_terminal_min_span_bands):
            return None
        new_centers = []
        new_left = []
        new_right = []
        for step in range(1, band_count + 1):
            fraction = step / max(1, band_count)
            current_width = width * (1.0 - fraction * (1.0 - p.v13_auxiliary_terminal_width_taper))
            center = endpoint + tangent * (step * step_px)
            new_centers.append(center)
            new_left.append(center - normal * current_width / 2.0)
            new_right.append(center + normal * current_width / 2.0)
        if at_start:
            new_centers = new_centers[::-1]
            new_left = new_left[::-1]
            new_right = new_right[::-1]
        return (np.asarray(new_centers, dtype=np.float64), np.asarray(new_left, dtype=np.float64), np.asarray(new_right, dtype=np.float64), {'end': 'start' if at_start else 'end', 'bands': int(band_count), 'length_px': float(band_count * step_px), 'source_families': sorted(families)})
    for at_start in (True, False):
        built = build(at_start)
        if built is None:
            continue
        ext_centers, ext_left, ext_right, info = built
        ext_sources = ['auxiliary_supported_extrapolated'] * len(ext_centers)
        if at_start:
            centers = np.vstack((ext_centers, centers))
            left = np.vstack((ext_left, left))
            right = np.vstack((ext_right, right))
            sources = ext_sources + sources
        else:
            centers = np.vstack((centers, ext_centers))
            left = np.vstack((left, ext_left))
            right = np.vstack((right, ext_right))
            sources = sources + ext_sources
        extensions.append(info)
    return (centers, left, right, sources, extensions)

# 【資料工具】複製目前有效的中心、左右邊界、來源與走廊資料供安全修改。
def _copy_active_corridor_geometry(corridor: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    transformed = dict(corridor)
    centers = np.asarray(corridor.get('centerline', []), dtype=np.float64).copy()
    left = np.asarray(corridor.get('left', []), dtype=np.float64).copy()
    right = np.asarray(corridor.get('right', []), dtype=np.float64).copy()
    sources = list(corridor.get('band_source_types', []))
    count = min(len(centers), len(left), len(right), len(sources))
    return (transformed, centers[:count], left[:count], right[:count], sources[:count], count)

# 【階段函式】套用 Roadline 內部支持、輔助邊界與早期末端重建規則。
def apply_v13_auxiliary_road_reconstruction(road: np.ndarray, obstacles: np.ndarray, corridor: dict, auxiliary_masks: dict[str, np.ndarray] | None, p: Parameters) -> dict:
    if not p.enable_auxiliary_road_reconstruction or not corridor.get('ok'):
        return corridor
    transformed, centers, left, right, sources, count = _copy_active_corridor_geometry(corridor)
    if count < 3:
        return transformed
    working_corridor = dict(corridor)
    working_corridor.update({'centerline': centers, 'left': left, 'right': right})
    records, selected_masks, v15_track_summary = _v13_associated_auxiliary_components(auxiliary_masks, working_corridor, road.shape, p)
    if not records:
        transformed['v13_reconstruction_status'] = 'no_associated_area_evidence'
        transformed['v13_reconstruction_summary'] = {'v15_auxiliary_track_summary': v15_track_summary}
        return transformed
    widths = np.linalg.norm(right - left, axis=1)
    normals = (right - left) / np.maximum(widths[:, None], 1e-06)
    global_angle_degrees = float(corridor.get('selected_direction_angle_deg', 90.0))
    v17_roadline_reference = _v18_select_auxiliary_curve_reference(records, road, centers, widths, global_angle_degrees, p, float(corridor.get('selected_joint_auxiliary_score', 0.0)))
    center_offsets: dict[int, list[float]] = {}
    central_target_points: dict[int, list[np.ndarray]] = {}
    edge_target_points: dict[int, list[np.ndarray]] = {}
    central_roadline_mask = np.zeros_like(road, dtype=bool)
    edge_roadline_mask = np.zeros_like(road, dtype=bool)
    roadline_station_records: list[dict] = []

    def mark_central_point(point: np.ndarray):
        x, y = np.rint(point).astype(np.int32)
        if 0 <= x < road.shape[1] and 0 <= y < road.shape[0]:
            central_roadline_mask[y, x] = True

    def mark_edge_point(point: np.ndarray):
        x, y = np.rint(point).astype(np.int32)
        if 0 <= x < road.shape[1] and 0 <= y < road.shape[0]:
            edge_roadline_mask[y, x] = True
    for record in records:
        if record['source_family'] != 'road_line':
            continue
        record_span = int(record['station_end'] - record['station_start'] + 1)
        if record_span < int(p.v13_roadline_center_min_span_bands):
            continue
        grouped: dict[int, dict[str, list]] = {}
        for station, offset, point in zip(record['station_indices'], record['signed_offsets'], np.asarray(record['points'], dtype=np.float64)):
            item = grouped.setdefault(int(station), {'offsets': [], 'points': []})
            item['offsets'].append(float(offset))
            item['points'].append(np.asarray(point, dtype=np.float64))
        station_data = {station: {'offset': float(np.median(item['offsets'])), 'point': np.median(np.asarray(item['points']), axis=0)} for station, item in grouped.items()}
        if station_data:
            roadline_station_records.append({'angle_degrees': float(record['angle_degrees']), 'stations': station_data, 'is_confirmed_center': False})
    single_center_station_count = 0
    paired_center_station_count = 0
    for record in roadline_station_records:
        accepted = []
        for station, item in record['stations'].items():
            if abs(item['offset']) <= float(widths[station]) * float(p.v13_roadline_single_center_fraction):
                accepted.append((station, item))
        if len(accepted) < int(p.v13_roadline_center_min_span_bands):
            continue
        record['is_confirmed_center'] = True
        for station, item in accepted:
            central_target_points.setdefault(station, []).append(item['point'])
            mark_central_point(item['point'])
            single_center_station_count += 1
    for first_index in range(len(roadline_station_records)):
        first = roadline_station_records[first_index]
        for second in roadline_station_records[first_index + 1:]:
            if angular_distance_degrees(first['angle_degrees'], second['angle_degrees']) > 8.0:
                continue
            overlap = sorted(set(first['stations']) & set(second['stations']))
            if len(overlap) < int(p.v13_roadline_pair_min_overlap_bands):
                continue
            separations = np.asarray([abs(first['stations'][station]['offset'] - second['stations'][station]['offset']) / max(float(widths[station]), 1.0) for station in overlap])
            median_separation = float(np.median(separations))
            if not 0.008 <= median_separation <= float(p.v13_roadline_pair_max_separation_fraction):
                continue
            pair_values = []
            for station in overlap:
                first_item = first['stations'][station]
                second_item = second['stations'][station]
                midpoint = 0.5 * (first_item['offset'] + second_item['offset'])
                if abs(midpoint) > float(widths[station]) * float(p.v13_roadline_center_max_offset_fraction):
                    continue
                pair_values.append((station, midpoint, first_item, second_item))
            if len(pair_values) < int(p.v13_roadline_pair_min_overlap_bands):
                continue
            first['is_confirmed_center'] = True
            second['is_confirmed_center'] = True
            for station, midpoint, first_item, second_item in pair_values:
                midpoint_point = 0.5 * (np.asarray(first_item['point'], dtype=np.float64) + np.asarray(second_item['point'], dtype=np.float64))
                central_target_points.setdefault(station, []).append(midpoint_point)
                mark_central_point(first_item['point'])
                mark_central_point(second_item['point'])
                paired_center_station_count += 1
    if paired_center_station_count > 0 and len(central_target_points) >= 2:
        pair_stations = np.asarray(sorted(central_target_points), dtype=np.int32)
        pair_points = np.asarray([np.median(np.asarray(central_target_points[int(index)], dtype=np.float64), axis=0) for index in pair_stations], dtype=np.float64)
        pair_curve = np.column_stack([regularized_curve_fit(pair_points[:, coordinate], np.ones(len(pair_points), dtype=np.float64), max(3.0, float(p.v16_curve_smoothness)), int(p.v16_curve_robust_iterations)) for coordinate in range(2)])
        pair_offsets = np.linalg.norm(pair_curve - centers[pair_stations], axis=1) / np.maximum(widths[pair_stations], 1.0)
        radians = math.radians(global_angle_degrees)
        pair_axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
        v17_roadline_reference = {'source_family': 'road_line', 'source_category': 'road_line_pair', 'curve_points': pair_curve, 'u_values': pair_curve @ pair_axis, 'station_positions': pair_stations, 'observed_station_positions': pair_stations, 'span_px': float(polyline_length(pair_curve)), 'coverage': float((pair_stations[-1] - pair_stations[0] + 1) / max(1, len(centers))), 'road_support': 1.0, 'road_centrality': 0.5, 'offset_fraction': float(np.median(pair_offsets)), 'role': 'confirmed_centerline', 'selection_reason': 'paired_center_roadlines'}
    elif single_center_station_count > 0 and len(central_target_points) >= 2:
        center_stations = np.asarray(sorted(central_target_points), dtype=np.int32)
        center_points = np.asarray([np.median(np.asarray(central_target_points[int(index)], dtype=np.float64), axis=0) for index in center_stations], dtype=np.float64)
        center_curve = np.column_stack([regularized_curve_fit(center_points[:, coordinate], np.ones(len(center_points), dtype=np.float64), max(3.0, float(p.v16_curve_smoothness)), int(p.v16_curve_robust_iterations)) for coordinate in range(2)])
        center_reference_offsets = np.linalg.norm(center_curve - centers[center_stations], axis=1) / np.maximum(widths[center_stations], 1.0)
        radians = math.radians(global_angle_degrees)
        center_axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
        v17_roadline_reference = {'source_family': 'road_line', 'source_category': 'road_line_local_center', 'curve_points': center_curve, 'u_values': center_curve @ center_axis, 'station_positions': center_stations, 'observed_station_positions': center_stations, 'span_px': float(polyline_length(center_curve)), 'coverage': float((center_stations[-1] - center_stations[0] + 1) / max(1, len(centers))), 'road_support': 1.0, 'road_centrality': 0.5, 'offset_fraction': float(np.median(center_reference_offsets)), 'role': 'confirmed_centerline', 'selection_reason': 'local_confirmed_central_roadline'}
    edge_roadline_station_count = 0
    edge_roadline_component_count = 0
    if not central_target_points:
        for record in roadline_station_records:
            if record['is_confirmed_center']:
                continue
            stations = sorted(record['stations'])
            if len(stations) < int(p.v13_roadline_center_min_span_bands):
                continue
            fractions = np.asarray([float(record['stations'][station]['offset']) / max(float(widths[station]), 1.0) for station in stations])
            median_fraction = float(np.median(fractions))
            fraction_mad = float(np.median(np.abs(fractions - median_fraction)))
            if not float(p.v14_roadline_edge_min_offset_fraction) <= abs(median_fraction) <= float(p.v14_roadline_edge_max_offset_fraction):
                continue
            if fraction_mad > float(p.v14_roadline_edge_max_fraction_mad):
                continue
            component_used = 0
            for station in stations:
                item = record['stations'][station]
                translated = np.asarray(item['point'], dtype=np.float64) - normals[station] * median_fraction * widths[station]
                edge_target_points.setdefault(station, []).append(translated)
                mark_edge_point(item['point'])
                edge_roadline_station_count += 1
                component_used += 1
            if component_used:
                edge_roadline_component_count += 1
    active_target_points = central_target_points if central_target_points else edge_target_points
    roadline_center_mode = 'v18_confirmed_central_roadline' if v17_roadline_reference is not None and v17_roadline_reference.get('role') == 'confirmed_centerline' else 'v18_auxiliary_curve_shape_shifted' if v17_roadline_reference is not None else 'central_roadline_direct' if central_target_points else 'edge_roadline_shape_shifted' if edge_target_points else 'road_geometry_fallback'
    center_anchor_count = 0
    center_fit_observation_count = len(active_target_points)
    if len(active_target_points) >= 2 and v17_roadline_reference is None:
        indices = np.asarray(sorted(active_target_points), dtype=np.int32)
        values = np.asarray([np.median(np.asarray(active_target_points[int(index)], dtype=np.float64), axis=0) for index in indices])
        lo, hi = (int(indices.min()), int(indices.max()))
        local_indices = np.arange(lo, hi + 1)
        target_weights = np.zeros(len(local_indices), dtype=np.float64)
        target_weights[indices - lo] = float(p.v13_roadline_center_fit_weight)
        fitted_targets = np.column_stack([regularized_curve_fit(np.interp(local_indices, indices, values[:, axis]), target_weights, p.v13_roadline_center_fit_smoothness, p.curve_robust_iterations) for axis in range(2)])
        blend = float(p.v14_roadline_direct_center_blend if central_target_points else p.v14_roadline_edge_shape_blend)
        for index, target in zip(range(lo, hi + 1), fitted_targets):
            delta = np.asarray(target, dtype=np.float64) - centers[index]
            maximum_shift = widths[index] * 0.6
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > maximum_shift:
                delta *= maximum_shift / max(delta_norm, 1e-06)
            delta *= blend
            centers[index] += delta
            left[index] += delta
            right[index] += delta
            center_offsets.setdefault(index, []).append(float(delta @ normals[index]))
            center_anchor_count += 1
    widths = np.linalg.norm(right - left, axis=1)
    normals = (right - left) / np.maximum(widths[:, None], 1e-06)
    boundary_offsets: dict[int, dict[int, list[float]]] = {}
    single_component_candidates: list[dict] = []
    for record in records:
        if record['source_family'] != 'roadside_boundary':
            continue
        component_offsets: dict[int, list[float]] = {}
        points = np.asarray(record['points'], dtype=np.float64)
        station_indices = np.asarray(record['station_indices'], dtype=np.int32)
        current_offsets = np.einsum('ij,ij->i', points - centers[station_indices], normals[station_indices])
        for station, offset in zip(station_indices, current_offsets):
            station = int(station)
            width = float(widths[station])
            fraction = abs(float(offset)) / max(width, 1.0)
            if not p.v13_boundary_min_offset_fraction <= fraction <= p.v13_boundary_max_offset_fraction:
                continue
            side = -1 if float(offset) < 0 else 1
            boundary_offsets.setdefault(station, {}).setdefault(side, []).append(float(offset))
            component_offsets.setdefault(station, []).append(float(offset))
        if not component_offsets:
            continue
        component_values = np.asarray([float(np.median(values)) for values in component_offsets.values()])
        side = -1 if float(np.median(component_values)) < 0 else 1
        same_side = {index: values for index, values in component_offsets.items() if (-1 if float(np.median(values)) < 0 else 1) == side}
        if same_side:
            stations = sorted(same_side)
            single_component_candidates.append({'side': side, 'station_offsets': same_side, 'station_start': int(stations[0]), 'station_end': int(stations[-1]), 'station_observations': int(len(stations)), 'span_bands': int(stations[-1] - stations[0] + 1)})
    boundary_anchor_count = 0
    paired_boundary_anchor_count = 0
    single_boundary_hint_count = 0
    long_single_boundary_anchor_count = 0
    for index, sides in boundary_offsets.items():
        pair = right[index] - left[index]
        width = float(np.linalg.norm(pair))
        if width < p.min_road_width:
            continue
        normal = pair / width
        center = (left[index] + right[index]) / 2.0
        paired = -1 in sides and 1 in sides
        if not paired:
            neighborhood = range(max(0, index - 6), min(len(centers), index + 7))
            seen_sides = {side for neighbor in neighborhood for side in boundary_offsets.get(neighbor, {})}
            paired = -1 in seen_sides and 1 in seen_sides
        if not paired:
            single_boundary_hint_count += sum((len(values) for values in sides.values()))
            continue
        for side, offsets in sides.items():
            measured = float(np.median(offsets))
            distance = float(np.clip(abs(measured), 0.425 * width, 0.65 * width))
            target = center + normal * side * distance
            if side < 0:
                left[index] = (1.0 - p.v13_boundary_blend) * left[index] + p.v13_boundary_blend * target
            else:
                right[index] = (1.0 - p.v13_boundary_blend) * right[index] + p.v13_boundary_blend * target
            boundary_anchor_count += 1
            paired_boundary_anchor_count += 1
    width_model = regularized_curve_fit(np.maximum(widths, float(p.min_road_width)), np.ones(len(widths), dtype=np.float64), max(2.0, p.width_curve_smoothness * 0.25), p.curve_robust_iterations)
    long_single_components_used = 0
    long_single_modified_indices: set[int] = set()
    for component in single_component_candidates:
        if not bool(p.v18_allow_single_boundary_position_constraint):
            continue
        if component['span_bands'] < int(p.v13_single_boundary_min_span_bands) or component['station_observations'] < int(p.v13_single_boundary_min_observations):
            continue
        side = int(component['side'])
        lo, hi = (int(component['station_start']), int(component['station_end']))
        paired_stations = sum((-1 in boundary_offsets.get(index, {}) and 1 in boundary_offsets.get(index, {}) for index in range(lo, hi + 1)))
        if paired_stations >= max(2, int(0.35 * (hi - lo + 1))):
            continue
        station_offsets = component['station_offsets']
        observed_indices = np.asarray(sorted(station_offsets), dtype=np.int32)
        observed_values = np.asarray([float(np.median(station_offsets[int(index)])) for index in observed_indices])
        interpolated_offsets = np.interp(np.arange(lo, hi + 1), observed_indices, observed_values)
        component_used = 0
        for index, measured in zip(range(lo, hi + 1), interpolated_offsets):
            if index < 0 or index >= len(centers):
                continue
            pair = right[index] - left[index]
            pair_width = float(np.linalg.norm(pair))
            if pair_width < p.min_road_width:
                continue
            normal = pair / pair_width
            model_width = max(float(p.min_road_width), float(width_model[index]))
            distance = float(np.clip(abs(measured), 0.35 * model_width, 0.78 * model_width))
            target = centers[index] + normal * side * distance
            blend = float(p.v13_single_boundary_blend)
            if side < 0:
                constrained = (1.0 - blend) * left[index] + blend * target
                left[index] = constrained
                right[index] = constrained + normal * model_width
            else:
                constrained = (1.0 - blend) * right[index] + blend * target
                right[index] = constrained
                left[index] = constrained - normal * model_width
            centers[index] = (left[index] + right[index]) / 2.0
            if sources[index] not in {'extrapolated', 'auxiliary_supported_extrapolated'}:
                sources[index] = 'auxiliary_boundary_inferred'
            boundary_anchor_count += 1
            long_single_boundary_anchor_count += 1
            long_single_modified_indices.add(index)
            component_used += 1
        if component_used:
            long_single_components_used += 1
    centers = (left + right) / 2.0
    v17_guided_indices: set[int] = set()
    v17_guided_mask = np.zeros(len(centers), dtype=bool)
    if v17_roadline_reference is not None:
        for point in np.asarray(v17_roadline_reference['curve_points'], dtype=np.float64):
            if v17_roadline_reference.get('role') == 'confirmed_centerline':
                mark_central_point(point)
            else:
                mark_edge_point(point)
    geometry_fit_weights = np.ones(len(centers), dtype=np.float64)
    if center_offsets:
        center_fit_indices = range(min(center_offsets), max(center_offsets) + 1)
        geometry_fit_weights[list(center_fit_indices)] = float(p.v13_roadline_center_fit_weight)
    if long_single_modified_indices:
        geometry_fit_weights[list(long_single_modified_indices)] = np.maximum(geometry_fit_weights[list(long_single_modified_indices)], 2.0)
    fitted_centers = np.column_stack([regularized_curve_fit(centers[:, axis], geometry_fit_weights, p.v13_roadline_center_fit_smoothness, p.curve_robust_iterations) for axis in range(2)])
    fitted_widths = regularized_curve_fit(np.linalg.norm(right - left, axis=1), geometry_fit_weights, max(2.0, p.width_curve_smoothness * 0.2), p.curve_robust_iterations)
    fitted_widths = np.maximum(fitted_widths, float(p.min_road_width))
    cross_units = (right - left) / np.maximum(np.linalg.norm(right - left, axis=1)[:, None], 1e-06)
    reference_normal = cross_units[len(cross_units) // 2]
    aligned_units = cross_units.copy()
    aligned_units[np.sum(aligned_units * reference_normal, axis=1) < 0] *= -1.0
    global_normal = np.median(aligned_units, axis=0)
    global_normal /= max(float(np.linalg.norm(global_normal)), 1e-06)
    centers = fitted_centers
    left = centers - global_normal[None, :] * fitted_widths[:, None] / 2.0
    right = centers + global_normal[None, :] * fitted_widths[:, None] / 2.0
    centers, left, right, v17_guided_indices, v17_guided_mask = _v17_apply_central_roadline_reference(centers, left, right, v17_roadline_reference, global_angle_degrees, p)
    station_normals = np.repeat(global_normal[None, :], len(centers), axis=0)
    if v17_guided_indices:
        transition = max(0, int(p.v18_curve_support_transition_bands if v17_roadline_reference is not None and 'observed_station_positions' in v17_roadline_reference else p.v17_center_transition_bands))
        guided_array = np.asarray(sorted(v17_guided_indices), dtype=np.int32)
        reference_curve = None
        reference_stations = None
        reference_tangents = None
        if v17_roadline_reference is not None and str(v17_roadline_reference.get('role', '')) == 'shape_shifted':
            candidate_curve = np.asarray(v17_roadline_reference.get('curve_points', []), dtype=np.float64)
            candidate_stations = np.asarray(v17_roadline_reference.get('station_positions', []), dtype=np.float64)
            if len(candidate_curve) >= 2 and len(candidate_curve) == len(candidate_stations):
                reference_curve = candidate_curve
                reference_stations = candidate_stations
                reference_tangents = np.gradient(reference_curve, axis=0)
        for index in range(len(centers)):
            distance = int(np.min(np.abs(guided_array - index)))
            if distance > transition:
                continue
            strength = 1.0 if distance == 0 else 1.0 - distance / max(1.0, transition + 1.0)
            if reference_tangents is not None and reference_stations is not None and (float(reference_stations.min()) <= index <= float(reference_stations.max())):
                tangent = np.asarray([np.interp(float(index), reference_stations, reference_tangents[:, coordinate]) for coordinate in range(2)], dtype=np.float64)
            else:
                tangent = fitted_local_tangent(centers, index, max(1.0, float(p.row_band_height)), max(12.0, float(p.v16_tangent_window_stations) * float(p.row_band_height)))
            tangent /= max(float(np.linalg.norm(tangent)), 1e-06)
            local_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            if float(local_normal @ global_normal) < 0:
                local_normal *= -1.0
            blended = (1.0 - strength) * global_normal + strength * local_normal
            station_normals[index] = blended / max(float(np.linalg.norm(blended)), 1e-06)
    left = centers - station_normals * fitted_widths[:, None] / 2.0
    right = centers + station_normals * fitted_widths[:, None] / 2.0
    centers, left, right, sources, terminal_extensions = _v13_extend_to_auxiliary_evidence(centers, left, right, sources, records, p)
    analysis = corridor_polygon_mask(road.shape, left, right)
    auxiliary_road_envelope, auxiliary_road_envelope_polygons, auxiliary_road_envelope_summary = _v19_auxiliary_road_envelope(road, obstacles, centers, left, right, records, p, confirmed_centerline_available=bool(v17_roadline_reference is not None and str(v17_roadline_reference.get('role', '')) == 'confirmed_centerline' and (str(v17_roadline_reference.get('source_family', '')) == 'road_line')))
    centers, left, right, sources, auxiliary_envelope_geometry_summary = _v19_expand_boundaries_to_envelope(road, auxiliary_road_envelope, centers, left, right, sources, p)
    analysis = corridor_polygon_mask(road.shape, left, right)
    roadline_mask = selected_masks['road_line'] & analysis
    central_roadline_mask &= analysis
    section_road_ratio = np.zeros(len(centers), dtype=np.float64)
    section_line_hit = np.zeros(len(centers), dtype=bool)
    for index, (left_point, right_point) in enumerate(zip(left, right)):
        xs, ys = _v13_section_samples(left_point, right_point, road.shape)
        if not len(xs):
            continue
        section_road_ratio[index] = float(road[ys, xs].mean())
        section_line_hit[index] = bool(np.any(roadline_mask[ys, xs]))
    anchor = section_road_ratio >= float(p.v13_roadline_anchor_width_ratio)
    radius = max(2, int(p.v13_internal_anchor_window_bands))
    roadline_internal_indices: list[int] = []
    for index in range(1, len(centers) - 1):
        if sources[index] == 'auxiliary_supported_extrapolated':
            continue
        if section_road_ratio[index] >= float(p.v13_roadline_internal_width_ratio):
            continue
        if not section_line_hit[index]:
            continue
        before = np.flatnonzero(anchor[max(0, index - radius):index])
        after = np.flatnonzero(anchor[index + 1:min(len(anchor), index + radius + 1)])
        if not len(before) or not len(after):
            continue
        sources[index] = 'roadline_supported_internal'
        roadline_internal_indices.append(index)
    v13_internal_mask = mask_from_boundary_sources(road.shape, left, right, sources, {'roadline_supported_internal'}, line_width=max(2, int(p.row_band_height))) & ~road
    existing_internal = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool)
    inferred_internal = existing_internal | v13_internal_mask | roadline_mask | auxiliary_road_envelope
    inferred_occlusion = analysis & obstacles & ~road
    obstacle_supported = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(road)), dtype=bool) & analysis & ~road
    full = road | inferred_internal | inferred_occlusion | obstacle_supported
    geometry_metrics = corridor_geometry_metrics(centers, left, right, central_roadline_mask, inferred_internal, analysis)
    transformed.update({'centerline': centers, 'left': left, 'right': right, 'band_source_types': sources, 'source_counts': {source: int(sum((value == source for value in sources))) for source in ('direct', 'interpolated', 'interior_occlusion_inferred', 'one_side_inferred', 'extrapolated')}, 'one_sided_extrapolations': list(corridor.get('one_sided_extrapolations', [])), 'one_sided_geometry': list(corridor.get('one_sided_geometry', [])), 'one_side_inferences': list(corridor.get('one_side_inferences', [])), 'interior_occlusion_inferences': list(corridor.get('interior_occlusion_inferences', [])), 'interpolation_ratio': float(sum((source == 'interpolated' for source in sources)) / max(1, len(sources))), 'extrapolation_ratio': float(sum((source == 'extrapolated' for source in sources)) / max(1, len(sources))), 'analysis_geometry_corridor': analysis, 'inferred_geometry_envelope': analysis, 'inferred_internal_gap_mask': inferred_internal, 'inferred_occlusion_mask': inferred_occlusion, 'obstacle_supported_reconstruction_mask': obstacle_supported, 'auxiliary_road_envelope_mask': auxiliary_road_envelope, 'auxiliary_road_envelope_polygons': auxiliary_road_envelope_polygons, 'full_road_corridor': full, 'corridor_mask': full, 'geometry_metrics': geometry_metrics, 'v13_associated_roadline_mask': roadline_mask, 'v13_central_roadline_mask': central_roadline_mask, 'v14_edge_roadline_mask': edge_roadline_mask & analysis, 'v18_selected_curve_reference': v17_roadline_reference, 'v13_associated_roadside_mask': selected_masks['roadside_boundary'] & analysis, 'v13_reconstruction_status': 'auxiliary_area_evidence_applied', 'v13_reconstruction_summary': {'v15_auxiliary_track_summary': v15_track_summary, 'v15_accepted_track_count': int(sum((item.get('accepted_track_count', 0) for item in v15_track_summary.values()))), 'v15_merged_track_count': int(sum((item.get('merged_track_count', 0) for item in v15_track_summary.values()))), 'v15_rejected_track_count': int(sum((item.get('rejected_track_count', 0) for item in v15_track_summary.values()))), 'associated_component_count': int(len(records)), 'center_anchor_bands': int(center_anchor_count), 'center_fit_observation_bands': int(center_fit_observation_count), 'single_center_roadline_observations': int(single_center_station_count), 'paired_center_roadline_observations': int(paired_center_station_count), 'center_fit_weight': float(p.v13_roadline_center_fit_weight), 'roadline_center_mode': roadline_center_mode, 'v17_auxiliary_guided_geometry': bool(v17_roadline_reference is not None), 'v17_guided_center_bands': int(len(v17_guided_indices)), 'v17_reference_span_px': float(v17_roadline_reference['span_px'] if v17_roadline_reference is not None else 0.0), 'v17_reference_road_support': float(v17_roadline_reference['road_support'] if v17_roadline_reference is not None else 0.0), 'v17_reference_centrality': float(v17_roadline_reference['road_centrality'] if v17_roadline_reference is not None else 0.0), 'v17_guardrail_ridge_mode': True, 'v18_auxiliary_curve_role': str(v17_roadline_reference.get('role', 'none')) if v17_roadline_reference is not None else 'none', 'v18_auxiliary_curve_source': str(v17_roadline_reference.get('source_category', 'none')) if v17_roadline_reference is not None else 'none', 'v18_auxiliary_curve_selection_reason': str(v17_roadline_reference.get('selection_reason', 'none')) if v17_roadline_reference is not None else 'none', 'v18_auxiliary_curve_coverage': float(v17_roadline_reference.get('coverage', 0.0) if v17_roadline_reference is not None else 0.0), 'v18_auxiliary_curve_offset_fraction': float(v17_roadline_reference.get('offset_fraction', 0.0) if v17_roadline_reference is not None else 0.0), 'v18_guardrail_curve_mode': 'road_facing_edge_smoothed', 'central_target_observation_bands': int(len(central_target_points)), 'edge_shape_target_observation_bands': int(len(edge_target_points)), 'edge_roadline_observations': int(edge_roadline_station_count), 'edge_roadline_components': int(edge_roadline_component_count), 'boundary_anchor_observations': int(boundary_anchor_count), 'paired_boundary_anchor_observations': int(paired_boundary_anchor_count), 'single_boundary_hint_observations': int(single_boundary_hint_count), 'long_single_boundary_anchor_bands': int(long_single_boundary_anchor_count), 'long_single_boundary_components': int(long_single_components_used), 'single_boundary_position_constraint_enabled': bool(p.v18_allow_single_boundary_position_constraint), 'single_boundary_shape_only_components': int(len(single_component_candidates) if not p.v18_allow_single_boundary_position_constraint else 0), 'roadline_internal_bands': int(len(roadline_internal_indices)), 'roadline_support_pixels': int(roadline_mask.sum()), 'roadline_internal_pixels': int(v13_internal_mask.sum()), 'terminal_extension_count': int(len(terminal_extensions)), 'terminal_extension_bands': int(sum((item['bands'] for item in terminal_extensions))), 'terminal_extensions': terminal_extensions, 'v19_auxiliary_road_envelope': auxiliary_road_envelope_summary, 'v19_auxiliary_envelope_geometry': auxiliary_envelope_geometry_summary}})
    transformed['source_counts'] = {source: int(sum((value == source for value in sources))) for source in SOURCE_WEIGHTS}
    transformed['observed_bands'] = int(sum((source == 'direct' for source in sources)))
    transformed['span_bands'] = int(len(sources))
    transformed['interpolation_ratio'] = float(sum((source in {'interpolated', 'roadline_supported_internal'} for source in sources)) / max(1, len(sources)))
    transformed['extrapolation_ratio'] = float(sum((source in {'extrapolated', 'auxiliary_supported_extrapolated'} for source in sources)) / max(1, len(sources)))
    transformed['interpolated_flags'] = [source in {'interpolated', 'interior_occlusion_inferred', 'roadline_supported_internal'} for source in sources]
    transformed['extrapolated_flags'] = [source in {'extrapolated', 'auxiliary_supported_extrapolated'} for source in sources]
    return refresh_reconstruction_metadata(transformed)

# 【相容入口】將舊的統一證據重建名稱轉交目前 V13 重建流程。
def apply_unified_evidence_reconstruction(road: np.ndarray, obstacles: np.ndarray, corridor: dict, auxiliary_masks: dict[str, np.ndarray] | None, p: Parameters) -> dict:
    reconstructed = apply_v13_auxiliary_road_reconstruction(road, obstacles, corridor, auxiliary_masks, p)
    return refresh_reconstruction_metadata(reconstructed)

# =============================================================================
# 13. 導航中心位置與方向場
# =============================================================================
# 分離「道路位於哪裡」與「道路往哪裡走」：Road 提供基礎位置及寬度，
# 可信 Roadline／側邊曲線可修正局部方向、曲率或中央位置。
# 【遮罩工具】將選中的輔助參考曲線畫成二值遮罩。
def _v20_selected_curve_mask(shape: tuple[int, int], reference: dict | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if reference is None:
        return mask
    points = np.asarray(reference.get('curve_points', []), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        return mask
    canvas = Image.new('1', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    draw.line([tuple(value) for value in points[:, :2]], fill=1, width=3)
    return np.asarray(canvas, dtype=bool)

# 【曲線工具】把輔助參考曲線重採樣成位置、切線及正規化站序。
def _v21_reference_samples(reference: dict | None, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference is None or count < 2:
        empty_points = np.empty((0, 2), dtype=np.float64)
        return (empty_points, np.empty(0), np.empty(0))
    curve = np.asarray(reference.get('curve_points', []), dtype=np.float64)
    stations = np.asarray(reference.get('station_positions', []), dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] < 2 or len(curve) < 2 or (len(curve) != len(stations)):
        empty_points = np.empty((0, 2), dtype=np.float64)
        return (empty_points, np.empty(0), np.empty(0))
    finite = np.all(np.isfinite(curve[:, :2]), axis=1) & np.isfinite(stations)
    curve, stations = (curve[finite, :2], stations[finite])
    if len(curve) < 2:
        return (curve, stations, np.empty(0))
    order = np.argsort(stations)
    curve, stations = (curve[order], stations[order])
    observed = np.asarray(reference.get('observed_station_positions', stations), dtype=np.float64)
    observed = observed[np.isfinite(observed) & (observed >= 0.0) & (observed <= count - 1)]
    return (curve, stations, np.unique(observed))

# 【角色工具】檢查輔助參考被當成中心、邊線或一般方向是否合理。
def _v21_validate_reference_role(reference: dict | None, centers: np.ndarray, left: np.ndarray, right: np.ndarray, p: Parameters) -> tuple[dict | None, dict]:
    diagnostics = {'checked': False, 'passed': False, 'reason': 'no_reference', 'sample_count': 0, 'inside_ratio': 0.0, 'median_balance': None, 'p90_balance': None, 'signed_offset_mad': None}
    if reference is None:
        return (None, diagnostics)
    checked = dict(reference)
    original_selection_reason = str(checked.get('selection_reason', 'unknown'))
    diagnostics['original_selection_reason'] = original_selection_reason
    if str(checked.get('role', '')) != 'confirmed_centerline' or str(checked.get('source_family', '')) != 'road_line':
        diagnostics.update({'reason': 'shape_only_reference'})
        return (checked, diagnostics)
    curve, stations, observed = _v21_reference_samples(checked, len(centers))
    diagnostics['checked'] = True
    if len(curve) < 2 or len(observed) < 4:
        checked['role'] = 'shape_shifted'
        checked['selection_reason'] = 'bilateral_center_validation_failed'
        diagnostics['reason'] = 'too_few_bilateral_samples'
        return (checked, diagnostics)
    observed = observed[(observed >= float(stations.min())) & (observed <= float(stations.max()))]
    points = np.column_stack([np.interp(observed, stations, curve[:, coordinate]) for coordinate in range(2)])
    station_axis = np.arange(len(centers), dtype=np.float64)
    base_centers = np.column_stack([np.interp(observed, station_axis, centers[:, coordinate]) for coordinate in range(2)])
    base_left = np.column_stack([np.interp(observed, station_axis, left[:, coordinate]) for coordinate in range(2)])
    base_right = np.column_stack([np.interp(observed, station_axis, right[:, coordinate]) for coordinate in range(2)])
    pairs = base_right - base_left
    widths = np.linalg.norm(pairs, axis=1)
    valid = widths >= float(p.min_road_width)
    if int(valid.sum()) < 4:
        checked['role'] = 'shape_shifted'
        checked['selection_reason'] = 'bilateral_center_validation_failed'
        diagnostics['reason'] = 'invalid_v12_boundary_width'
        return (checked, diagnostics)
    points, base_centers, pairs, widths = (points[valid], base_centers[valid], pairs[valid], widths[valid])
    normals = pairs / widths[:, None]
    signed = np.sum((points - base_centers) * normals, axis=1) / widths
    balance = 2.0 * np.abs(signed)
    inside = np.abs(signed) <= 0.5
    median_signed = float(np.median(signed))
    inside_ratio = float(np.mean(inside))
    median_balance = float(np.median(balance))
    p90_balance = float(np.percentile(balance, 90))
    offset_mad = float(np.median(np.abs(signed - median_signed)))
    coverage = float(checked.get('coverage', 0.0))
    ranked_multiple = original_selection_reason == 'closest_of_multiple_roadlines'
    locally_confirmed = original_selection_reason == 'local_confirmed_central_roadline'
    minimum_coverage = 0.0 if locally_confirmed else 0.45 if ranked_multiple else float(p.v21_center_min_coverage)
    authoritative_ranked = bool(p.enable_v22_authoritative_roadline_center and ranked_multiple)
    median_limit = float(p.v26_local_center_median_balance_max) if locally_confirmed else float(p.v22_ranked_center_median_balance_max) if authoritative_ranked else float(p.v21_center_median_balance_max)
    p90_limit = float(p.v26_local_center_p90_balance_max) if locally_confirmed else float(p.v22_ranked_center_p90_balance_max) if authoritative_ranked else float(p.v21_center_p90_balance_max)
    passed = bool(coverage >= minimum_coverage and inside_ratio >= float(p.v21_centerline_bilateral_min_inside_ratio) and (median_balance <= median_limit) and (p90_balance <= p90_limit) and (offset_mad <= float(p.v21_center_offset_mad_max)))
    diagnostics.update({'passed': passed, 'reason': 'v26_local_roadline_authoritative' if passed and locally_confirmed else 'v22_ranked_roadline_authoritative' if passed and authoritative_ranked else 'bilaterally_balanced' if passed else 'balance_threshold_failed', 'sample_count': int(len(widths)), 'inside_ratio': inside_ratio, 'median_balance': median_balance, 'p90_balance': p90_balance, 'signed_offset_mad': offset_mad, 'coverage': coverage, 'ranked_multiple_roadlines': ranked_multiple, 'locally_confirmed_roadline': locally_confirmed, 'v22_authoritative_ranked_candidate': authoritative_ranked, 'median_balance_limit': median_limit, 'p90_balance_limit': p90_limit})
    if not passed:
        checked['role'] = 'shape_shifted'
        checked['selection_reason'] = 'bilateral_center_validation_failed'
    return (checked, diagnostics)

# 【品質工具】計算中心線局部轉角的第 95 百分位，偵測急折。
def _v21_center_turn_p95(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    steps = np.diff(points, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    valid = (norms[:-1] > 1e-06) & (norms[1:] > 1e-06)
    if not np.any(valid):
        return 0.0
    first = steps[:-1][valid] / norms[:-1][valid, None]
    second = steps[1:][valid] / norms[1:][valid, None]
    turns = np.degrees(np.arccos(np.clip(np.sum(first * second, axis=1), -1.0, 1.0)))
    return float(np.percentile(turns, 95)) if len(turns) else 0.0

# 【品質工具】檢查候選中心是否位於走廊內、順序合理且沒有過度偏移。
def _v21_center_safety(candidate: np.ndarray, base: np.ndarray, corridor_mask: np.ndarray, p: Parameters) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    base_length = max(polyline_length(base), 1e-06)
    length_ratio = float(polyline_length(candidate) / base_length)
    turn_p95 = _v21_center_turn_p95(candidate)
    xs = np.rint(candidate[:, 0]).astype(np.int32)
    ys = np.rint(candidate[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < corridor_mask.shape[1]) & (ys >= 0) & (ys < corridor_mask.shape[0])
    inside = np.zeros(len(candidate), dtype=bool)
    inside[valid] = corridor_mask[ys[valid], xs[valid]]
    inside_ratio = float(np.mean(inside)) if len(inside) else 0.0
    overall = base[-1] - base[0]
    overall /= max(float(np.linalg.norm(overall)), 1e-06)
    progress = np.diff(candidate, axis=0) @ overall
    backtrack_ratio = float(np.mean(progress < -0.001)) if len(progress) else 0.0
    maximum_turn = max(float(p.v21_center_max_turn_p95_deg), _v21_center_turn_p95(base) + 8.0)
    passed = bool(length_ratio <= float(p.v21_center_max_length_ratio) and turn_p95 <= maximum_turn and (inside_ratio >= float(p.v21_center_min_inside_ratio)) and (backtrack_ratio <= float(p.v21_center_max_backtrack_ratio)))
    return {'passed': passed, 'length_ratio': length_ratio, 'turn_p95_deg': turn_p95, 'turn_limit_deg': maximum_turn, 'inside_ratio': inside_ratio, 'backtrack_ratio': backtrack_ratio}

# 【中心工具】把確認的中央 Roadline 映射成基礎中心站點的權威目標。
def _v22_authoritative_roadline_targets(curve: np.ndarray, stations: np.ndarray, base_centers: np.ndarray, band_source_types: list[str] | None, p: Parameters) -> np.ndarray:
    order = np.argsort(stations)
    curve = np.asarray(curve, dtype=np.float64)[order, :2]
    stations = np.asarray(stations, dtype=np.float64)[order]
    base_centers = np.asarray(base_centers, dtype=np.float64)
    count = len(base_centers)
    unique_stations = np.unique(stations)
    if len(unique_stations) != len(stations):
        curve = np.asarray([np.median(curve[stations == value], axis=0) for value in unique_stations])
        stations = unique_stations
    indices = np.arange(count, dtype=np.float64)
    measured = np.column_stack([np.interp(indices, stations, curve[:, coordinate]) for coordinate in range(2)])
    targets = base_centers.copy()
    in_span = (indices >= stations[0]) & (indices <= stations[-1])
    targets[in_span] = measured[in_span]
    fit_count = min(len(stations), max(2, int(p.v22_authoritative_tangent_fit_points)))

    def endpoint_slope(sample_stations: np.ndarray, points: np.ndarray) -> np.ndarray:
        centered = sample_stations - float(np.mean(sample_stations))
        denominator = float(centered @ centered)
        if denominator <= 1e-09:
            delta_station = max(1e-06, float(sample_stations[-1] - sample_stations[0]))
            return (points[-1] - points[0]) / delta_station
        return (centered[:, None] * points).sum(axis=0) / denominator
    start_slope = endpoint_slope(stations[:fit_count], curve[:fit_count])
    end_slope = endpoint_slope(stations[-fit_count:], curve[-fit_count:])
    del band_source_types
    start_index = int(np.clip(math.ceil(stations[0]), 0, count - 1))
    end_index = int(np.clip(math.floor(stations[-1]), 0, count - 1))
    base_derivatives = np.gradient(base_centers, axis=0)

    def hermite_segment(point_0: np.ndarray, tangent_0: np.ndarray, point_1: np.ndarray, tangent_1: np.ndarray, steps: int) -> np.ndarray:
        parameter = np.linspace(0.0, 1.0, steps + 1)
        parameter_2 = parameter * parameter
        parameter_3 = parameter_2 * parameter
        h00 = 2.0 * parameter_3 - 3.0 * parameter_2 + 1.0
        h10 = parameter_3 - 2.0 * parameter_2 + parameter
        h01 = -2.0 * parameter_3 + 3.0 * parameter_2
        h11 = parameter_3 - parameter_2
        scale = float(max(1, steps))
        return h00[:, None] * point_0 + h10[:, None] * tangent_0 * scale + h01[:, None] * point_1 + h11[:, None] * tangent_1 * scale
    for side in (-1, 1):
        endpoint = start_index if side < 0 else end_index
        roadline_point = measured[endpoint]
        roadline_slope = np.asarray(start_slope if side < 0 else end_slope, dtype=np.float64)
        nominal_transition = max(3, int(p.v21_center_transition_bands))
        local_lo = max(0, endpoint - nominal_transition)
        local_hi = min(count, endpoint + nominal_transition + 1)
        local_speed = float(np.median(np.linalg.norm(base_derivatives[local_lo:local_hi], axis=1)))
        if not np.isfinite(local_speed) or local_speed <= 1e-06:
            local_speed = max(1e-06, float(np.linalg.norm(base_centers[-1] - base_centers[0])) / max(1, count - 1))
        slope_speed = float(np.linalg.norm(roadline_slope))
        maximum_speed = max(1e-06, 2.0 * local_speed)
        if slope_speed > maximum_speed:
            roadline_slope *= maximum_speed / slope_speed
        if float(roadline_slope @ base_derivatives[endpoint]) < 0.0:
            roadline_slope *= -1.0
        lateral_correction = float(np.linalg.norm(roadline_point - base_centers[endpoint]))
        correction_transition = int(math.ceil(lateral_correction / max(1.0, 0.35 * local_speed)))
        available = endpoint if side < 0 else count - 1 - endpoint
        transition = min(available, max(nominal_transition, min(40, correction_transition)))
        outer = int(endpoint + side * transition)
        if outer == endpoint:
            continue
        base_slope = np.asarray(base_derivatives[outer], dtype=np.float64)
        base_speed = float(np.linalg.norm(base_slope))
        if base_speed > maximum_speed:
            base_slope *= maximum_speed / base_speed
        if side < 0:
            bridge = np.arange(outer, endpoint + 1, dtype=np.int32)
            targets[bridge] = hermite_segment(base_centers[outer], base_slope, roadline_point, roadline_slope, endpoint - outer)
        else:
            bridge = np.arange(endpoint, outer + 1, dtype=np.int32)
            targets[bridge] = hermite_segment(roadline_point, roadline_slope, base_centers[outer], base_slope, outer - endpoint)
    return targets

# 【導航工具】保留輔助曲線形狀，將其以平滑偏移移到道路內部。
def _v26_shape_from_parallel_curve(base_centers: np.ndarray, curve: np.ndarray, stations: np.ndarray, core: np.ndarray) -> np.ndarray:
    base = np.asarray(base_centers, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    targets = base.copy()
    core_indices = np.flatnonzero(core)
    if len(core_indices) < 2 or len(curve) < 2:
        return targets
    start = int(core_indices[0])
    end = int(core_indices[-1])
    if end - start < 2:
        return targets
    query = np.arange(len(base), dtype=np.float64)
    auxiliary = np.column_stack([np.interp(query, stations, curve[:, coordinate]) for coordinate in range(2)])
    tangent = np.gradient(auxiliary, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-06)
    base_delta = np.gradient(base, axis=0)
    base_unit = base_delta / np.maximum(np.linalg.norm(base_delta, axis=1)[:, None], 1e-06)
    for index in range(start, end + 1):
        if float(tangent[index] @ base_unit[index]) < 0.0:
            tangent[index] *= -1.0
    tangent_angles = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    tangent_angles = smooth_scalar_series(tangent_angles, 7)
    tangent = np.column_stack((np.cos(tangent_angles), np.sin(tangent_angles)))
    auxiliary_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    raw_offset = np.sum((base - auxiliary) * auxiliary_normal, axis=1)
    span = end - start + 1
    smoothing_window = max(7, min(21, 2 * (span // 12) + 1))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    local_offset = smooth_scalar_series(raw_offset[start:end + 1], smoothing_window)
    parallel = auxiliary[start:end + 1] + local_offset[:, None] * auxiliary_normal[start:end + 1]
    weight = np.ones(span, dtype=np.float64)
    splice = min(10, max(3, span // 6))
    ramp = np.linspace(0.0, 1.0, splice + 1)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    weight[:splice + 1] = np.minimum(weight[:splice + 1], ramp)
    weight[-(splice + 1):] = np.minimum(weight[-(splice + 1):], ramp[::-1])
    targets[start:end + 1] = base[start:end + 1] + weight[:, None] * (parallel - base[start:end + 1])
    return targets

# 【導航工具】依輔助物角色建立中心位置候選及可信覆蓋區。
def _v21_guided_center_candidate(base_centers: np.ndarray, base_left: np.ndarray, base_right: np.ndarray, reference: dict, role: str, p: Parameters, band_source_types: list[str] | None=None) -> tuple[np.ndarray, set[int], np.ndarray]:
    curve, stations, observed = _v21_reference_samples(reference, len(base_centers))
    guided_mask = np.zeros(len(base_centers), dtype=bool)
    if len(curve) < 2 or len(observed) < 2:
        return (base_centers.copy(), set(), guided_mask)
    widths = np.linalg.norm(base_right - base_left, axis=1)
    indices = np.arange(len(base_centers), dtype=np.float64)
    in_span = (indices >= stations.min()) & (indices <= stations.max())
    curve_targets = np.column_stack([np.interp(indices, stations, curve[:, coordinate]) for coordinate in range(2)])
    core_radius = max(0, int(p.v18_curve_support_transition_bands))
    observed_distance = np.min(np.abs(indices[:, None] - observed[None, :]), axis=1)
    core = in_span & (observed_distance <= float(core_radius))
    if int(core.sum()) < 2:
        return (base_centers.copy(), set(), guided_mask)
    authoritative = bool(role == 'confirmed_centerline' and str(reference.get('source_family', '')) == 'road_line' and p.enable_v22_authoritative_roadline_center)
    if role == 'shape_shifted':
        targets = _v26_shape_from_parallel_curve(base_centers, curve, stations, core)
        blend = float(np.clip(p.v18_shape_center_blend, 0.0, 1.0))
        base_tangents = np.gradient(base_centers, axis=0)
        tangent_norms = np.linalg.norm(base_tangents, axis=1)
        base_tangents /= np.maximum(tangent_norms[:, None], 1e-06)
        shape_normals = np.column_stack((-base_tangents[:, 1], base_tangents[:, 0]))
    else:
        targets = _v22_authoritative_roadline_targets(curve, stations, base_centers, band_source_types, p) if authoritative else curve_targets
        blend = 1.0 if authoritative else float(np.clip(p.v14_roadline_direct_center_blend, 0.0, 1.0))
        shape_normals = None
    if authoritative:
        strength = np.full(len(base_centers), blend, dtype=np.float64)
    else:
        core_indices = np.flatnonzero(core)
        transition = max(0, int(p.v21_center_transition_bands))
        distance_to_core = np.min(np.abs(np.arange(len(base_centers))[:, None] - core_indices[None, :]), axis=1)
        strength = np.clip(1.0 - distance_to_core / max(1.0, transition + 1.0), 0.0, 1.0)
        strength[~((indices >= stations.min() - transition) & (indices <= stations.max() + transition))] = 0.0
        strength *= blend
    candidate = base_centers.copy()
    for index in np.flatnonzero(strength > 0.0):
        delta = targets[index] - base_centers[index]
        if shape_normals is not None:
            delta = shape_normals[index] * float(delta @ shape_normals[index])
        if not authoritative:
            maximum_shift = max(float(p.min_road_width), float(widths[index])) * float(p.v21_center_max_shift_width_fraction)
            magnitude = float(np.linalg.norm(delta))
            if magnitude > maximum_shift:
                delta *= maximum_shift / max(magnitude, 1e-06)
        candidate[index] += strength[index] * delta
        guided_mask[index] = True
    if authoritative:
        pairs = base_right - base_left
        squared_widths = np.sum(pairs * pairs, axis=1)
        valid_width = squared_widths > 1e-09
        fractions = np.full(len(candidate), 0.5, dtype=np.float64)
        fractions[valid_width] = np.sum((candidate[valid_width] - base_left[valid_width]) * pairs[valid_width], axis=1) / squared_widths[valid_width]
        margin = float(np.clip(p.v22_authoritative_boundary_margin_fraction, 0.0, 0.49))
        fractions = np.clip(fractions, margin, 1.0 - margin)
        candidate[valid_width] = base_left[valid_width] + fractions[valid_width, None] * pairs[valid_width]
    return (candidate, set(np.flatnonzero(guided_mask).tolist()), guided_mask)

# 【導航工具】將選中參考曲線的方向或位置寫入道路中心幾何。
def _v20_apply_reference_direction(centers: np.ndarray, left: np.ndarray, right: np.ndarray, reference: dict | None, global_angle_degrees: float, p: Parameters, band_source_types: list[str] | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int], np.ndarray, dict | None, dict]:
    del global_angle_degrees
    base_centers = np.asarray(centers, dtype=np.float64).copy()
    base_left = np.asarray(left, dtype=np.float64).copy()
    base_right = np.asarray(right, dtype=np.float64).copy()
    base_mask = corridor_polygon_mask((int(max(np.max(base_left[:, 1]), np.max(base_right[:, 1])) + 3), int(max(np.max(base_left[:, 0]), np.max(base_right[:, 0])) + 3)), base_left, base_right)
    checked_reference, role_diagnostics = _v21_validate_reference_role(reference, base_centers, base_left, base_right, p)
    empty_mask = np.zeros(len(base_centers), dtype=bool)
    if checked_reference is None:
        return (base_centers, base_left, base_right, set(), empty_mask, None, {'status': 'v12_fallback', 'role_validation': role_diagnostics})
    if str(role_diagnostics.get('original_selection_reason', '')) == 'local_confirmed_central_roadline' and (not bool(role_diagnostics.get('passed', False))):
        return (base_centers, base_left, base_right, set(), empty_mask, None, {'status': 'ambiguous_local_roadline_v12_fallback', 'role_validation': role_diagnostics})
    requested_role = str(checked_reference.get('role', 'shape_shifted'))
    if bool(p.enable_v31_multi_aux_pretrack_direction) and bool(p.v31_disable_shape_center_shift) and (requested_role == 'shape_shifted'):
        return (base_centers, base_left, base_right, set(), empty_mask, checked_reference, {'status': 'v31_shape_direction_only', 'role_validation': role_diagnostics, 'applied_role': 'direction_only'})
    attempts = [requested_role]
    if requested_role == 'confirmed_centerline':
        attempts.append('shape_shifted')
    attempt_diagnostics: list[dict] = []
    for role in attempts:
        candidate, guided_indices, guided_mask = _v21_guided_center_candidate(base_centers, base_left, base_right, checked_reference, role, p, band_source_types)
        safety = _v21_center_safety(candidate, base_centers, base_mask, p)
        safety['role'] = role
        safety['guided_band_count'] = int(len(guided_indices))
        attempt_diagnostics.append(safety)
        if guided_indices and safety['passed']:
            applied = dict(checked_reference)
            applied['role'] = role
            if role != requested_role:
                applied['selection_reason'] = 'center_safety_downgraded_to_shape'
            return (candidate, base_left, base_right, guided_indices, guided_mask, applied, {'status': 'auxiliary_navigation_center_applied', 'role_validation': role_diagnostics, 'attempts': attempt_diagnostics, 'applied_role': role})
    return (base_centers, base_left, base_right, set(), empty_mask, None, {'status': 'unsafe_auxiliary_center_v12_fallback', 'role_validation': role_diagnostics, 'attempts': attempt_diagnostics, 'applied_role': 'v12_fallback'})

# 【方向工具】以長視窗估計穩定 Road 前進方向，避免局部邊界噪聲。
def _v31_long_window_road_tangents(centers: np.ndarray, window_bands: int) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    count = len(centers)
    if count < 2:
        return np.empty((0, 2), dtype=np.float64)
    radius = max(2, int(window_bands))
    tangents = np.empty((count, 2), dtype=np.float64)
    for index in range(count):
        lo = max(0, index - radius)
        hi = min(count, index + radius + 1)
        points = centers[lo:hi]
        centered = points - points.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, int(np.argmax(values))]
        forward = centers[min(count - 1, index + 1)] - centers[max(0, index - 1)]
        if float(tangent @ forward) < 0.0:
            tangent *= -1.0
        norm = float(np.linalg.norm(tangent))
        tangents[index] = tangent / max(norm, 1e-06)
    for index in range(1, count):
        if float(tangents[index] @ tangents[index - 1]) < 0.0:
            tangents[index] *= -1.0
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    angles = smooth_scalar_series(angles, max(5, 2 * (radius // 2) + 1))
    return np.column_stack((np.cos(angles), np.sin(angles)))

# 【方向工具】將單一輔助曲線轉成可融合的站點方向與可信權重。
def _v31_auxiliary_direction_samples(record: dict, centers: np.ndarray, road_tangents: np.ndarray, p: Parameters) -> dict | None:
    points = np.asarray(record.get('points', []), dtype=np.float64)
    stations = np.asarray(record.get('station_indices', []), dtype=np.int32)
    ridge = np.asarray(record.get('curve_points', []), dtype=np.float64)
    intrinsic_curve = np.empty((0, 2), dtype=np.float64)
    intrinsic_tangents = np.empty((0, 2), dtype=np.float64)
    if ridge.ndim == 2 and ridge.shape[1] >= 2 and (len(ridge) >= 4):
        raw_curve = ridge[:, :2]
        finite = np.all(np.isfinite(raw_curve), axis=1)
        raw_curve = raw_curve[finite]
        if len(raw_curve) >= 4:
            curve_parameter = np.arange(len(raw_curve), dtype=np.float64)
            intrinsic_weights = np.ones(len(raw_curve), dtype=np.float64)
            intrinsic_curve = np.column_stack([regularized_curve_fit(raw_curve[:, coordinate], intrinsic_weights, max(18.0, 2.0 * float(p.v16_curve_smoothness)), int(p.v16_curve_robust_iterations)) for coordinate in range(2)])
            derivative = np.gradient(intrinsic_curve, axis=0)
            norms = np.linalg.norm(derivative, axis=1)
            derivative /= np.maximum(norms[:, None], 1e-06)
            intrinsic_angles = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0]))
            intrinsic_angles = smooth_scalar_series(intrinsic_angles, max(7, 2 * int(p.v31_local_support_radius_bands) + 3))
            intrinsic_tangents = np.column_stack((np.cos(intrinsic_angles), np.sin(intrinsic_angles)))
    count = len(centers)
    valid = points.ndim == 2 and points.shape[1] >= 2 and (len(points) == len(stations))
    if not valid:
        return None
    keep = (stations >= 0) & (stations < count) & np.all(np.isfinite(points[:, :2]), axis=1)
    points, stations = (points[keep, :2], stations[keep])
    unique = np.unique(stations)
    if len(unique) < 4:
        return None
    observed = np.asarray([np.median(points[stations == station], axis=0) for station in unique], dtype=np.float64)
    positions = np.arange(int(unique[0]), int(unique[-1]) + 1, dtype=np.int32)
    weights = np.zeros(len(positions), dtype=np.float64)
    weights[unique - positions[0]] = 1.0
    curve = np.column_stack([regularized_curve_fit(np.interp(positions, unique, observed[:, coordinate]), weights, max(5.0, float(p.v16_curve_smoothness)), int(p.v16_curve_robust_iterations)) for coordinate in range(2)])
    if len(intrinsic_tangents) == len(intrinsic_curve) and len(intrinsic_curve):
        observed_angles: list[float] = []
        previous_angle: float | None = None
        for station, observed_point in zip(unique, observed):
            curve_index = int(np.argmin(np.linalg.norm(intrinsic_curve - observed_point, axis=1)))
            tangent = intrinsic_tangents[curve_index].copy()
            norm = float(np.linalg.norm(tangent))
            if norm <= 1e-06:
                tangent = road_tangents[station]
            else:
                tangent /= norm
            angle = float(math.atan2(tangent[1], tangent[0]))
            if previous_angle is not None:
                angle += round((previous_angle - angle) / (2.0 * math.pi)) * (2.0 * math.pi)
            observed_angles.append(angle)
            previous_angle = angle
        local_angles = np.interp(positions, unique, np.asarray(observed_angles, dtype=np.float64))
    else:
        derivative = np.gradient(curve, axis=0)
        norms = np.linalg.norm(derivative, axis=1)
        if int(np.sum(norms > 1e-06)) < 3:
            return None
        derivative /= np.maximum(norms[:, None], 1e-06)
        local_angles = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0]))
    local_angles = smooth_scalar_series(local_angles, max(7, 2 * int(p.v31_local_support_radius_bands) + 3))
    derivative = np.column_stack((np.cos(local_angles), np.sin(local_angles)))
    for local_index, station in enumerate(positions):
        if float(derivative[local_index] @ road_tangents[station]) < 0.0:
            derivative[local_index] *= -1.0
    support = np.zeros(len(positions), dtype=bool)
    radius = max(0, int(p.v31_local_support_radius_bands))
    for station in unique:
        lo = max(int(positions[0]), int(station) - radius)
        hi = min(int(positions[-1]), int(station) + radius)
        support[lo - positions[0]:hi - positions[0] + 1] = True
    max_gap = max(1, int(p.v30_auxiliary_consensus_station_gap))
    for first, second in zip(unique[:-1], unique[1:]):
        if int(second - first) <= max_gap:
            support[int(first - positions[0]):int(second - positions[0]) + 1] = True
    family = str(record.get('source_family', ''))
    category = str(record.get('source_category', record.get('category', family)))
    family_weight = 1.0 if family == 'road_line' else 0.72
    if family == 'cone_sequence' or category == 'cone':
        family_weight = 0.3
    quality = float(np.clip(family_weight * max(0.25, float(record.get('angle_consistency', 0.0))) * max(0.35, float(record.get('associated_fraction', 0.0))), 0.0, 1.0))
    return {'positions': positions, 'observed_station_indices': unique, 'curve': curve, 'intrinsic_curve': intrinsic_curve, 'tangents': derivative, 'support': support, 'weight': quality, 'source_family': family, 'source_category': category}

# 【導航函式】融合多條一致輔助曲線，建立分段平滑的最終方向場。
def _v31_build_navigation_direction_field(records: list[dict], centers: np.ndarray, global_angle_degrees: float, p: Parameters) -> tuple[np.ndarray, np.ndarray, dict]:
    centers = np.asarray(centers, dtype=np.float64)
    count = len(centers)
    road_tangents = _v31_long_window_road_tangents(centers, int(p.v31_road_direction_window_bands))
    if count < 2 or len(road_tangents) != count:
        radians = math.radians(float(global_angle_degrees))
        fallback = np.tile(np.asarray([math.cos(radians), math.sin(radians)]), (count, 1))
        return (fallback, np.zeros(count, dtype=bool), {'accepted_curve_count': 0, 'supported_bands': 0, 'conflict_bands': 0})
    samples = [sample for sample in (_v31_auxiliary_direction_samples(record, centers, road_tangents, p) for record in records) if sample is not None]
    target_angles = np.arctan2(road_tangents[:, 1], road_tangents[:, 0])
    strength = np.zeros(count, dtype=np.float64)
    support_mask = np.zeros(count, dtype=bool)
    conflict_bands = 0
    source_diversity_by_band = np.zeros(count, dtype=np.int32)
    for station in range(count):
        observations: list[dict] = []
        base_angle_deg = float(math.degrees(target_angles[station]) % 180.0)
        for sample_index, sample in enumerate(samples):
            positions = sample['positions']
            if station < int(positions[0]) or station > int(positions[-1]):
                continue
            local_index = station - int(positions[0])
            if not bool(sample['support'][local_index]):
                continue
            tangent = np.asarray(sample['tangents'][local_index], dtype=np.float64)
            if float(tangent @ road_tangents[station]) < 0.0:
                tangent *= -1.0
            angle_deg = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 180.0)
            observations.append({'sample_index': sample_index, 'angle_deg': angle_deg, 'tangent': tangent, 'weight': float(sample['weight']), 'family': sample['source_family'], 'category': sample['source_category']})
        if not observations:
            continue
        groups: list[list[dict]] = []
        for observation in sorted(observations, key=lambda item: item['weight'], reverse=True):
            placed = False
            for group in groups:
                mean_angle, _ = weighted_axial_mean_degrees([item['angle_deg'] for item in group], [item['weight'] for item in group])
                if angular_distance_degrees(observation['angle_deg'], mean_angle) <= float(p.v31_consensus_angle_deg):
                    group.append(observation)
                    placed = True
                    break
            if not placed:
                groups.append([observation])
        scored_groups: list[tuple[float, float, list[dict]]] = []
        for group in groups:
            categories = {item['category'] for item in group}
            families = {item['family'] for item in group}
            mean_angle, consistency = weighted_axial_mean_degrees([item['angle_deg'] for item in group], [item['weight'] for item in group])
            category_weights: dict[str, float] = {}
            for item in group:
                category_weights[item['category']] = max(category_weights.get(item['category'], 0.0), float(item['weight']))
            total_weight = float(sum(category_weights.values()))
            has_roadline = 'road_line' in families
            reliable = bool(consistency >= 0.8 and total_weight >= float(p.v31_local_min_consensus_weight) and (has_roadline or len(categories) >= 2))
            if not reliable:
                continue
            road_agreement = max(0.0, 1.0 - angular_distance_degrees(mean_angle, base_angle_deg) / max(float(p.v31_conflict_angle_deg), 1.0))
            score = total_weight * (0.75 + 0.25 * consistency) + 0.2 * road_agreement
            scored_groups.append((score, mean_angle, group))
        if not scored_groups:
            continue
        scored_groups.sort(key=lambda item: item[0], reverse=True)
        if len(scored_groups) > 1 and scored_groups[0][0] < 1.2 * scored_groups[1][0] and (angular_distance_degrees(scored_groups[0][1], scored_groups[1][1]) >= float(p.v31_conflict_angle_deg)):
            conflict_bands += 1
            continue
        score, mean_angle, group = scored_groups[0]
        tangent = np.asarray([math.cos(math.radians(mean_angle)), math.sin(math.radians(mean_angle))], dtype=np.float64)
        if float(tangent @ road_tangents[station]) < 0.0:
            tangent *= -1.0
        target_angles[station] = math.atan2(tangent[1], tangent[0])
        strength[station] = float(np.clip(score, 0.0, 1.0))
        support_mask[station] = True
        source_diversity_by_band[station] = len({item['category'] for item in group})
    if np.any(support_mask):
        base_angles = np.unwrap(np.arctan2(road_tangents[:, 1], road_tangents[:, 0]))
        target_angles = np.unwrap(target_angles)
        target_angles += np.round((base_angles - target_angles) / math.pi) * math.pi
        transition = max(3, int(p.v31_local_transition_bands))
        blend = smooth_scalar_series(strength, 2 * transition + 1)
        blend = np.clip(blend, 0.0, 1.0)
        fused = base_angles + blend * (target_angles - base_angles)
        fused = smooth_scalar_series(fused, max(5, transition // 2 * 2 + 1))
    else:
        fused = np.unwrap(np.arctan2(road_tangents[:, 1], road_tangents[:, 0]))
    maximum_delta = math.radians(float(p.v31_local_max_adjacent_delta_deg))
    for index in range(1, count):
        fused[index] = fused[index - 1] + float(np.clip(fused[index] - fused[index - 1], -maximum_delta, maximum_delta))
    tangents = np.column_stack((np.cos(fused), np.sin(fused)))
    return (tangents, support_mask, {'accepted_curve_count': int(len(samples)), 'supported_bands': int(support_mask.sum()), 'conflict_bands': int(conflict_bands), 'maximum_source_diversity': int(np.max(source_diversity_by_band) if len(source_diversity_by_band) else 0), 'mean_source_diversity_on_support': float(source_diversity_by_band[support_mask].mean() if np.any(support_mask) else 0.0)})

# 【導航函式】以 Road 作位置錨點、輔助曲線作方向，建立權威導航軸。
def _v32_authoritative_navigation_axis(records: list[dict], centers: np.ndarray, left: np.ndarray, right: np.ndarray, source_types: list[str], global_angle_degrees: float, corridor_mask: np.ndarray, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    centers = np.asarray(centers, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    count = min(len(centers), len(left), len(right), len(source_types))
    if count < 2:
        empty = np.zeros(count, dtype=bool)
        return (centers[:count].copy(), np.empty((0, 2)), empty, [], {'status': 'too_few_road_positions', 'authoritative_bands': 0, 'conflict_bands': 0, 'accepted_curve_count': 0})
    centers, left, right = (centers[:count], left[:count], right[:count])
    sources = list(source_types[:count])
    widths = np.linalg.norm(right - left, axis=1)
    road_tangents = _v31_long_window_road_tangents(centers, int(p.v32_road_fallback_window_bands))
    if len(road_tangents) != count:
        radians = math.radians(float(global_angle_degrees))
        road_tangents = np.tile(np.asarray([math.cos(radians), math.sin(radians)]), (count, 1))
    samples: list[dict] = []
    for record_index, record in enumerate(records):
        sample = _v31_auxiliary_direction_samples(record, centers, road_tangents, p)
        if sample is None:
            continue
        positions = np.asarray(sample['positions'], dtype=np.int32)
        observed_stations = np.unique(np.asarray(sample.get('observed_station_indices', record.get('station_indices', [])), dtype=np.int32))
        observed_stations = observed_stations[(observed_stations >= 0) & (observed_stations < count)]
        station_count = int(len(observed_stations))
        coverage = float(station_count / max(1, count))
        family = str(sample['source_family'])
        category = str(sample['source_category'])
        consistency = float(record.get('angle_consistency', 0.0))
        if family == 'road_line' or category == 'road_line':
            single_authoritative = bool(station_count >= int(p.v32_single_roadline_min_stations) or coverage >= float(p.v32_single_roadline_min_coverage))
        elif family == 'roadside_boundary' and category != 'cone':
            single_authoritative = bool(station_count >= int(p.v32_single_side_min_stations) or coverage >= float(p.v32_single_side_min_coverage))
        else:
            single_authoritative = False
        single_authoritative = bool(single_authoritative and consistency >= float(p.v32_min_angle_consistency))
        if p.enable_v33_physical_roadline_tracks and str(record.get('v33_role', 'D')) == 'C':
            single_authoritative = True
        station_indices = np.asarray(record.get('station_indices', []), dtype=np.int32)
        offsets = np.asarray(record.get('signed_offsets', []), dtype=np.float64)
        valid_offsets = (station_indices >= 0) & (station_indices < count) & np.isfinite(offsets) if len(station_indices) == len(offsets) else np.zeros(0, dtype=bool)
        normalized_offset = float('inf')
        if len(valid_offsets) and np.any(valid_offsets):
            normalized_offset = float(np.median(offsets[valid_offsets] / np.maximum(widths[station_indices[valid_offsets]], 1.0)))
        sample_tangents = np.asarray(sample.get('tangents', []), dtype=np.float64)
        sample_support = np.asarray(sample.get('support', []), dtype=bool)
        if sample_tangents.ndim == 2 and sample_tangents.shape[1] >= 2 and (len(sample_tangents) == len(sample_support)) and np.any(sample_support):
            sample_angles = np.degrees(np.arctan2(sample_tangents[sample_support, 1], sample_tangents[sample_support, 0])) % 180.0
            sample_global_angle, _ = weighted_axial_mean_degrees(sample_angles.tolist(), np.ones(len(sample_angles), dtype=np.float64).tolist())
        else:
            sample_global_angle = float(record.get('angle_degrees', 0.0))
        sample.update({'record_index': int(record_index), 'observed_station_count': station_count, 'coverage': coverage, 'angle_consistency': consistency, 'global_angle': float(sample_global_angle), 'single_authoritative': single_authoritative, 'central_roadline': bool(p.enable_v33_physical_roadline_tracks and str(record.get('v33_role', 'D')) == 'C'), 'normalized_offset': normalized_offset, 'v33_track_id': str(record.get('v33_track_id', f'Auxiliary Track {record_index + 1}')), 'v33_role': str(record.get('v33_role', 'D')), 'v33_vanishing_group': record.get('v33_vanishing_group'), 'v33_center_group': record.get('v33_center_group')})
        vanishing_point = record.get('v33_vanishing_point')
        if p.enable_v33_physical_roadline_tracks and vanishing_point is not None and (str(record.get('v33_role', 'D')) != 'C') and (len(sample_tangents) == len(sample_support)):
            point = np.asarray(vanishing_point, dtype=np.float64)
            vp_tangents = np.asarray(sample_tangents, dtype=np.float64).copy()
            for local_index, station in enumerate(positions):
                if not bool(sample_support[local_index]):
                    continue
                vector = point - centers[int(station)]
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-06:
                    continue
                vector /= norm
                if float(vector @ road_tangents[int(station)]) < 0.0:
                    vector *= -1.0
                vp_tangents[local_index] = vector
            sample['tangents'] = vp_tangents
            supported_angles = np.degrees(np.arctan2(vp_tangents[sample_support, 1], vp_tangents[sample_support, 0])) % 180.0
            if len(supported_angles):
                sample['global_angle'], _ = weighted_axial_mean_degrees(supported_angles.tolist(), np.ones(len(supported_angles), dtype=np.float64).tolist())
        sample['curve_positions'] = np.asarray(sample.get('curve', []), dtype=np.float64)
        samples.append(sample)
    center_candidates = sorted([sample for sample in samples if sample['source_family'] == 'road_line' and np.isfinite(sample['normalized_offset'])], key=lambda sample: abs(float(sample['normalized_offset'])))
    if center_candidates and (not bool(p.enable_v33_physical_roadline_tracks)):
        best_offset = abs(float(center_candidates[0]['normalized_offset']))
        second_offset = abs(float(center_candidates[1]['normalized_offset'])) if len(center_candidates) > 1 else float('inf')
        if best_offset <= 0.12 and second_offset - best_offset >= 0.04:
            center_candidates[0]['central_roadline'] = True
    global_groups: list[list[dict]] = []
    for sample in sorted(samples, key=lambda value: (bool(value['single_authoritative']), float(value['coverage']), float(value['weight'])), reverse=True):
        for group in global_groups:
            shared_vanishing_group = bool(p.enable_v33_physical_roadline_tracks and sample.get('v33_vanishing_group') is not None and any((member.get('v33_vanishing_group') == sample.get('v33_vanishing_group') for member in group)))
            mean_angle, _ = weighted_axial_mean_degrees([item['global_angle'] for item in group], [max(0.05, item['weight']) for item in group])
            if shared_vanishing_group or angular_distance_degrees(sample['global_angle'], mean_angle) <= float(p.v32_consensus_angle_deg):
                group.append(sample)
                break
        else:
            global_groups.append([sample])
    scored_global_groups: list[tuple[float, float, list[dict]]] = []
    for group in global_groups:
        categories = {item['source_category'] for item in group}
        mean_angle, consistency = weighted_axial_mean_degrees([item['global_angle'] for item in group], [max(0.05, item['weight']) for item in group])
        reliable = bool(any((item['single_authoritative'] for item in group)) or len(categories) >= 2 or any((str(item.get('v33_role', 'D')) == 'C' for item in group)))
        has_unified_center = any((str(item.get('v33_role', 'D')) == 'C' for item in group))
        if not reliable or (consistency < float(p.v32_min_angle_consistency) and (not has_unified_center)):
            continue
        category_coverage: dict[str, float] = {}
        category_quality: dict[str, float] = {}
        for item in group:
            category = str(item['source_category'])
            category_coverage[category] = max(category_coverage.get(category, 0.0), float(item['coverage']))
            category_quality[category] = max(category_quality.get(category, 0.0), float(item['weight']))
        union_like_coverage = min(1.0, sum(category_coverage.values()))
        score = 2.0 * union_like_coverage + 0.55 * sum(category_quality.values()) + 0.18 * min(2, max(0, len(categories) - 1)) + (0.1 if any((item['source_family'] == 'road_line' for item in group)) else 0.0)
        scored_global_groups.append((float(score), float(mean_angle), group))
    scored_global_groups.sort(key=lambda item: item[0], reverse=True)
    if bool(p.enable_v33_physical_roadline_tracks):
        central_groups = [item for item in scored_global_groups if any((str(member.get('v33_role', 'D')) == 'C' for member in item[2]))]
        if central_groups:
            chosen = max(central_groups, key=lambda item: item[0])
            scored_global_groups.remove(chosen)
            scored_global_groups.insert(0, chosen)
    allowed_record_indices: set[int] = set()
    global_direction_conflict = False
    if scored_global_groups:
        use_best = True
        if len(scored_global_groups) > 1:
            best, second = scored_global_groups[:2]
            if best[0] < float(p.v32_conflict_margin_ratio) * second[0] and angular_distance_degrees(best[1], second[1]) >= float(p.v32_conflict_angle_deg):
                central_roadline_groups = [item for item in scored_global_groups if any((member['source_family'] == 'road_line' and bool(member['single_authoritative']) and bool(member['central_roadline']) for member in item[2]))]
                roadline_groups = [item for item in scored_global_groups if any((member['source_family'] == 'road_line' and bool(member['single_authoritative']) for member in item[2]))]
                best_categories = {str(member['source_category']) for member in best[2]}
                second_categories = {str(member['source_category']) for member in second[2]}
                best_has_roadline = any((member['source_family'] == 'road_line' for member in best[2]))
                if best_has_roadline and len(best_categories) >= 2 and (len(second_categories) == 1):
                    pass
                elif len(central_roadline_groups) == 1 and central_roadline_groups[0][0] >= 0.45 * best[0]:
                    chosen = central_roadline_groups[0]
                    scored_global_groups.remove(chosen)
                    scored_global_groups.insert(0, chosen)
                elif len(roadline_groups) == 1 and roadline_groups[0][0] >= 0.55 * best[0]:
                    chosen = roadline_groups[0]
                    scored_global_groups.remove(chosen)
                    scored_global_groups.insert(0, chosen)
                else:
                    use_best = False
                    global_direction_conflict = True
        if use_best:
            allowed_record_indices = {int(item['record_index']) for item in scored_global_groups[0][2]}
    base_angles = np.unwrap(np.arctan2(road_tangents[:, 1], road_tangents[:, 0]))
    target_angles = base_angles.copy()
    authoritative = np.zeros(count, dtype=bool)
    conflict_mask = np.zeros(count, dtype=bool)
    winning_records: list[set[int]] = [set() for _ in range(count)]
    central_targets: dict[int, list[np.ndarray]] = {}
    for station in range(count):
        observations: list[dict] = []
        for sample in samples:
            if int(sample['record_index']) not in allowed_record_indices:
                continue
            positions = np.asarray(sample['positions'], dtype=np.int32)
            if station < int(positions[0]) or station > int(positions[-1]):
                continue
            local_index = station - int(positions[0])
            if not bool(sample['support'][local_index]):
                continue
            tangent = np.asarray(sample['tangents'][local_index], dtype=np.float64)
            if float(tangent @ road_tangents[station]) < 0.0:
                tangent *= -1.0
            angle = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 180.0)
            observations.append({'angle': angle, 'tangent': tangent, 'weight': float(sample['weight']), 'family': str(sample['source_family']), 'category': str(sample['source_category']), 'sample': sample, 'local_index': int(local_index)})
        if not observations:
            continue
        groups: list[list[dict]] = []
        for observation in sorted(observations, key=lambda item: item['weight'], reverse=True):
            for group in groups:
                mean_angle, _ = weighted_axial_mean_degrees([item['angle'] for item in group], [item['weight'] for item in group])
                if angular_distance_degrees(observation['angle'], mean_angle) <= float(p.v32_consensus_angle_deg):
                    group.append(observation)
                    break
            else:
                groups.append([observation])
        scored: list[tuple[float, float, list[dict]]] = []
        for group in groups:
            categories = {item['category'] for item in group}
            mean_angle, consistency = weighted_axial_mean_degrees([item['angle'] for item in group], [item['weight'] for item in group])
            category_weights: dict[str, float] = {}
            for item in group:
                category_weights[item['category']] = max(category_weights.get(item['category'], 0.0), float(item['weight']))
            has_long_single = any((bool(item['sample']['single_authoritative']) for item in group))
            has_multi_source = len(categories) >= 2
            has_unified_center = any((str(item['sample'].get('v33_role', 'D')) == 'C' for item in group))
            if not ((consistency >= float(p.v32_min_angle_consistency) or has_unified_center) and (has_long_single or has_multi_source or has_unified_center)):
                continue
            coverage = max((float(item['sample']['coverage']) for item in group))
            source_bonus = 0.18 * min(2, max(0, len(categories) - 1))
            roadline_bonus = 0.15 if any((item['family'] == 'road_line' for item in group)) else 0.0
            score = float(sum(category_weights.values())) * (0.7 + 0.3 * float(consistency)) + 0.45 * coverage + source_bonus + roadline_bonus
            scored.append((score, float(mean_angle), group))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        if bool(p.enable_v33_physical_roadline_tracks):
            central_scored = [item for item in scored if any((str(member['sample'].get('v33_role', 'D')) == 'C' for member in item[2]))]
            if central_scored:
                chosen = max(central_scored, key=lambda item: item[0])
                scored.remove(chosen)
                scored.insert(0, chosen)
        if len(scored) > 1 and scored[0][0] < float(p.v32_conflict_margin_ratio) * scored[1][0] and (angular_distance_degrees(scored[0][1], scored[1][1]) >= float(p.v32_conflict_angle_deg)):
            conflict_mask[station] = True
            continue
        _, mean_angle, group = scored[0]
        tangent = np.asarray([math.cos(math.radians(mean_angle)), math.sin(math.radians(mean_angle))], dtype=np.float64)
        if float(tangent @ road_tangents[station]) < 0.0:
            tangent *= -1.0
        target_angles[station] = math.atan2(tangent[1], tangent[0])
        authoritative[station] = True
        for item in group:
            sample = item['sample']
            winning_records[station].add(int(sample['record_index']))
            if bool(sample['central_roadline']):
                curve = np.asarray(sample.get('curve_positions', []), dtype=np.float64)
                local_index = int(item['local_index'])
                if len(curve) == len(sample['positions']):
                    central_targets.setdefault(station, []).append(curve[local_index])
    core_indices = np.flatnonzero(authoritative)
    maximum_gap = max(0, int(p.v32_small_gap_bands))
    for first, second in zip(core_indices[:-1], core_indices[1:]):
        if second - first <= maximum_gap + 1 and (not np.any(conflict_mask[first:second + 1])):
            start_angle = float(target_angles[first])
            end_angle = float(target_angles[second])
            end_angle += round((start_angle - end_angle) / math.pi) * math.pi
            for station in range(int(first) + 1, int(second)):
                fraction = (station - first) / float(second - first)
                target_angles[station] = (1.0 - fraction) * start_angle + fraction * end_angle
                authoritative[station] = True
                winning_records[station] = winning_records[first] | winning_records[second]
    navigation_owned = authoritative.copy()
    core_indices = np.flatnonzero(authoritative)
    central_selected = bool(p.enable_v33_physical_roadline_tracks and any((str(records[index].get('v33_role', 'D')) == 'C' for index in allowed_record_indices if 0 <= index < len(records))))
    if len(core_indices) and allowed_record_indices and (not central_selected):
        first_core = int(core_indices[0])
        last_core = int(core_indices[-1])
        target_angles[:first_core] = float(target_angles[first_core])
        target_angles[last_core + 1:] = float(target_angles[last_core])
        for first, second in zip(core_indices[:-1], core_indices[1:]):
            if second <= first + 1:
                continue
            start_angle = float(target_angles[first])
            end_angle = float(target_angles[second])
            end_angle += round((start_angle - end_angle) / math.pi) * math.pi
            for station in range(int(first) + 1, int(second)):
                fraction = (station - first) / float(second - first)
                target_angles[station] = (1.0 - fraction) * start_angle + fraction * end_angle
        navigation_owned[:] = True
    elif len(core_indices) and central_selected:
        navigation_owned = authoritative.copy()
    final_angles = base_angles.copy()
    transition = max(0, int(p.v32_transition_bands))
    for start, end in _boolean_true_runs(navigation_owned):
        run = np.asarray(target_angles[start:end], dtype=np.float64).copy()
        if not len(run):
            continue
        run[0] += round((base_angles[start] - run[0]) / math.pi) * math.pi
        for local_index in range(1, len(run)):
            run[local_index] += round((run[local_index - 1] - run[local_index]) / math.pi) * math.pi
        if len(run) >= 5:
            run = smooth_scalar_series(run, 5)
        final_angles[start:end] = run
        lo = max(0, start - transition)
        hi = min(count, end + transition)
        if start > lo:
            target = float(run[0])
            road_start = float(base_angles[lo])
            road_start += round((target - road_start) / (2.0 * math.pi)) * (2.0 * math.pi)
            for station in range(lo, start):
                fraction = (station - lo + 1) / float(start - lo + 1)
                final_angles[station] = (1.0 - fraction) * road_start + fraction * target
        if hi > end:
            target = float(run[-1])
            road_end = float(base_angles[hi - 1])
            road_end += round((target - road_end) / (2.0 * math.pi)) * (2.0 * math.pi)
            for station in range(end, hi):
                fraction = (station - end + 1) / float(hi - end + 1)
                final_angles[station] = (1.0 - fraction) * target + fraction * road_end
    maximum_delta = math.radians(float(p.v32_max_adjacent_delta_deg))
    for _ in range(2):
        adjusted = final_angles.copy()
        for station in range(1, count - 1):
            midpoint = 0.5 * (final_angles[station - 1] + final_angles[station + 1])
            adjusted[station] = midpoint + float(np.clip(final_angles[station] - midpoint, -maximum_delta, maximum_delta))
        final_angles = adjusted
    progress = centers[-1] - centers[0]
    progress_norm = float(np.linalg.norm(progress))
    if progress_norm <= 1e-06:
        progress = np.sum(road_tangents, axis=0)
        progress_norm = float(np.linalg.norm(progress))
    if progress_norm > 1e-06:
        progress /= progress_norm
        progress_angle = math.atan2(progress[1], progress[0])
        maximum_progress_delta = math.radians(80.0)
        for station in range(count):
            angle = float(final_angles[station])
            angle += round((progress_angle - angle) / (2.0 * math.pi)) * (2.0 * math.pi)
            delta = float(angle - progress_angle)
            final_angles[station] = progress_angle + float(np.clip(delta, -maximum_progress_delta, maximum_progress_delta))
    if bool(p.enable_v33_physical_roadline_tracks) and central_selected:
        central_stations = np.asarray(sorted(central_targets), dtype=np.int32)
        central_stations = central_stations[(central_stations >= 0) & (central_stations < count)]
        final_angles[central_stations] = target_angles[central_stations]
    tangents = np.column_stack((np.cos(final_angles), np.sin(final_angles)))
    segment_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = segment_steps[segment_steps > 1e-06]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    segment_steps = np.clip(segment_steps, 0.35 * median_step, 2.5 * median_step)
    anchor = int(count // 2)
    direct_indices = [index for index, source in enumerate(sources) if source == 'direct']
    if direct_indices:
        anchor = min(direct_indices, key=lambda index: abs(index - count // 2))
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    anchor_weights = np.zeros(count, dtype=np.float64)
    anchor_targets = centers.copy()
    stride = max(4, int(p.v32_position_anchor_stride_bands))
    road_anchor_indices = list(range(0, count, stride))
    road_anchor_indices.extend([0, anchor, count - 1])
    authoritative_position_anchors: set[int] = set()
    for start, end in _boolean_true_runs(authoritative):
        authoritative_position_anchors.add(int((start + end - 1) // 2))
    road_anchor_indices.extend(authoritative_position_anchors)
    for station in sorted(set(road_anchor_indices)):
        if sources[station] not in {'direct', 'interpolated'}:
            continue
        anchor_weights[station] = 6.0 if sources[station] == 'direct' else 3.5 if sources[station] == 'interpolated' else 2.5
    for station, targets in central_targets.items():
        target = np.median(np.asarray(targets, dtype=np.float64), axis=0)
        anchor_targets[station] = target
        anchor_weights[station] = float(p.v33_center_position_weight) if p.enable_v33_physical_roadline_tracks else 12.0

    def solve_axis(extra_targets: dict[int, np.ndarray] | None=None) -> np.ndarray:
        rows: list[np.ndarray] = []
        values: list[float] = []

        def append_row(entries: list[tuple[int, float]], value: float, weight: float) -> None:
            row = np.zeros(2 * count, dtype=np.float64)
            scale = math.sqrt(max(float(weight), 1e-09))
            for column, coefficient in entries:
                row[column] = coefficient * scale
            rows.append(row)
            values.append(float(value) * scale)
        for station in range(count - 1):
            direction = tangents[station] + tangents[station + 1]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-06:
                direction = tangents[station]
                norm = float(np.linalg.norm(direction))
            direction = direction / max(norm, 1e-06)
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
            direction_weight = 28.0 if navigation_owned[station] or navigation_owned[station + 1] else 10.0
            append_row([(2 * station, -float(normal[0])), (2 * station + 1, -float(normal[1])), (2 * (station + 1), float(normal[0])), (2 * (station + 1) + 1, float(normal[1]))], 0.0, direction_weight)
            measured_delta = centers[station + 1] - centers[station]
            desired_progress = float(direction @ measured_delta)
            if desired_progress < 0.0:
                desired_progress = float(segment_steps[station])
            append_row([(2 * station, -float(direction[0])), (2 * station + 1, -float(direction[1])), (2 * (station + 1), float(direction[0])), (2 * (station + 1) + 1, float(direction[1]))], desired_progress, 0.85)
        for station in range(1, count - 1):
            append_row([(2 * (station - 1), 1.0), (2 * station, -2.0), (2 * (station + 1), 1.0)], 0.0, 0.8)
            append_row([(2 * (station - 1) + 1, 1.0), (2 * station + 1, -2.0), (2 * (station + 1) + 1, 1.0)], 0.0, 0.8)
        append_row([(2 * anchor, 1.0)], centers[anchor, 0], 5.0)
        append_row([(2 * anchor + 1, 1.0)], centers[anchor, 1], 5.0)
        for station in np.flatnonzero(anchor_weights > 0.0):
            target = anchor_targets[station]
            weight = float(anchor_weights[station])
            append_row([(2 * station, 1.0)], float(target[0]), weight)
            append_row([(2 * station + 1, 1.0)], float(target[1]), weight)
        if extra_targets:
            for station, target in extra_targets.items():
                append_row([(2 * station, 1.0)], float(target[0]), 20.0)
                append_row([(2 * station + 1, 1.0)], float(target[1]), 20.0)
        matrix = np.vstack(rows)
        fitted, *_ = np.linalg.lstsq(matrix, np.asarray(values, dtype=np.float64), rcond=None)
        return fitted.reshape(count, 2)
    axis = solve_axis()
    mask = np.asarray(corridor_mask, dtype=bool)
    height, width = mask.shape
    projection_count = 0
    margin = float(np.clip(p.v32_corridor_projection_margin, 0.0, 0.45))
    nearest_label_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    try:
        import cv2
        _, labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        mask_y, mask_x = np.nonzero(mask)
        mask_labels = labels[mask_y, mask_x]
        maximum_label = int(labels.max())
        label_x = np.full(maximum_label + 1, -1, dtype=np.int32)
        label_y = np.full(maximum_label + 1, -1, dtype=np.int32)
        label_x[mask_labels] = mask_x
        label_y[mask_labels] = mask_y
        nearest_label_map = (labels, label_x, label_y)
    except Exception:
        nearest_label_map = None

    def nearest_corridor_point(point: np.ndarray, station: int) -> np.ndarray:
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        if mask[y, x]:
            return np.asarray([float(x), float(y)], dtype=np.float64)
        if nearest_label_map is not None:
            labels, label_x, label_y = nearest_label_map
            label = int(labels[y, x])
            if 0 < label < len(label_x) and label_x[label] >= 0 and (label_y[label] >= 0):
                return np.asarray([float(label_x[label]), float(label_y[label])], dtype=np.float64)
        span = right[station] - left[station]
        denominator = float(span @ span)
        if denominator <= 1e-06:
            return centers[station].copy()
        fraction = float((point - left[station]) @ span / denominator)
        fraction = float(np.clip(fraction, margin, 1.0 - margin))
        return left[station] + fraction * span
    projected_targets: dict[int, np.ndarray] = {}
    for station in range(count):
        x = int(round(float(axis[station, 0])))
        y = int(round(float(axis[station, 1])))
        inside = 0 <= x < width and 0 <= y < height and bool(mask[y, x])
        if inside:
            continue
        projected_targets[station] = nearest_corridor_point(axis[station], station)
    if projected_targets:
        axis = solve_axis(projected_targets)
    for station in range(count):
        x = int(round(float(axis[station, 0])))
        y = int(round(float(axis[station, 1])))
        inside = 0 <= x < width and 0 <= y < height and bool(mask[y, x])
        if inside:
            continue
        axis[station] = nearest_corridor_point(axis[station], station)
        projection_count += 1
    if bool(p.enable_v33_physical_roadline_tracks) and central_selected:
        for station, targets in central_targets.items():
            if not 0 <= station < count:
                continue
            target = np.median(np.asarray(targets, dtype=np.float64), axis=0)
            x = int(round(float(target[0])))
            y = int(round(float(target[1])))
            if 0 <= x < width and 0 <= y < height and bool(mask[y, x]):
                axis[station] = target
    accepted_record_indices = sorted(set().union(*winning_records)) if any(winning_records) else []
    if bool(p.enable_v33_physical_roadline_tracks):
        accepted_set = set(accepted_record_indices)
        for record_index, record in enumerate(records):
            if record_index in accepted_set:
                record['v33_adoption'] = 'used'
            elif str(record.get('v33_role', 'D')) == 'X':
                record['v33_adoption'] = 'rejected'
            else:
                record['v33_adoption'] = 'candidate'
    authoritative_curves: list[dict] = []
    for record_index in accepted_record_indices:
        record = records[record_index]
        matching_sample = next((item for item in samples if int(item['record_index']) == int(record_index)), None)
        points = np.asarray(record.get('v33_curve', []) if p.enable_v33_physical_roadline_tracks else matching_sample.get('intrinsic_curve', []) if matching_sample is not None else [], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
            points = np.asarray(record.get('curve_points', record.get('points', [])), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
            continue
        if len(points) > 256:
            take = np.linspace(0, len(points) - 1, 256).astype(np.int32)
            points = points[take]
        authoritative_curves.append({'source_family': str(record.get('source_family', 'auxiliary')), 'source_category': str(record.get('source_category', 'auxiliary')), 'points': points[:, :2].tolist(), 'track_id': str(record.get('v33_track_id', f'Auxiliary Track {record_index + 1}')), 'role': str(record.get('v33_role', 'D')), 'adoption': str(record.get('v33_adoption', 'used')), 'record_index': int(record_index)})
    summary = {'status': 'authoritative_auxiliary_direction_applied' if np.any(authoritative) else 'road_long_window_direction_fallback', 'candidate_curve_count': int(len(samples)), 'accepted_curve_count': int(len(accepted_record_indices)), 'accepted_record_indices': accepted_record_indices, 'authoritative_bands': int(authoritative.sum()), 'authoritative_fraction': float(authoritative.mean()), 'conflict_bands': int(conflict_mask.sum()), 'global_direction_conflict': bool(global_direction_conflict), 'central_roadline_anchor_bands': int(len(central_targets)), 'v33_central_track_selected': bool(central_selected), 'v33_role_source': 'unified_physical_track_role' if p.enable_v33_physical_roadline_tracks else 'v32_offset_rule', 'corridor_projection_count': int(projection_count), 'road_position_anchor_count': int(np.sum(anchor_weights > 0.0)), 'candidate_curves': [{'record_index': int(item['record_index']), 'category': str(item['source_category']), 'family': str(item['source_family']), 'angle_deg': float(item['global_angle']), 'coverage': float(item['coverage']), 'observed_stations': int(item['observed_station_count']), 'angle_consistency': float(item['angle_consistency']), 'single_authoritative': bool(item['single_authoritative']), 'central_roadline': bool(item['central_roadline']), 'track_id': str(item.get('v33_track_id', '')), 'role': str(item.get('v33_role', 'D')), 'vanishing_group': item.get('v33_vanishing_group'), 'normalized_offset': float(item['normalized_offset'])} for item in samples], 'global_direction_groups': [{'score': float(score), 'angle_deg': float(angle), 'record_indices': [int(member['record_index']) for member in group], 'categories': sorted({str(member['source_category']) for member in group})} for score, angle, group in scored_global_groups]}
    return (axis, tangents, authoritative, authoritative_curves, summary)

# =============================================================================
# 14. 導航確立後的障礙遮擋道路重建
# =============================================================================
# 依凍結後的導航方向處理側向內凹及道路末端遮擋：障礙物提供支持，
# 左右寬度不得無限制增加，並將新增區域與來源狀態寫回完整道路。
# 【量測工具】計算預測橫斷面落在 Road 或障礙遮罩中的比例。
def _v25_cross_section_mask_fraction(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    length = max(2, int(math.ceil(float(np.linalg.norm(right - left)))) + 1)
    xs = np.rint(np.linspace(left[0], right[0], length)).astype(np.int32)
    ys = np.rint(np.linspace(left[1], right[1], length)).astype(np.int32)
    supported = np.zeros(length, dtype=bool)
    height, width = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            qx = xs + dx
            qy = ys + dy
            valid = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            supported[valid] |= mask[qy[valid], qx[valid]]
    return float(np.mean(supported)) if len(supported) else 0.0

# 【片段配對】沿凍結導航向前搜尋可與末端道路銜接的後段 Road 片段。
def _v25_forward_fragment_pair(road: np.ndarray, obstacles: np.ndarray, core_geometry: np.ndarray, anchor_center: np.ndarray, anchor_left: np.ndarray, anchor_right: np.ndarray, tangent: np.ndarray, step_length: float, p: Parameters, transverse_direction: np.ndarray | None=None, left_distance: float | None=None, right_distance: float | None=None) -> list[dict]:
    anchor_center = np.asarray(anchor_center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-06:
        return []
    tangent = tangent / tangent_norm
    width_vector = np.asarray(anchor_right, dtype=np.float64) - np.asarray(anchor_left, dtype=np.float64)
    anchor_width = float(np.linalg.norm(width_vector))
    if anchor_width < float(p.min_road_width):
        return []
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    if float(normal @ width_vector) < 0.0:
        normal *= -1.0
    supplied_q = np.asarray(transverse_direction if transverse_direction is not None else [], dtype=np.float64)
    if supplied_q.shape == (2,) and float(np.linalg.norm(supplied_q)) > 1e-06:
        supplied_q /= float(np.linalg.norm(supplied_q))
        if float(supplied_q @ width_vector) < 0.0:
            supplied_q *= -1.0
        normal = supplied_q
    left_share = float(left_distance / max(anchor_width, 1e-06) if left_distance is not None else 0.5)
    right_share = float(right_distance / max(anchor_width, 1e-06) if right_distance is not None else 0.5)
    share_sum = max(1e-06, left_share + right_share)
    left_share, right_share = (left_share / share_sum, right_share / share_sum)
    components = connected_component_point_sets(binary_close(road, max(0, int(p.close_radius))), max(24, int(round(anchor_width * max(1.0, step_length)))))
    candidates: list[dict] = []
    height, width = road.shape
    for component in components:
        rounded = np.rint(component).astype(np.int32)
        valid_pixels = (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
        rounded = rounded[valid_pixels]
        points = component[valid_pixels]
        if len(points) < 12:
            continue
        relative = points - anchor_center[None, :]
        if np.any(np.linalg.norm(relative, axis=1) < 0.35 * anchor_width):
            continue
        inside_ratio = float(np.mean(core_geometry[rounded[:, 1], rounded[:, 0]]))
        longitudinal = relative @ tangent
        lateral = relative @ normal
        ahead = longitudinal > max(step_length, anchor_width * 0.2)
        if int(np.sum(ahead)) < 12:
            continue
        longitudinal = longitudinal[ahead]
        lateral = lateral[ahead]
        bins = np.floor(longitudinal / max(1.0, step_length)).astype(int)
        observations: list[dict] = []
        minimum_points = max(5, int(round(anchor_width * 0.1)))
        for band in np.unique(bins):
            selected = bins == band
            if int(np.sum(selected)) < minimum_points:
                continue
            u_values = longitudinal[selected]
            v_values = lateral[selected]
            low, middle, high = np.quantile(v_values, [0.05, 0.5, 0.95])
            observed_width = float(high - low)
            if not anchor_width * 0.25 <= observed_width <= anchor_width * 1.8:
                continue
            u_value = float(np.median(u_values))
            lateral_limit = max(anchor_width * 2.5, u_value * 0.4)
            if abs(float(middle)) > lateral_limit:
                continue
            center = anchor_center + tangent * u_value + normal * float(middle)
            observations.append({'band': int(band), 'center': center, 'width': float(np.clip(observed_width, anchor_width * 0.45, anchor_width * 1.25)), 'u': u_value, 'lateral': float(middle)})
        observations.sort(key=lambda item: item['u'])
        if len(observations) < 3:
            continue
        continuous = [observations[0]]
        for item in observations[1:]:
            if int(item['band']) - int(continuous[-1]['band']) > 3:
                break
            if float(np.linalg.norm(item['center'] - continuous[-1]['center'])) > max(anchor_width * 1.5, step_length * 4.0):
                break
            continuous.append(item)
        if len(continuous) < 3:
            continue
        local_delta = continuous[min(3, len(continuous) - 1)]['center'] - continuous[0]['center']
        local_norm = float(np.linalg.norm(local_delta))
        local_tangent = tangent if local_norm <= 1e-06 else local_delta / local_norm
        if float(local_tangent @ tangent) < 0.0:
            local_tangent *= -1.0
        angle_penalty = 1.0 - float(np.clip(local_tangent @ tangent, -1.0, 1.0))
        first = continuous[0]
        score = float(first['u']) / max(anchor_width, 1.0) + 1.5 * abs(float(first['lateral'])) / max(anchor_width, 1.0) + 4.0 * angle_penalty
        candidates.append({'score': score, 'observations': continuous, 'end_tangent': local_tangent, 'inside_ratio': inside_ratio})
    if not candidates:
        return []
    selected = min(candidates, key=lambda item: item['score'])
    observations = selected['observations']
    target_center = np.asarray(observations[0]['center'], dtype=np.float64)
    target_width = float(observations[0]['width'])
    distance = float(np.linalg.norm(target_center - anchor_center))
    bridge_steps = max(3, int(math.ceil(distance / max(1.0, step_length))))
    end_tangent = np.asarray(selected['end_tangent'], dtype=np.float64)
    parameter = np.linspace(0.0, 1.0, bridge_steps + 1)
    h00 = 2.0 * parameter ** 3 - 3.0 * parameter ** 2 + 1.0
    h10 = parameter ** 3 - 2.0 * parameter ** 2 + parameter
    h01 = -2.0 * parameter ** 3 + 3.0 * parameter ** 2
    h11 = parameter ** 3 - parameter ** 2
    derivative_scale = max(distance * 0.55, step_length * 2.0)
    curve = h00[:, None] * anchor_center + h10[:, None] * tangent * derivative_scale + h01[:, None] * target_center + h11[:, None] * end_tangent * derivative_scale
    widths = np.linspace(anchor_width, target_width, bridge_steps + 1)
    result: list[dict] = []
    obstacle_hits = 0
    threshold = float(p.v24_obstacle_min_cross_section_fraction)
    for index in range(1, len(curve)):
        if index == len(curve) - 1:
            local_direction = end_tangent
        else:
            local_direction = curve[index + 1] - curve[index - 1]
            norm = float(np.linalg.norm(local_direction))
            local_direction = tangent if norm <= 1e-06 else local_direction / norm
        if bool(p.enable_v37_paired_cross_sections) and supplied_q.shape == (2,):
            local_normal = supplied_q
        else:
            local_normal = np.asarray([-local_direction[1], local_direction[0]], dtype=np.float64)
            if float(local_normal @ width_vector) < 0.0:
                local_normal *= -1.0
        item_left = curve[index] - local_normal * widths[index] * left_share
        item_right = curve[index] + local_normal * widths[index] * right_share
        obstacle_fraction = _v25_cross_section_mask_fraction(item_left, item_right, obstacles)
        obstacle_hits += int(obstacle_fraction >= threshold)
        result.append({'center': curve[index].copy(), 'left': item_left, 'right': item_right, 'obstacle_width_fraction': obstacle_fraction, 'road_width_fraction': _v25_cross_section_mask_fraction(item_left, item_right, road), 'source': 'direct' if index == len(curve) - 1 else 'interpolated', 'fragment_pair_reacquired': True})
    if obstacle_hits < 2:
        return []
    previous_center = target_center
    for observation in observations[1:]:
        center = np.asarray(observation['center'], dtype=np.float64)
        local_direction = center - previous_center
        norm = float(np.linalg.norm(local_direction))
        local_direction = end_tangent if norm <= 1e-06 else local_direction / norm
        local_normal = np.asarray([-local_direction[1], local_direction[0]], dtype=np.float64)
        if float(local_normal @ width_vector) < 0.0:
            local_normal *= -1.0
        local_width = float(observation['width'])
        item_left = center - local_normal * local_width / 2.0
        item_right = center + local_normal * local_width / 2.0
        result.append({'center': center, 'left': item_left, 'right': item_right, 'obstacle_width_fraction': _v25_cross_section_mask_fraction(item_left, item_right, obstacles), 'road_width_fraction': _v25_cross_section_mask_fraction(item_left, item_right, road), 'source': 'direct', 'fragment_pair_reacquired': True})
        previous_center = center
    return result

def _v28_apply_obstacle_lateral_reconstruction(corridor: dict, navigation_centers: np.ndarray, obstacles: np.ndarray, road: np.ndarray, p: Parameters) -> tuple[dict, np.ndarray, np.ndarray, list[str], dict]:
    # 【主要功能：障礙物支持的側向道路補齊】
    # 若可見 Road 某側突然內凹，沿已確立導航的橫向方向補回缺失側；
    # 補齊不得超過附近預期寬度，也不得把新增區域當成直接 Road 觀測。
    transformed = dict(corridor)
    centers = np.asarray(navigation_centers, dtype=np.float64).copy()
    left = np.asarray(corridor.get('left', []), dtype=np.float64).copy()
    right = np.asarray(corridor.get('right', []), dtype=np.float64).copy()
    sources = list(corridor.get('band_source_types', []))
    count = min(len(centers), len(left), len(right), len(sources))
    if count < 3:
        return (transformed, left[:count], right[:count], sources[:count], {'applied': False, 'reason': 'too_few_stations', 'modified_station_count': 0})
    centers = centers[:count]
    left = left[:count]
    right = right[:count]
    sources = sources[:count]
    original_left = left.copy()
    original_right = right.copy()
    widths = np.linalg.norm(right - left, axis=1)
    reliable = np.asarray([source not in {'extrapolated', 'auxiliary_supported_extrapolated'} for source in sources], dtype=bool)
    radius = max(2, int(p.v28_lateral_width_window))
    records: list[dict] = []
    for index in range(1, count - 1):
        if not reliable[index] or widths[index] < float(p.min_road_width):
            continue
        lo = max(0, index - radius)
        hi = min(count, index + radius + 1)
        local_indices = np.flatnonzero(reliable[lo:hi]) + lo
        if len(local_indices) < 3:
            continue
        expected_width = max(float(widths[index]), float(np.percentile(widths[local_indices], 65)))
        if widths[index] >= expected_width * float(p.v28_lateral_visible_width_ratio):
            continue
        missing = float(expected_width - widths[index])
        if missing < 2.0:
            continue
        cross_vector = right[index] - left[index]
        cross_norm = float(np.linalg.norm(cross_vector))
        if cross_norm <= 1e-06:
            continue
        cross_unit = cross_vector / cross_norm
        candidate_left = left[index] - cross_unit * missing
        candidate_right = right[index] + cross_unit * missing
        left_support = _v25_cross_section_mask_fraction(candidate_left, left[index], obstacles)
        right_support = _v25_cross_section_mask_fraction(right[index], candidate_right, obstacles)
        threshold = float(p.v28_lateral_obstacle_fraction)
        if max(left_support, right_support) < threshold:
            continue
        old_left = left[index].copy()
        old_right = right[index].copy()
        if left_support > right_support:
            previous = next((item for item in range(index - 1, lo - 1, -1) if reliable[item] and widths[item] >= 0.9 * expected_width), None)
            following = next((item for item in range(index + 1, hi) if reliable[item] and widths[item] >= 0.9 * expected_width), None)
            if previous is None or following is None:
                continue
            alpha = (index - previous) / max(1, following - previous)
            smooth_side = (1.0 - alpha) * original_left[previous] + alpha * original_left[following]
            allowed = max(0.0, float((left[index] - smooth_side) @ cross_unit)) + 0.06 * expected_width
            actual_missing = min(missing, allowed)
            if actual_missing < 1.0:
                continue
            left[index] = left[index] - cross_unit * actual_missing
            side = 'left'
        else:
            previous = next((item for item in range(index - 1, lo - 1, -1) if reliable[item] and widths[item] >= 0.9 * expected_width), None)
            following = next((item for item in range(index + 1, hi) if reliable[item] and widths[item] >= 0.9 * expected_width), None)
            if previous is None or following is None:
                continue
            alpha = (index - previous) / max(1, following - previous)
            smooth_side = (1.0 - alpha) * original_right[previous] + alpha * original_right[following]
            allowed = max(0.0, float((smooth_side - right[index]) @ cross_unit)) + 0.06 * expected_width
            actual_missing = min(missing, allowed)
            if actual_missing < 1.0:
                continue
            right[index] = right[index] + cross_unit * actual_missing
            side = 'right'
        sources[index] = 'one_side_inferred'
        records.append({'station_index': int(index), 'band': int(index), 'side': side, 'old_left': old_left.tolist(), 'old_right': old_right.tolist(), 'new_left': left[index].tolist(), 'new_right': right[index].tolist(), 'visible_width_px': float(widths[index]), 'expected_width_px': float(expected_width), 'left_obstacle_fraction': float(left_support), 'right_obstacle_fraction': float(right_support), 'core_state': 'internal_reconstruction', 'detail_source': 'one_side_inferred', 'support_evidence': ['road_width_prior', 'obstacle']})
    lateral_canvas = Image.new('1', (road.shape[1], road.shape[0]), 0)
    draw = ImageDraw.Draw(lateral_canvas)
    line_width = max(2, int(p.row_band_height))
    affected = {int(record['station_index']): record for record in records}
    for index, record in affected.items():
        if record['side'] == 'left':
            old_point = original_left[index]
            new_point = left[index]
        else:
            old_point = original_right[index]
            new_point = right[index]
        draw.line((*tuple(np.rint(old_point).astype(int)), *tuple(np.rint(new_point).astype(int))), fill=1, width=line_width)
        next_record = affected.get(index + 1)
        if next_record is None or next_record['side'] != record['side']:
            continue
        if record['side'] == 'left':
            polygon = [tuple(np.rint(original_left[index]).astype(int)), tuple(np.rint(left[index]).astype(int)), tuple(np.rint(left[index + 1]).astype(int)), tuple(np.rint(original_left[index + 1]).astype(int))]
        else:
            polygon = [tuple(np.rint(original_right[index]).astype(int)), tuple(np.rint(right[index]).astype(int)), tuple(np.rint(right[index + 1]).astype(int)), tuple(np.rint(original_right[index + 1]).astype(int))]
        draw.polygon(polygon, fill=1)
    lateral_mask = np.asarray(lateral_canvas, dtype=bool) & ~road
    analysis = corridor_polygon_mask(road.shape, left, right) | road
    old_internal = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool) & ~road
    old_occlusion = np.asarray(corridor.get('inferred_occlusion_mask', np.zeros_like(road)), dtype=bool) & ~road
    old_supported = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(road)), dtype=bool) & ~road
    old_auxiliary = np.asarray(corridor.get('auxiliary_road_envelope_mask', np.zeros_like(road)), dtype=bool) & ~road
    transformed.update({'centerline': centers, 'left': left, 'right': right, 'band_source_types': sources, 'analysis_geometry_corridor': analysis, 'inferred_geometry_envelope': analysis, 'inferred_internal_gap_mask': old_internal, 'inferred_occlusion_mask': old_occlusion, 'obstacle_supported_reconstruction_mask': old_supported | lateral_mask, 'v28_lateral_reconstruction_mask': lateral_mask, 'full_road_corridor': road | old_internal | old_occlusion | old_supported | old_auxiliary | lateral_mask, 'corridor_mask': road | old_internal | old_occlusion | old_supported | old_auxiliary | lateral_mask, 'one_side_inferences': list(corridor.get('one_side_inferences', [])) + records})
    return (transformed, left, right, sources, {'applied': bool(records), 'reason': 'obstacle_supported_inward_side' if records else 'no_supported_inward_side', 'modified_station_count': int(len(records)), 'left_count': int(sum((record['side'] == 'left' for record in records))), 'right_count': int(sum((record['side'] == 'right' for record in records))), 'added_area_pixels': int(lateral_mask.sum()), 'centerline_modified': False, 'maximum_width_policy': 'local_road_width_prior'})

def _v25_rebuild_navigation_guided_terminal(corridor: dict, navigation_centers: np.ndarray, obstacles: np.ndarray, road: np.ndarray, guided_mask: np.ndarray, p: Parameters, navigation_tangents: np.ndarray | None=None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    # 【主要功能：依最終導航方向重建受遮擋的道路末端】
    # 從最後可靠道路的中心、左右邊界與寬度向前延伸；障礙物只作為遮擋
    # 支持，延伸寬度受近期道路限制，並記錄 extrapolated 來源與統計。
    transformed = dict(corridor)
    centers = np.asarray(navigation_centers, dtype=np.float64)
    left = np.asarray(corridor.get('left', []), dtype=np.float64)
    right = np.asarray(corridor.get('right', []), dtype=np.float64)
    sources = list(corridor.get('band_source_types', []))
    count = min(len(centers), len(left), len(right), len(sources))
    if count < 3:
        return (transformed, centers[:count], left[:count], right[:count], sources[:count], np.asarray(guided_mask, dtype=bool)[:count], {'applied': False, 'reason': 'too_few_stations'})
    centers = centers[:count]
    left = left[:count]
    right = right[:count]
    sources = sources[:count]
    guide = np.asarray(guided_mask, dtype=bool)
    if len(guide) < count:
        guide = np.pad(guide, (0, count - len(guide)), constant_values=False)
    guide = guide[:count]
    supplied_tangents = np.asarray(navigation_tangents if navigation_tangents is not None else [], dtype=np.float64)
    if supplied_tangents.ndim != 2 or supplied_tangents.shape != (count, 2):
        supplied_tangents = np.empty((0, 2), dtype=np.float64)
    reliable = np.asarray([source != 'extrapolated' for source in sources], dtype=bool)
    reliable_indices = np.flatnonzero(reliable)
    if len(reliable_indices) < 2:
        return (transformed, centers, left, right, sources, guide, {'applied': False, 'reason': 'too_few_reliable_v12_stations'})
    core_start = int(reliable_indices[0])
    core_end = int(reliable_indices[-1])
    core_centers = centers[core_start:core_end + 1].copy()
    core_left = left[core_start:core_end + 1].copy()
    core_right = right[core_start:core_end + 1].copy()
    core_sources = sources[core_start:core_end + 1]
    core_guide = guide[core_start:core_end + 1]
    core_geometry = corridor_polygon_mask(road.shape, core_left, core_right)
    old_terminal_mask = mask_from_boundary_sources(road.shape, left, right, sources, {'extrapolated'}, line_width=max(1, int(p.row_band_height)))

    def signed_turn(first: np.ndarray, second: np.ndarray) -> float:
        return float(math.atan2(first[0] * second[1] - first[1] * second[0], float(first @ second)))

    def generate(side: int) -> list[dict]:
        window = min(len(core_centers), max(3, int(p.one_sided_fit_observations)))
        sequence = core_centers[-window:] if side > 0 else core_centers[:window][::-1]
        deltas = np.diff(sequence, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-06
        if not np.any(valid):
            return []
        directions = deltas[valid] / lengths[valid, None]
        for index in range(1, len(directions)):
            if float(directions[index] @ directions[index - 1]) < 0.0:
                directions[index] *= -1.0
        tangent = directions[-1].copy()
        global_anchor_index = core_end if side > 0 else core_start
        if len(supplied_tangents) == count:
            supplied = supplied_tangents[global_anchor_index].copy()
            supplied_norm = float(np.linalg.norm(supplied))
            if supplied_norm > 1e-06:
                supplied /= supplied_norm
                if float(supplied @ tangent) < 0.0:
                    supplied *= -1.0
                tangent = supplied
        turns = [signed_turn(directions[index - 1], directions[index]) for index in range(1, len(directions))]
        turn = float(np.median(turns)) if turns else 0.0
        turn_limit = math.radians(max(0.0, float(p.v25_terminal_turn_limit_deg)))
        turn = float(np.clip(turn, -turn_limit, turn_limit))
        step_length = max(1.0, float(np.median(lengths[valid])))
        anchor = len(core_centers) - 1 if side > 0 else 0
        anchor_left = core_left[anchor]
        anchor_right = core_right[anchor]
        previous_center = core_centers[anchor].copy()
        width_vector = anchor_right - anchor_left
        width_value = max(float(p.min_road_width), float(np.linalg.norm(width_vector)))
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        if float(normal @ width_vector) < 0.0:
            normal *= -1.0
        pair_slice = slice(max(0, len(core_centers) - window), len(core_centers)) if side > 0 else slice(0, min(window, len(core_centers)))
        pair_vectors = core_right[pair_slice] - core_left[pair_slice]
        pair_norms = np.linalg.norm(pair_vectors, axis=1)
        pair_valid = pair_norms > 1e-06
        if np.any(pair_valid):
            pair_units = pair_vectors[pair_valid] / pair_norms[pair_valid, None]
            for pair_index in range(1, len(pair_units)):
                if float(pair_units[pair_index] @ pair_units[pair_index - 1]) < 0.0:
                    pair_units[pair_index] *= -1.0
            q_direction = np.median(pair_units, axis=0)
            q_direction /= max(float(np.linalg.norm(q_direction)), 1e-06)
        else:
            q_direction = normal.copy()
        anchor_q = width_vector / max(float(np.linalg.norm(width_vector)), 1e-06)
        if float(q_direction @ anchor_q) < 0.0:
            q_direction *= -1.0
        local_centers_for_width = core_centers[pair_slice][pair_valid]
        local_left_for_width = core_left[pair_slice][pair_valid]
        local_right_for_width = core_right[pair_slice][pair_valid]
        if len(local_centers_for_width):
            left_distance_value = max(0.25 * float(p.min_road_width), float(np.median(np.linalg.norm(local_centers_for_width - local_left_for_width, axis=1))))
            right_distance_value = max(0.25 * float(p.min_road_width), float(np.median(np.linalg.norm(local_right_for_width - local_centers_for_width, axis=1))))
        else:
            left_distance_value = right_distance_value = width_value / 2.0
        paired_fragment = _v25_forward_fragment_pair(road, obstacles, core_geometry, previous_center, anchor_left, anchor_right, tangent, step_length, p, transverse_direction=q_direction if bool(p.enable_v37_paired_cross_sections) else None, left_distance=left_distance_value, right_distance=right_distance_value)
        if paired_fragment:
            return paired_fragment
        maximum_steps = min(max(1, int(p.v28_terminal_max_bands)), int(math.ceil(math.hypot(*road.shape) / step_length)))
        base_tangent = tangent.copy()

        def probe_direction(candidate_tangent: np.ndarray) -> list[dict]:
            candidate_tangent = np.asarray(candidate_tangent, dtype=np.float64)
            candidate_tangent /= max(1e-06, float(np.linalg.norm(candidate_tangent)))
            candidate_dot = float(np.clip(base_tangent @ candidate_tangent, -1.0, 1.0))
            direction_delta = math.degrees(math.acos(candidate_dot))
            transition_steps = 1 if direction_delta < 12.0 else 4
            local_center = core_centers[anchor].copy()
            items: list[dict] = []
            for step in range(1, maximum_steps + 1):
                alpha = min(1.0, float(step) / float(transition_steps))
                local_tangent = (1.0 - alpha) * base_tangent + alpha * candidate_tangent
                local_norm = float(np.linalg.norm(local_tangent))
                if local_norm <= 1e-06:
                    break
                local_tangent /= local_norm
                next_center = local_center + local_tangent * step_length
                if bool(p.enable_v37_paired_cross_sections):
                    local_normal = q_direction
                    next_left = next_center - local_normal * left_distance_value
                    next_right = next_center + local_normal * right_distance_value
                else:
                    local_normal = np.asarray([-local_tangent[1], local_tangent[0]], dtype=np.float64)
                    if float(local_normal @ width_vector) < 0.0:
                        local_normal *= -1.0
                    next_left = next_center - local_normal * width_value / 2.0
                    next_right = next_center + local_normal * width_value / 2.0
                points = np.vstack((next_center, next_left, next_right))
                if np.all(points[:, 0] < 0) or np.all(points[:, 0] >= road.shape[1]) or np.all(points[:, 1] < 0) or np.all(points[:, 1] >= road.shape[0]):
                    break
                items.append({'center': next_center.copy(), 'left': next_left.copy(), 'right': next_right.copy(), 'obstacle_width_fraction': _v25_cross_section_mask_fraction(next_left, next_right, obstacles), 'road_width_fraction': _v25_cross_section_mask_fraction(next_left, next_right, road), 'source': 'extrapolated', 'terminal_direction_delta_deg': float(direction_delta), 'terminal_transverse_direction': local_normal.tolist()})
                local_center = next_center
            return items
        threshold = float(p.v28_terminal_obstacle_fraction)

        def supported_prefix(items: list[dict]) -> tuple[int, int]:
            start_gap_bands = max(int(p.one_sided_start_gap_bands), int(math.ceil(float(p.v28_terminal_start_gap_widths) * width_value / max(step_length, 1.0))))
            first = next((index for index, item in enumerate(items[:max(1, start_gap_bands + 1)]) if item['obstacle_width_fraction'] >= threshold), None)
            if first is None:
                return (0, 0)
            last_supported = first - 1
            misses = 0
            support_count = 0
            for index in range(first, len(items)):
                if items[index]['obstacle_width_fraction'] >= threshold:
                    last_supported = index
                    support_count += 1
                    misses = 0
                else:
                    misses += 1
                    if misses > int(p.one_sided_empty_tolerance_bands):
                        break
            return (last_supported + 1, support_count)
        probed = probe_direction(base_tangent)
        terminal_count, terminal_support_count = supported_prefix(probed)
        terminal_direction_mode = 'final_navigation_tangent'
        if terminal_support_count < 2 and (not bool(p.enable_v29_direction_locked_terminal)):
            best = None
            for delta_degrees in range(-105, 106, 15):
                if delta_degrees == 0:
                    continue
                radians = math.radians(float(delta_degrees))
                cosine, sine = (math.cos(radians), math.sin(radians))
                candidate = np.asarray([cosine * base_tangent[0] - sine * base_tangent[1], sine * base_tangent[0] + cosine * base_tangent[1]])
                candidate_items = probe_direction(candidate)
                candidate_count, candidate_support = supported_prefix(candidate_items)
                if candidate_support < 3:
                    continue
                obstacle_sum = float(sum((item['obstacle_width_fraction'] for item in candidate_items[:candidate_count])))
                score = (int(candidate_support), obstacle_sum, -abs(int(delta_degrees)))
                if best is None or score > best[0]:
                    best = (score, candidate_items, candidate_count, delta_degrees)
            if best is not None:
                _, probed, terminal_count, chosen_delta = best
                terminal_direction_mode = f'local_obstacle_continuation_{chosen_delta:+d}deg'
        for item in probed[:terminal_count]:
            item['terminal_direction_mode'] = terminal_direction_mode
        first_road = None
        run = 0
        road_threshold = 0.2
        for index, item in enumerate(probed[terminal_count:], terminal_count):
            if item['road_width_fraction'] >= road_threshold:
                run += 1
                if run >= 3:
                    first_road = index - run + 1
                    break
            else:
                run = 0
        if first_road is None or terminal_count == 0:
            return probed[:terminal_count]
        last_road = first_road
        misses = 0
        for index in range(first_road, len(probed)):
            if probed[index]['road_width_fraction'] >= road_threshold:
                last_road = index
                misses = 0
            else:
                misses += 1
                if misses > 2:
                    break
        accepted = probed[:last_road + 1]
        for index, item in enumerate(accepted):
            item['source'] = 'direct' if index >= first_road else 'interpolated'
            item['fragment_pair_reacquired'] = True
        return accepted
    lower = generate(-1)
    upper = generate(1)
    lower_centers = np.asarray([item['center'] for item in reversed(lower)], dtype=np.float64).reshape(-1, 2)
    lower_left = np.asarray([item['left'] for item in reversed(lower)], dtype=np.float64).reshape(-1, 2)
    lower_right = np.asarray([item['right'] for item in reversed(lower)], dtype=np.float64).reshape(-1, 2)
    upper_centers = np.asarray([item['center'] for item in upper], dtype=np.float64).reshape(-1, 2)
    upper_left = np.asarray([item['left'] for item in upper], dtype=np.float64).reshape(-1, 2)
    upper_right = np.asarray([item['right'] for item in upper], dtype=np.float64).reshape(-1, 2)
    new_centers = np.vstack((lower_centers, core_centers, upper_centers))
    new_left = np.vstack((lower_left, core_left, upper_left))
    new_right = np.vstack((lower_right, core_right, upper_right))
    new_sources = [str(item.get('source', 'extrapolated')) for item in reversed(lower)] + list(core_sources) + [str(item.get('source', 'extrapolated')) for item in upper]
    new_guide = np.concatenate((np.full(len(lower), True, dtype=bool), core_guide, np.full(len(upper), True, dtype=bool)))
    new_analysis = corridor_polygon_mask(road.shape, new_left, new_right) | road
    new_terminal_mask = mask_from_boundary_sources(road.shape, new_left, new_right, new_sources, {'extrapolated'}, line_width=max(1, int(p.row_band_height))) & new_analysis & ~road
    new_internal_mask = mask_from_boundary_sources(road.shape, new_left, new_right, new_sources, {'interpolated'}, line_width=max(1, int(p.row_band_height))) & new_analysis & ~road
    old_occlusion = np.asarray(corridor.get('inferred_occlusion_mask', np.zeros_like(road)), dtype=bool) & ~road
    old_supported = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(road)), dtype=bool) & ~road
    old_internal = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool) & ~road
    transformed.update({'centerline': new_centers, 'left': new_left, 'right': new_right, 'band_source_types': new_sources, 'analysis_geometry_corridor': new_analysis, 'inferred_geometry_envelope': new_analysis, 'inferred_occlusion_mask': old_occlusion & ~old_terminal_mask | new_terminal_mask & obstacles, 'obstacle_supported_reconstruction_mask': old_supported & ~old_terminal_mask | new_terminal_mask, 'inferred_internal_gap_mask': old_internal | new_internal_mask, 'one_sided_extrapolations': [{'direction': direction, 'bands': int(sum((item.get('source', 'extrapolated') == 'extrapolated' for item in items))), 'mean_obstacle_width_fraction': float(np.mean([item['obstacle_width_fraction'] for item in items])) if items else 0.0, 'guidance': 'v25_final_navigation_direction'} for direction, items in (('lower', lower), ('upper', upper)) if any((item.get('source', 'extrapolated') == 'extrapolated' for item in items))], 'one_sided_geometry': [{'direction': direction, 'centerline': [np.asarray(item['center'], dtype=np.float64).tolist() for item in items if item.get('source', 'extrapolated') == 'extrapolated'], 'left': [np.asarray(item['left'], dtype=np.float64).tolist() for item in items if item.get('source', 'extrapolated') == 'extrapolated'], 'right': [np.asarray(item['right'], dtype=np.float64).tolist() for item in items if item.get('source', 'extrapolated') == 'extrapolated']} for direction, items in (('lower', lower), ('upper', upper)) if any((item.get('source', 'extrapolated') == 'extrapolated' for item in items))], 'extrapolation_ratio': float(sum((source == 'extrapolated' for source in new_sources)) / max(1, len(new_sources)))})
    summary = {'applied': bool(lower or upper), 'reason': 'state_routed_terminal_or_fragment_pair' if lower or upper else 'no_supported_terminal_or_fragment_pair', 'old_terminal_bands': int(sum((source == 'extrapolated' for source in sources))), 'new_terminal_bands': int(sum((source == 'extrapolated' for source in new_sources))), 'lower_bands': int(sum((item.get('source', 'extrapolated') == 'extrapolated' for item in lower))), 'upper_bands': int(sum((item.get('source', 'extrapolated') == 'extrapolated' for item in upper))), 'lower_total_bands': int(len(lower)), 'upper_total_bands': int(len(upper)), 'fragment_pair_reacquired': bool(any((item.get('fragment_pair_reacquired', False) for item in lower + upper))), 'obstacle_threshold': float(p.v28_terminal_obstacle_fraction), 'width_policy': 'frozen_v12_anchor_width', 'direction_policy': 'frozen_final_navigation_tangent', 'terminal_direction_modes': sorted({str(item.get('terminal_direction_mode', 'fragment_pair')) for item in lower + upper}), 'centerline_feedback_enabled': False}
    return (transformed, new_centers, new_left, new_right, new_sources, new_guide, summary)

# =============================================================================
# 15. V40 核心整合器
# =============================================================================
# 保留 V12 道路本體，融合實體 Roadline、導航方向、遮擋重建與專用邊界
# 配對；這是判定前最重要的整合層，輸出最終中心、邊界及診斷資訊。
def apply_v20_v12_base_v19_direction(road: np.ndarray, obstacles: np.ndarray, corridor: dict, auxiliary_masks: dict[str, np.ndarray] | None, p: Parameters) -> dict:
    # 【主要功能：V40 道路幾何總整合】
    # 輸入已選定的 V12 道路候選，依序建立輔助實體軌跡、判定中央
    # Roadline、決定導航模式、執行側向／末端重建及專用左右邊界配對。
    # 輸出供 road_aligned_method 量測阻塞率的最終道路幾何與完整診斷欄位。
    if not corridor.get('ok'):
        return corridor
    transformed, centers, left, right, sources, count = _copy_active_corridor_geometry(corridor)
    if count < 3:
        return transformed
    base_centers = centers.copy()
    base_left = left.copy()
    base_right = right.copy()
    base_sources = list(sources)
    working = dict(corridor)
    working.update({'centerline': centers, 'left': left, 'right': right})
    records, selected_masks, track_summary = _v13_associated_auxiliary_components(auxiliary_masks, working, road.shape, p)
    legacy_probe = apply_v13_auxiliary_road_reconstruction(road, obstacles, working, auxiliary_masks, p) if records else working
    reference = legacy_probe.get('v18_selected_curve_reference')
    global_angle = float(corridor.get('selected_direction_angle_deg', 90.0))
    if bool(p.enable_v40_v38_roadline_center_sections):
        centers = base_centers.copy()
        left = base_left.copy()
        right = base_right.copy()
        sources = list(base_sources)
        guided_indices: set[int] = set()
        guided_mask = np.zeros(len(centers), dtype=bool)
        center_safety = {'applied': False, 'reason': 'v40_v12_geometry_preserved_before_center_selection'}
    else:
        centers, left, right, guided_indices, guided_mask, reference, center_safety = _v20_apply_reference_direction(centers, left, right, reference, global_angle, p, sources)
    provisional_navigation_tangents, _, _ = _v31_build_navigation_direction_field(records, centers, global_angle, p) if bool(p.enable_v31_multi_aux_pretrack_direction) and (not bool(p.enable_v40_v38_roadline_center_sections)) else (_v31_long_window_road_tangents(centers, int(p.v31_road_direction_window_bands)), np.zeros(len(centers), dtype=bool), {})
    lateral_summary = {'applied': False, 'reason': 'v28_disabled'}
    terminal_summary = {'applied': False, 'reason': 'v28_disabled'}
    if p.enable_v28_frozen_center_state_reconstruction and (not bool(p.enable_v37_paired_cross_sections)):
        corridor, left, right, sources, lateral_summary = _v28_apply_obstacle_lateral_reconstruction(corridor, centers, obstacles, road, p)
        corridor['centerline'] = centers.copy()
        corridor, centers, left, right, sources, guided_mask, terminal_summary = _v25_rebuild_navigation_guided_terminal(corridor, centers, obstacles, road, guided_mask, p, provisional_navigation_tangents)
        guided_indices = set(np.flatnonzero(guided_mask).tolist())
        base_left = left.copy()
        base_right = right.copy()
        base_sources = list(sources)
    elif p.enable_v25_navigation_guided_reconstruction and (not bool(p.enable_v37_paired_cross_sections)):
        corridor, centers, left, right, sources, guided_mask, terminal_summary = _v25_rebuild_navigation_guided_terminal(corridor, centers, obstacles, road, guided_mask, p, provisional_navigation_tangents)
        guided_indices = set(np.flatnonzero(guided_mask).tolist())
        base_left = left.copy()
        base_right = right.copy()
        base_sources = list(sources)
    if bool(p.enable_v31_multi_aux_pretrack_direction):
        lower_station_shift = int(terminal_summary.get('lower_total_bands', 0))
        direction_records: list[dict] = []
        for record in records:
            shifted = dict(record)
            station_indices = np.asarray(record.get('station_indices', []), dtype=np.int32)
            if lower_station_shift and len(station_indices):
                shifted['station_indices'] = station_indices + lower_station_shift
            direction_records.append(shifted)
        navigation_tangents, navigation_support_mask, navigation_direction_summary = _v31_build_navigation_direction_field(direction_records, centers, float(corridor.get('selected_direction_angle_deg', global_angle)), p)
    else:
        navigation_tangents = _v31_long_window_road_tangents(centers, int(p.v31_road_direction_window_bands))
        navigation_support_mask = np.zeros(len(centers), dtype=bool)
        navigation_direction_summary = {'accepted_curve_count': 0, 'supported_bands': 0, 'conflict_bands': 0}
    if bool(p.enable_v32_authoritative_navigation_direction):
        auxiliary_area = np.zeros_like(road, dtype=bool)
        auxiliary_polygons: list[list[list[float]]] = []
        auxiliary_area_summary = {'enabled': False, 'reason': 'v32_direction_only_frozen_road_geometry', 'accepted_component_count': 0, 'accepted_run_count': 0, 'accepted_station_count': 0, 'added_pixels': 0}
    else:
        auxiliary_area, auxiliary_polygons, auxiliary_area_summary = _v19_auxiliary_road_envelope(road, obstacles, centers, left, right, records, p, confirmed_centerline_available=bool(reference is not None and str(reference.get('role', '')) == 'confirmed_centerline' and (str(reference.get('source_family', '')) == 'road_line')))
    expansion_summary = {'enabled': False, 'reason': 'v21_boundaries_frozen', 'modified_station_count': 0, 'left_expansion_count': 0, 'right_expansion_count': 0, 'maximum_width_growth_ratio': 1.0}
    left = base_left
    right = base_right
    sources = base_sources
    base_analysis = np.asarray(corridor.get('analysis_geometry_corridor', corridor_polygon_mask(road.shape, base_left, base_right)), dtype=bool)
    base_internal = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool) & ~road
    base_occlusion = np.asarray(corridor.get('inferred_occlusion_mask', np.zeros_like(road)), dtype=bool) & ~road
    base_obstacle_supported = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(road)), dtype=bool) & ~road
    auxiliary_area = np.asarray(auxiliary_area, dtype=bool) & ~road
    analysis = base_analysis | auxiliary_area | road
    inferred_internal = base_internal | auxiliary_area
    full = road | base_internal | base_occlusion | base_obstacle_supported | auxiliary_area
    if bool(p.enable_v32_authoritative_navigation_direction):
        v32_anchor_centers = 0.5 * (np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64))
        direction_working = dict(corridor)
        direction_working.update({'centerline': v32_anchor_centers, 'left': left, 'right': right})
        v32_records, _, v32_track_summary = _v13_associated_auxiliary_components(auxiliary_masks, direction_working, road.shape, p, direction_only=True)
        if bool(p.enable_v33_physical_roadline_tracks):
            v32_records, v33_physical_tracks = _v33_build_unified_auxiliary_tracks(v32_records, v32_anchor_centers, left, right, road, np.asarray(analysis | road, dtype=bool), p, legacy_reference=reference)
        else:
            v33_physical_tracks = []
        if bool(p.enable_v40_v38_roadline_center_sections):
            central_indices = [index for index, record in enumerate(v32_records) if str(record.get('source_family', '')) == 'road_line' and str(record.get('v33_role', 'D')) == 'C']
            for index, record in enumerate(v32_records):
                if index in central_indices:
                    record['v33_adoption'] = 'used'
                elif str(record.get('v33_role', 'D')) == 'X':
                    record['v33_adoption'] = 'rejected'
                else:
                    record['v33_adoption'] = 'candidate'
            v32_navigation_axis = v32_anchor_centers.copy()
            v32_navigation_tangents = _v31_long_window_road_tangents(v32_navigation_axis, int(p.v32_road_fallback_window_bands))
            v32_authoritative_mask = np.zeros(len(v32_navigation_axis), dtype=bool)
            v32_authoritative_curves = []
            v32_navigation_summary = {'status': 'v40_confirmed_central_roadline_ready' if len(central_indices) == 1 else 'v40_v12_global_no_unique_center', 'accepted_curve_count': int(len(central_indices) == 1), 'accepted_record_indices': central_indices if len(central_indices) == 1 else [], 'authoritative_bands': 0, 'conflict_bands': 0, 'v33_central_track_selected': bool(len(central_indices) == 1), 'auxiliary_native_enabled': False}
        else:
            v32_navigation_axis, v32_navigation_tangents, v32_authoritative_mask, v32_authoritative_curves, v32_navigation_summary = _v32_authoritative_navigation_axis(v32_records, v32_anchor_centers, left, right, sources, float(corridor.get('selected_direction_angle_deg', global_angle)), np.asarray(analysis | road, dtype=bool), p)
        v40_v12_centers = np.asarray(v32_anchor_centers, dtype=np.float64).copy()
        v40_v12_left = np.asarray(left, dtype=np.float64).copy()
        v40_v12_right = np.asarray(right, dtype=np.float64).copy()
        v40_v12_sources = list(sources)
        v40_v12_tangents = _v31_long_window_road_tangents(v40_v12_centers, int(p.v32_road_fallback_window_bands))
        v35_native_center_summary = {'applied': False, 'reason': 'no_confirmed_center_selected', 'native_station_count': 0}
        v35_native_center_mask = np.zeros(len(v32_navigation_axis), dtype=bool)
        v36_center_source_labels = ['v12'] * len(v32_navigation_axis)
        if bool(p.enable_v35_native_roadline_center) and bool(v32_navigation_summary.get('v33_central_track_selected', False)):
            if bool(p.enable_v36_geometric_roadline_authority):
                v32_navigation_axis, left, right, sources, v32_navigation_tangents, v35_native_center_mask, v36_center_source_labels, v35_native_center_summary = _v36_apply_geometric_central_roadline(v32_navigation_axis, left, right, sources, v32_records, np.asarray(analysis | road, dtype=bool), p)
            else:
                v32_navigation_axis, left, right, sources, v32_navigation_tangents, v35_native_center_mask, v35_native_center_summary = _v35_apply_native_central_roadline(v32_navigation_axis, left, right, sources, v32_records, np.asarray(analysis | road, dtype=bool), p)
                v36_center_source_labels = ['roadline_center' if value else 'v12' for value in v35_native_center_mask]
            if bool(v35_native_center_summary.get('applied', False)):
                v32_authoritative_mask = v35_native_center_mask.copy()
                v32_navigation_summary['authoritative_bands'] = int(v35_native_center_mask.sum())
                v32_navigation_summary['central_roadline_anchor_bands'] = int(v35_native_center_mask.sum())
                v32_navigation_summary['status'] = 'v36_geometric_central_roadline_applied' if bool(p.enable_v36_geometric_roadline_authority) else 'v35_native_central_roadline_applied'
        v32_navigation_summary['v35_native_center'] = v35_native_center_summary
        if bool(p.enable_v33_physical_roadline_tracks):
            for item in v33_physical_tracks:
                record_index = int(item.get('record_index', -1))
                if 0 <= record_index < len(v32_records):
                    record = v32_records[record_index]
                    item['adoption'] = str(record.get('v33_adoption', item.get('adoption', 'candidate')))
                    item['role'] = str(record.get('v33_role', item.get('role', 'D')))
        v32_navigation_summary['track_summary'] = v32_track_summary
    else:
        v32_navigation_axis = centers.copy()
        v32_navigation_tangents = navigation_tangents.copy()
        v32_authoritative_mask = np.zeros(len(centers), dtype=bool)
        v32_authoritative_curves = []
        v32_navigation_summary = {'status': 'disabled', 'accepted_curve_count': 0, 'authoritative_bands': 0, 'conflict_bands': 0}
        v33_physical_tracks = []
        v35_native_center_mask = np.zeros(len(centers), dtype=bool)
        v36_center_source_labels = ['v12'] * len(centers)
        v35_native_center_summary = {'applied': False, 'reason': 'v32_disabled', 'native_station_count': 0}
    if bool(v35_native_center_summary.get('applied', False)):
        centers = np.asarray(v32_navigation_axis, dtype=np.float64).copy()
    elif bool(p.enable_v33_physical_roadline_tracks and v32_navigation_summary.get('v33_central_track_selected', False) and (len(v32_navigation_axis) == len(centers))):
        centers = np.asarray(v32_navigation_axis, dtype=np.float64).copy()
    v40_enabled = bool(p.enable_v40_v38_roadline_center_sections)
    v40_navigation_mode = 'roadline_center' if v40_enabled and bool(v35_native_center_summary.get('applied', False)) else 'v12_global'
    v40_native_fallback = False
    v40_native_pairing_quality = {'accepted': v40_navigation_mode != 'roadline_center', 'success_ratio': 0.0, 'inside_ratio': 0.0, 'monotonic_ratio': 0.0, 'width_step_p90': float('inf'), 'crossing_count': 0}
    v40_central_records = [record for record in v32_records if str(record.get('source_family', '')) == 'road_line' and str(record.get('v33_role', 'D')) == 'C'] if bool(p.enable_v32_authoritative_navigation_direction) else []
    v40_central_name = str(v40_central_records[0].get('v33_track_id', 'Roadline center')) if len(v40_central_records) == 1 else 'none'
    v40_central_coverage = float(v40_central_records[0].get('coverage', 0.0)) if len(v40_central_records) == 1 else 0.0
    v40_frozen_pair_valid_mask = np.ones(len(centers), dtype=bool)
    v37_post_pairing_summary = {'enabled': bool(p.enable_v37_paired_cross_sections), 'initial_success_ratio': 0.0, 'repaired_success_ratio': 0.0}
    if bool(p.enable_v37_paired_cross_sections) and len(centers) >= 3:
        if len(v32_navigation_tangents) != len(centers):
            v32_navigation_tangents = _v37_unit_vectors(centers, window=4)
        frozen_mask = np.asarray(analysis | road, dtype=bool)
        first_pairing = _v37_pair_boundaries_monotone(centers, np.asarray(v32_navigation_tangents, dtype=np.float64), left, right, frozen_mask, p, anchor_indices=np.arange(len(centers), dtype=np.int32), road_mask=road)
        if v40_navigation_mode == 'roadline_center':
            v40_native_pairing_quality = _v40_pairing_quality(first_pairing, p)
            if not bool(v40_native_pairing_quality.get('accepted', False)):
                v40_native_fallback = True
                v40_navigation_mode = 'v12_global'
                centers = np.asarray(v40_v12_centers, dtype=np.float64).copy()
                left = np.asarray(v40_v12_left, dtype=np.float64).copy()
                right = np.asarray(v40_v12_right, dtype=np.float64).copy()
                sources = list(v40_v12_sources)
                v32_navigation_tangents = np.asarray(v40_v12_tangents, dtype=np.float64).copy()
                v35_native_center_mask = np.zeros(len(centers), dtype=bool)
                v36_center_source_labels = ['v12'] * len(centers)
                v35_native_center_summary = dict(v35_native_center_summary)
                v35_native_center_summary.update({'applied': False, 'reason': 'v40_native_pairing_quality_fallback_to_v38', 'native_station_count': 0})
                frozen_mask = np.asarray(analysis | road, dtype=bool)
                first_pairing = _v37_pair_boundaries_monotone(centers, np.asarray(v32_navigation_tangents, dtype=np.float64), left, right, frozen_mask, p, anchor_indices=np.arange(len(centers), dtype=np.int32), road_mask=road)
        paired_valid = np.asarray(first_pairing['valid'], dtype=bool)
        v40_frozen_pair_valid_mask = paired_valid.copy()
        if len(paired_valid) == len(left):
            left = np.asarray(left, dtype=np.float64).copy()
            right = np.asarray(right, dtype=np.float64).copy()
            left[paired_valid] = np.asarray(first_pairing['left'])[paired_valid]
            right[paired_valid] = np.asarray(first_pairing['right'])[paired_valid]
        corridor.update({'centerline': np.asarray(centers, dtype=np.float64).copy(), 'left': left, 'right': right, 'band_source_types': list(sources), 'analysis_geometry_corridor': corridor_polygon_mask(road.shape, left, right) | road})
        corridor, left, right, sources, lateral_summary = _v28_apply_obstacle_lateral_reconstruction(corridor, np.asarray(centers, dtype=np.float64), obstacles, road, p)
        if v40_navigation_mode == 'roadline_center':
            repaired_pairing = first_pairing
            selected_pairing = first_pairing
        else:
            repaired_mask = np.asarray(corridor.get('analysis_geometry_corridor', frozen_mask) | road, dtype=bool)
            repaired_pairing = _v37_pair_boundaries_monotone(centers, np.asarray(v32_navigation_tangents, dtype=np.float64), left, right, repaired_mask, p, anchor_indices=np.arange(len(centers), dtype=np.int32), road_mask=road)
            selected_pairing = _v38_merge_pairings_without_regression(first_pairing, repaired_pairing) if bool(p.enable_v38_pairing_quality_gate) else repaired_pairing
            repaired_valid = np.asarray(selected_pairing['valid'], dtype=bool)
            if len(repaired_valid) == len(left):
                left = np.asarray(left, dtype=np.float64).copy()
                right = np.asarray(right, dtype=np.float64).copy()
                left[repaired_valid] = np.asarray(selected_pairing['left'])[repaired_valid]
                right[repaired_valid] = np.asarray(selected_pairing['right'])[repaired_valid]
        corridor.update({'centerline': np.asarray(centers, dtype=np.float64).copy(), 'left': left, 'right': right, 'band_source_types': list(sources), 'analysis_geometry_corridor': corridor_polygon_mask(road.shape, left, right) | road})
        terminal_guide = np.zeros(len(centers), dtype=bool)
        corridor, centers, left, right, sources, terminal_guide, terminal_summary = _v25_rebuild_navigation_guided_terminal(corridor, np.asarray(centers, dtype=np.float64), obstacles, road, terminal_guide, p, np.asarray(v32_navigation_tangents, dtype=np.float64))
        lower_added = int(terminal_summary.get('lower_total_bands', 0))
        upper_added = int(terminal_summary.get('upper_total_bands', 0))
        if lower_added or upper_added:
            v32_navigation_tangents = _v37_unit_vectors(centers, window=4)
            v40_frozen_pair_valid_mask = np.concatenate((np.ones(lower_added, dtype=bool), np.asarray(v40_frozen_pair_valid_mask, dtype=bool), np.ones(upper_added, dtype=bool)))
            v35_native_center_mask = np.concatenate((np.zeros(lower_added, dtype=bool), np.asarray(v35_native_center_mask, dtype=bool), np.zeros(upper_added, dtype=bool)))
            v36_center_source_labels = ['terminal_extrapolation'] * lower_added + list(v36_center_source_labels) + ['terminal_extrapolation'] * upper_added
        v32_navigation_axis = np.asarray(centers, dtype=np.float64).copy()
        analysis = np.asarray(corridor.get('analysis_geometry_corridor', corridor_polygon_mask(road.shape, left, right) | road), dtype=bool)
        base_internal = np.asarray(corridor.get('inferred_internal_gap_mask', base_internal), dtype=bool) & ~road
        base_occlusion = np.asarray(corridor.get('inferred_occlusion_mask', base_occlusion), dtype=bool) & ~road
        base_obstacle_supported = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', base_obstacle_supported), dtype=bool) & ~road
        inferred_internal = base_internal | auxiliary_area
        full = road | base_internal | base_occlusion | base_obstacle_supported | auxiliary_area
        v37_post_pairing_summary = {'enabled': True, 'initial_success_ratio': float(first_pairing['success_ratio']), 'repaired_success_ratio': float(repaired_pairing['success_ratio']), 'selected_success_ratio': float(selected_pairing['success_ratio']), 'initial_crossing_rejections': int(first_pairing['crossing_rejections']), 'repaired_crossing_rejections': int(repaired_pairing['crossing_rejections']), 'pairing_quality_gate': selected_pairing.get('quality_gate', {'policy': 'disabled_repaired_pairing_replaces_initial'}), 'lateral_reconstruction': lateral_summary, 'terminal_reconstruction': terminal_summary, 'lateral_direction_policy': 'final_paired_Q', 'terminal_center_policy': 'frozen_final_T', 'terminal_boundary_policy': 'last_reliable_paired_Q_and_asymmetric_offsets', 'v40_navigation_mode': v40_navigation_mode, 'v40_native_pairing_quality': v40_native_pairing_quality, 'v40_roadline_native_fallback_to_v38': bool(v40_native_fallback), 'v40_frozen_navigation_preserved': True}
    corridor['v40_navigation_mode'] = v40_navigation_mode
    corridor['v40_central_roadline_used'] = bool(v40_navigation_mode == 'roadline_center')
    corridor['v40_central_roadline_name'] = v40_central_name
    corridor['v40_central_roadline_coverage'] = float(v40_central_coverage)
    corridor['v40_native_pairing_quality'] = v40_native_pairing_quality
    corridor['v40_roadline_native_fallback_to_v38'] = bool(v40_native_fallback)
    corridor['v40_frozen_pair_valid_mask'] = np.asarray(v40_frozen_pair_valid_mask, dtype=bool).tolist()
    corridor['v40_frozen_navigation_preserved'] = True
    selected_curve_mask = _v20_selected_curve_mask(road.shape, reference)
    selected_role = str(reference.get('role', 'none')) if reference else 'none'
    selected_family = str(reference.get('source_family', 'none')) if reference else 'none'
    central_mask = selected_curve_mask if selected_role == 'confirmed_centerline' and selected_family == 'road_line' else np.zeros_like(road, dtype=bool)
    edge_mask = selected_curve_mask if selected_role == 'shape_shifted' and selected_family == 'road_line' else np.zeros_like(road, dtype=bool)
    geometry_metrics = corridor_geometry_metrics(centers, left, right, central_mask, inferred_internal, analysis)
    probe_summary = legacy_probe.get('v13_reconstruction_summary', {})
    paired_center_observations = int(probe_summary.get('paired_center_roadline_observations', 0))
    single_center_observations = int(probe_summary.get('single_center_roadline_observations', 0))
    center_fit_observations = int(probe_summary.get('center_fit_observation_bands', 0))
    central_target_observations = int(probe_summary.get('central_target_observation_bands', 0))
    edge_shape_observations = int(probe_summary.get('edge_shape_target_observation_bands', 0))
    single_boundary_hints = int(probe_summary.get('single_boundary_hint_observations', 0))
    shape_only_components = int(probe_summary.get('single_boundary_shape_only_components', 0))
    transformed.update({'centerline': centers, 'left': left, 'right': right, 'band_source_types': sources, 'source_counts': {source: int(sum((value == source for value in sources))) for source in ('direct', 'interpolated', 'interior_occlusion_inferred', 'one_side_inferred', 'extrapolated')}, 'one_sided_extrapolations': list(corridor.get('one_sided_extrapolations', [])), 'one_sided_geometry': list(corridor.get('one_sided_geometry', [])), 'one_side_inferences': list(corridor.get('one_side_inferences', [])), 'interior_occlusion_inferences': list(corridor.get('interior_occlusion_inferences', [])), 'interpolation_ratio': float(sum((source == 'interpolated' for source in sources)) / max(1, len(sources))), 'extrapolation_ratio': float(sum((source == 'extrapolated' for source in sources)) / max(1, len(sources))), 'analysis_geometry_corridor': analysis, 'inferred_geometry_envelope': analysis, 'inferred_internal_gap_mask': inferred_internal, 'inferred_occlusion_mask': base_occlusion, 'obstacle_supported_reconstruction_mask': base_obstacle_supported, 'auxiliary_road_envelope_mask': auxiliary_area, 'auxiliary_road_envelope_polygons': auxiliary_polygons, 'full_road_corridor': full, 'corridor_mask': full, 'geometry_metrics': geometry_metrics, 'v13_associated_roadline_mask': selected_masks['road_line'] & analysis, 'v13_central_roadline_mask': central_mask & analysis, 'v14_edge_roadline_mask': edge_mask & analysis, 'v13_associated_roadside_mask': selected_masks['roadside_boundary'] & analysis, 'v18_selected_curve_reference': reference, 'v21_guided_center_mask': guided_mask.tolist(), 'v31_navigation_tangents': navigation_tangents.tolist(), 'v31_navigation_support_mask': navigation_support_mask.tolist(), 'v31_navigation_direction_summary': navigation_direction_summary, 'v32_navigation_axis': v32_navigation_axis.tolist(), 'v32_navigation_tangents': v32_navigation_tangents.tolist(), 'v32_authoritative_support_mask': v32_authoritative_mask.tolist(), 'v32_authoritative_curves': v32_authoritative_curves, 'v32_navigation_direction_summary': v32_navigation_summary, 'v33_physical_tracks': v33_physical_tracks, 'v33_navigation_axis': v32_navigation_axis.tolist(), 'v33_navigation_tangents': v32_navigation_tangents.tolist(), 'v35_native_center_authority_mask': v35_native_center_mask.tolist(), 'v35_native_center_summary': v35_native_center_summary, 'v36_center_source_labels': v36_center_source_labels, 'v37_post_navigation_pairing': v37_post_pairing_summary, 'v40_navigation_mode': v40_navigation_mode, 'v40_central_roadline_used': bool(v40_navigation_mode == 'roadline_center'), 'v40_central_roadline_name': v40_central_name, 'v40_central_roadline_coverage': float(v40_central_coverage), 'v40_native_pairing_quality': v40_native_pairing_quality, 'v40_roadline_native_fallback_to_v38': bool(v40_native_fallback), 'v40_frozen_pair_valid_mask': np.asarray(v40_frozen_pair_valid_mask, dtype=bool).tolist(), 'v40_frozen_navigation_preserved': True, 'v13_reconstruction_status': 'v21_decoupled_geometry_applied', 'v13_reconstruction_summary': {'architecture': 'v40_v38_roadline_center_sections', 'v40_navigation_mode': v40_navigation_mode, 'v40_central_roadline_used': bool(v40_navigation_mode == 'roadline_center'), 'v40_central_roadline_name': v40_central_name, 'v40_central_roadline_coverage': float(v40_central_coverage), 'v40_native_pairing_quality': v40_native_pairing_quality, 'v40_roadline_native_fallback_to_v38': bool(v40_native_fallback), 'v40_frozen_navigation_preserved': True, 'v37_paired_cross_sections_enabled': bool(p.enable_v37_paired_cross_sections), 'v37_post_navigation_pairing': v37_post_pairing_summary, 'v36_geometric_roadline_authority_enabled': bool(p.enable_v36_geometric_roadline_authority), 'v35_native_roadline_center_enabled': bool(p.enable_v35_native_roadline_center), 'v35_native_roadline_center': v35_native_center_summary, 'v33_unified_roadline_roles': bool(p.enable_v33_physical_roadline_tracks), 'v33_physical_track_count': int(len(v33_physical_tracks)), 'v32_frozen_road_geometry': True, 'v32_auxiliary_controls_width': False, 'v32_auxiliary_controls_road_area': False, 'v32_authoritative_navigation': v32_navigation_summary, 'v31_direction_position_decoupled': True, 'v31_auxiliary_controls_width': False, 'v31_auxiliary_controls_road_area': False, 'v31_shape_only_moves_center': False, 'v31_navigation_direction': navigation_direction_summary, 'v29_terminal_direction_policy': 'road_or_trusted_auxiliary_direction_only', 'v29_terminal_allows_unlabelled_initial_gap': True, 'v29_cross_section_policy': 'continuous_tangent_then_boundary_pair_fallback', 'v28_road_reconstruction_base': 'v27_v23_frozen_road_v26_local_navigation_center', 'v28_centerline_policy': 'final_navigation_center_frozen_before_reconstruction', 'v28_state_routing': {'direct': 'direct_road_observation', 'internal': 'double_anchor_fragment_pair_or_gap', 'lateral': 'obstacle_supported_missing_side', 'terminal': 'obstacle_supported_final_navigation_march'}, 'v28_lateral_reconstruction': lateral_summary, 'v28_terminal_and_fragment_reconstruction': terminal_summary, 'v28_reconstruction_can_modify_centerline': False, 'v27_road_reconstruction_base': 'v23_auxiliary_segmentation_envelope', 'v27_centerline_policy': 'v26_local_only_no_post_guidance_reconstruction', 'v27_v24_aggressive_obstacle_reconstruction_enabled': bool(p.enable_v24_obstacle_driven_reconstruction), 'v27_v25_post_navigation_terminal_enabled': bool(p.enable_v25_navigation_guided_reconstruction), 'v28_frozen_center_state_reconstruction_enabled': bool(p.enable_v28_frozen_center_state_reconstruction), 'v22_authoritative_roadline_center_enabled': bool(p.enable_v22_authoritative_roadline_center), 'v22_authoritative_roadline_center_applied': bool(reference is not None and selected_role == 'confirmed_centerline' and (selected_family == 'road_line')), 'v12_base_center_preserved_without_auxiliary': bool(reference is None), 'auxiliary_changed_road_track_identity': False, 'auxiliary_participated_in_width_fit': False, 'auxiliary_changed_v12_boundaries': False, 'auxiliary_terminal_extension_enabled': bool(p.enable_v28_frozen_center_state_reconstruction), 'terminal_extension_count': int(bool(terminal_summary.get('lower_bands', 0) or terminal_summary.get('upper_bands', 0))), 'terminal_extension_bands': int(terminal_summary.get('lower_bands', 0) + terminal_summary.get('upper_bands', 0)), 'v12_obstacle_terminal_extrapolation_bands': int(sum((source == 'extrapolated' for source in base_sources))), 'v25_navigation_guided_terminal': terminal_summary, 'v15_auxiliary_track_summary': track_summary, 'associated_component_count': int(len(records)), 'v17_auxiliary_guided_geometry': bool(reference is not None), 'v17_guided_center_bands': int(len(guided_indices)), 'v18_auxiliary_curve_role': selected_role, 'v18_auxiliary_curve_source': str(reference.get('source_category', 'none')) if reference else 'none', 'v18_auxiliary_curve_selection_reason': str(reference.get('selection_reason', 'none')) if reference else 'none', 'v18_auxiliary_curve_coverage': float(reference.get('coverage', 0.0) if reference else 0.0), 'v30_auxiliary_consensus_track_count': int(reference.get('v30_consensus_track_count', 0) if reference else 0), 'v30_auxiliary_consensus_source_diversity': int(reference.get('v30_consensus_source_diversity', 0) if reference else 0), 'v30_auxiliary_consensus_union_coverage': float(reference.get('v30_consensus_union_coverage', 0.0) if reference else 0.0), 'v30_auxiliary_consensus_promoted': bool(reference.get('v30_consensus_promoted', False) if reference else False), 'v18_guardrail_curve_mode': 'shared_longitudinal_modal_ridge', 'paired_center_roadline_observations': paired_center_observations, 'single_center_roadline_observations': single_center_observations, 'center_fit_observation_bands': center_fit_observations, 'center_fit_weight': float(p.v13_roadline_center_fit_weight), 'central_target_observation_bands': central_target_observations, 'edge_shape_target_observation_bands': edge_shape_observations, 'roadline_center_mode': 'v22_roadline_authoritative_center' if selected_role == 'confirmed_centerline' else 'v30_consensus_auxiliary_shape_shifted' if selected_role == 'shape_shifted' else 'road_geometry_fallback', 'single_boundary_hint_observations': single_boundary_hints, 'long_single_boundary_anchor_bands': 0, 'long_single_boundary_components': 0, 'single_boundary_position_constraint_enabled': False, 'single_boundary_shape_only_components': shape_only_components, 'roadline_internal_bands': 0, 'roadline_internal_pixels': 0, 'roadline_support_pixels': int((selected_masks['road_line'] & analysis).sum()), 'auxiliary_supported_area_pixels': int(auxiliary_area.sum()), 'legacy_probe_paired_center_observations': paired_center_observations, 'v19_auxiliary_road_envelope': auxiliary_area_summary, 'v19_auxiliary_envelope_geometry': expansion_summary, 'v21_center_safety': center_safety}})
    return refresh_reconstruction_metadata(transformed)

# =============================================================================
# 16. 輔助方向覆蓋區的局部橫斷面模式
# =============================================================================
# 保留局部 spline／piecewise 橫斷面模式供相容與消融；主流程只有證據與
# 品質條件成立時才使用，否則維持全域角度的穩定橫斷面。
# 【品質工具】計算左右端點線段落在分析走廊內的比例。
def _section_inside_ratio(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    samples = max(3, int(math.ceil(float(np.linalg.norm(right - left)))) + 1)
    xs = np.rint(np.linspace(left[0], right[0], samples)).astype(np.int32)
    ys = np.rint(np.linspace(left[1], right[1], samples)).astype(np.int32)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not np.any(valid):
        return 0.0
    return float(mask[ys[valid], xs[valid]].mean())

# 【序列工具】回傳布林序列中每段連續 True 的起訖索引。
def _boolean_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs

# 【序列工具】補起長度不超過門檻的短 False 缺口。
def _close_short_boolean_gaps(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = np.asarray(values, dtype=bool).copy()
    maximum_gap = max(0, int(maximum_gap))
    if maximum_gap == 0:
        return result
    false_runs = _boolean_true_runs(~result)
    for start, end in false_runs:
        if start > 0 and end < len(result) and (end - start <= maximum_gap) and result[start - 1] and result[end]:
            result[start:end] = True
    return result

# 【歷史橫斷面】在可信輔助覆蓋區使用平滑局部切線建立橫斷面。
def apply_auxiliary_partial_spline_sections(centers: np.ndarray, left: np.ndarray, right: np.ndarray, station_sources: list[str], analysis_corridor: np.ndarray, auxiliary_summary: dict, base_angle_degrees: float, p: Parameters) -> tuple[np.ndarray, np.ndarray, dict] | None:
    centers = np.asarray(centers, dtype=np.float64)
    next_left = np.asarray(left, dtype=np.float64).copy()
    next_right = np.asarray(right, dtype=np.float64).copy()
    count = min(len(centers), len(next_left), len(next_right))
    evidence = [item for item in auxiliary_summary.get('evidence', []) if len(item.get('station_curve_xy', [])) >= 2 and item.get('source_family') in {'road_line', 'roadside_boundary'}]
    if count < 3 or not evidence:
        return None
    centers = centers[:count]
    next_left = next_left[:count]
    next_right = next_right[:count]
    angle_contributions: list[list[float]] = [[] for _ in range(count)]
    weight_contributions: list[list[float]] = [[] for _ in range(count)]
    family_contributions: list[set[str]] = [set() for _ in range(count)]
    accepted_curves: list[dict] = []
    tangent_radius = max(1, int(p.v16_tangent_window_stations))
    minimum_span = max(3, int(p.v16_min_curve_station_span))
    for item in evidence:
        observations = np.asarray(item.get('station_curve_xy', []), dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] < 3:
            continue
        station_indices = np.rint(np.clip(observations[:, 0], 0.0, 1.0) * max(1, count - 1)).astype(np.int32)
        points = observations[:, 1:3]
        unique_indices = np.unique(station_indices)
        if len(unique_indices) < 2:
            continue
        grouped_points = np.asarray([np.median(points[station_indices == index], axis=0) for index in unique_indices], dtype=np.float64)
        start = int(unique_indices[0])
        end = int(unique_indices[-1])
        if end - start + 1 < minimum_span:
            continue
        local_indices = np.arange(start, end + 1, dtype=np.int32)
        target_x = np.interp(local_indices, unique_indices, grouped_points[:, 0])
        target_y = np.interp(local_indices, unique_indices, grouped_points[:, 1])
        observation_weights = np.zeros(len(local_indices), dtype=np.float64)
        observation_weights[unique_indices - start] = 1.0
        fitted_x = regularized_curve_fit(target_x, observation_weights, float(p.v16_curve_smoothness), int(p.v16_curve_robust_iterations))
        fitted_y = regularized_curve_fit(target_y, observation_weights, float(p.v16_curve_smoothness), int(p.v16_curve_robust_iterations))
        fitted_curve = np.column_stack((fitted_x, fitted_y))
        quality = float(max(0.0, item.get('quality_weight', 0.0)))
        if quality <= 0:
            continue
        curve_angles: list[float] = []
        curve_stations: list[int] = []
        for local_position, station_index in enumerate(local_indices):
            before = max(0, local_position - tangent_radius)
            after = min(len(fitted_curve) - 1, local_position + tangent_radius)
            tangent = fitted_curve[after] - fitted_curve[before]
            if float(np.linalg.norm(tangent)) < 1e-06:
                continue
            angle = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 180.0)
            angle_contributions[int(station_index)].append(angle)
            weight_contributions[int(station_index)].append(quality)
            family_contributions[int(station_index)].add(str(item.get('source_family', 'auxiliary')))
            curve_angles.append(angle)
            curve_stations.append(int(station_index))
        if curve_stations:
            accepted_curves.append({'station_start': int(curve_stations[0]), 'station_end': int(curve_stations[-1]), 'source_family': str(item.get('source_family', 'auxiliary')), 'source_category': str(item.get('source_category', 'auxiliary')), 'quality_weight': quality})
    if not accepted_curves:
        return None
    local_angles = np.full(count, np.nan, dtype=np.float64)
    local_weights = np.zeros(count, dtype=np.float64)
    local_consistency = np.zeros(count, dtype=np.float64)
    trusted = np.zeros(count, dtype=bool)
    for index in range(count):
        if not angle_contributions[index]:
            continue
        angle, consistency = weighted_axial_mean_degrees(angle_contributions[index], weight_contributions[index])
        weight = float(sum(weight_contributions[index]))
        local_angles[index] = angle
        local_weights[index] = weight
        local_consistency[index] = float(consistency)
        trusted[index] = bool(weight >= float(p.v16_min_local_evidence_weight) and consistency >= float(p.v16_min_local_consistency))
    trusted = _close_short_boolean_gaps(trusted, max(1, int(p.v16_transition_stations) // 3))
    trusted_runs = [(start, end) for start, end in _boolean_true_runs(trusted) if end - start >= minimum_span]
    if not trusted_runs:
        return None
    transition = max(1, int(p.v16_transition_stations))
    blend_weights = np.zeros(count, dtype=np.float64)
    local_delta_targets = np.zeros(count, dtype=np.float64)
    segment_records: list[dict] = []
    for start, end in trusted_runs:
        core_indices = np.arange(start, end, dtype=np.int32)
        finite = core_indices[np.isfinite(local_angles[core_indices])]
        if len(finite) < 2:
            continue
        raw_deltas = np.asarray([signed_axial_angle_difference(local_angles[index], base_angle_degrees) for index in finite], dtype=np.float64)
        raw_deltas = np.clip(raw_deltas, -float(p.v16_max_global_delta_deg), float(p.v16_max_global_delta_deg))
        core_targets = np.interp(core_indices, finite, raw_deltas)
        core_weights = np.asarray([max(0.05, local_consistency[index]) if np.isfinite(local_angles[index]) else 0.1 for index in core_indices], dtype=np.float64)
        fitted_deltas = regularized_curve_fit(core_targets, core_weights, max(1.0, float(p.v16_curve_smoothness) * 0.35), 1)
        fitted_deltas = np.clip(fitted_deltas, -float(p.v16_max_global_delta_deg), float(p.v16_max_global_delta_deg))
        mean_consistency = float(np.mean(local_consistency[finite]))
        core_blend = float(p.v16_core_direction_blend) * float(np.clip(mean_consistency, float(p.v16_min_local_consistency), 1.0))
        for offset, index in enumerate(core_indices):
            if core_blend >= blend_weights[index]:
                blend_weights[index] = core_blend
                local_delta_targets[index] = fitted_deltas[offset]
        for distance in range(1, transition + 1):
            fraction = (transition - distance + 1) / (transition + 1)
            taper = fraction * fraction * (3.0 - 2.0 * fraction)
            before = start - distance
            after = end - 1 + distance
            transition_weight = core_blend * taper
            if before >= 0 and transition_weight > blend_weights[before]:
                blend_weights[before] = transition_weight
                local_delta_targets[before] = fitted_deltas[0]
            if after < count and transition_weight > blend_weights[after]:
                blend_weights[after] = transition_weight
                local_delta_targets[after] = fitted_deltas[-1]
        families = sorted(set().union(*family_contributions[start:end]))
        categories = sorted({curve['source_category'] for curve in accepted_curves if curve['station_end'] >= start and curve['station_start'] < end})
        segment_records.append({'station_start': int(start), 'station_end': int(end - 1), 'transition_start': int(max(0, start - transition)), 'transition_end': int(min(count - 1, end - 1 + transition)), 'mean_angle_delta_from_global_deg': float(np.mean(fitted_deltas)), 'maximum_angle_delta_from_global_deg': float(np.max(np.abs(fitted_deltas))), 'mean_evidence_weight': float(np.mean(local_weights[finite])), 'angle_consistency': mean_consistency, 'source_families': families, 'source_categories': categories, 'refined_sections': 0, 'rejected_sections': 0})
    if not segment_records:
        return None
    target_deltas = local_delta_targets * blend_weights
    active = blend_weights > 1e-06
    final_deltas = regularized_curve_fit(target_deltas, np.where(active, 1.0, 2.0), 1.5, 1)
    final_deltas[~active] = 0.0
    maximum_adjacent = float(p.v16_max_adjacent_delta_deg)
    for index in range(1, count):
        change = float(np.clip(final_deltas[index] - final_deltas[index - 1], -maximum_adjacent, maximum_adjacent))
        final_deltas[index] = final_deltas[index - 1] + change
    for index in range(count - 2, -1, -1):
        change = float(np.clip(final_deltas[index] - final_deltas[index + 1], -maximum_adjacent, maximum_adjacent))
        final_deltas[index] = final_deltas[index + 1] + change
    final_deltas = np.clip(final_deltas, -float(p.v16_max_global_delta_deg), float(p.v16_max_global_delta_deg))
    total_refined = total_rejected = 0
    for index in np.flatnonzero(active):
        if index >= len(station_sources) or station_sources[index] in {'extrapolated', 'auxiliary_supported_extrapolated'}:
            continue
        pair = next_right[index] - next_left[index]
        width = float(np.linalg.norm(pair))
        if width < p.min_road_width:
            continue
        angle = (float(base_angle_degrees) + float(final_deltas[index])) % 180.0
        theta = math.radians(angle)
        normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
        if float(np.dot(normal, pair)) < 0:
            normal = -normal
        candidate_left = centers[index] - normal * width / 2.0
        candidate_right = centers[index] + normal * width / 2.0
        inside_ratio = _section_inside_ratio(candidate_left, candidate_right, analysis_corridor)
        endpoint_shift = max(float(np.linalg.norm(candidate_left - next_left[index])), float(np.linalg.norm(candidate_right - next_right[index]))) / max(width, 1.0)
        accepted = bool(inside_ratio >= float(p.v16_min_section_inside_ratio) and endpoint_shift <= float(p.v16_max_endpoint_shift_fraction))
        for record in segment_records:
            if record['transition_start'] <= index <= record['transition_end']:
                key = 'refined_sections' if accepted else 'rejected_sections'
                record[key] += 1
        if not accepted:
            total_rejected += 1
            continue
        next_left[index] = candidate_left
        next_right[index] = candidate_right
        total_refined += 1
    output = {'status': 'piecewise_applied' if total_refined else 'associated_but_geometry_rejected', 'mode': 'auxiliary_partial_spline_local_tangent' if total_refined else 'global_angle_cross_sections', 'segments': segment_records, 'refined_sections': int(total_refined), 'rejected_sections': int(total_rejected), 'terminal_extension_used': False, 'accepted_curve_count': int(len(accepted_curves)), 'trusted_core_sections': int(sum((end - start for start, end in trusted_runs))), 'transition_sections': int(np.count_nonzero(active & ~trusted))}
    return (next_left, next_right, output)

# 【歷史橫斷面】以少量長分段共用角度修正橫斷方向。
def apply_auxiliary_piecewise_sections(centers: np.ndarray, left: np.ndarray, right: np.ndarray, station_sources: list[str], analysis_corridor: np.ndarray, auxiliary_summary: dict, base_angle_degrees: float, p: Parameters) -> tuple[np.ndarray, np.ndarray, dict]:
    output = {'status': auxiliary_summary.get('status', 'no_masks'), 'mode': 'global_angle_cross_sections', 'segments': [], 'refined_sections': 0, 'rejected_sections': 0, 'terminal_extension_used': False}
    evidence = list(auxiliary_summary.get('evidence', []))
    if not evidence or len(centers) < 2:
        return (left, right, output)
    if bool(p.enable_v16_partial_auxiliary_spline):
        partial_spline = apply_auxiliary_partial_spline_sections(centers, left, right, station_sources, analysis_corridor, auxiliary_summary, base_angle_degrees, p)
        if partial_spline is not None:
            return partial_spline
    next_left = np.asarray(left, dtype=np.float64).copy()
    next_right = np.asarray(right, dtype=np.float64).copy()
    centers = np.asarray(centers, dtype=np.float64)
    segment_size = max(3, int(p.auxiliary_segment_stations))
    previous_angle = float(base_angle_degrees) % 180.0
    total_refined = total_rejected = 0
    segments: list[dict] = []
    for start in range(0, len(centers), segment_size):
        end = min(len(centers), start + segment_size)
        segment_start_fraction = float(start / max(1, len(centers) - 1))
        segment_end_fraction = float((end - 1) / max(1, len(centers) - 1))
        active = [item for item in evidence if float(item['station_end_fraction']) >= segment_start_fraction and float(item['station_start_fraction']) <= segment_end_fraction]
        evidence_weight = float(sum((item['quality_weight'] for item in active)))
        if evidence_weight < float(p.auxiliary_min_segment_evidence_weight):
            continue
        auxiliary_angle, auxiliary_consistency = weighted_axial_mean_degrees([float(item['angle_degrees']) for item in active], [float(item['quality_weight']) for item in active])
        window_start = max(0, start - segment_size)
        window_end = min(len(centers), end + segment_size)
        road_geometry = axial_angle_from_points(centers[window_start:window_end])
        road_local_angle = float(road_geometry['angle_degrees']) if road_geometry is not None else float(base_angle_degrees)
        road_aux_delta = angular_distance_degrees(road_local_angle, auxiliary_angle)
        joint_local_support = bool(road_geometry is not None and road_aux_delta <= float(p.auxiliary_joint_local_max_disagreement_deg))
        local_angles = [float(item['angle_degrees']) for item in active]
        local_weights = [float(item['quality_weight']) for item in active]
        if joint_local_support:
            local_angles.append(road_local_angle)
            local_weights.append(float(p.auxiliary_joint_local_road_weight))
        local_angle, local_consistency = weighted_axial_mean_degrees(local_angles, local_weights)
        fused, consistency = weighted_axial_mean_degrees([float(base_angle_degrees), local_angle], [float(p.auxiliary_global_prior_weight), max(evidence_weight, 1e-06)])
        maximum_global_delta = float(p.auxiliary_joint_local_max_global_delta_deg) if joint_local_support else float(p.auxiliary_max_segment_delta_deg)
        base_delta = signed_axial_angle_difference(fused, base_angle_degrees)
        base_delta = float(np.clip(base_delta, -maximum_global_delta, maximum_global_delta))
        fused = (float(base_angle_degrees) + base_delta) % 180.0
        adjacent_delta = signed_axial_angle_difference(fused, previous_angle)
        adjacent_delta = float(np.clip(adjacent_delta, -float(p.auxiliary_max_adjacent_delta_deg), float(p.auxiliary_max_adjacent_delta_deg)))
        fused = (previous_angle + adjacent_delta) % 180.0
        theta = math.radians(fused)
        normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
        refined_here = rejected_here = 0
        for index in range(start, end):
            if index >= len(station_sources) or station_sources[index] in {'extrapolated', 'auxiliary_supported_extrapolated'}:
                continue
            pair = next_right[index] - next_left[index]
            width = float(np.linalg.norm(pair))
            if width < p.min_road_width:
                continue
            if float(np.dot(normal, pair)) < 0:
                section_normal = -normal
            else:
                section_normal = normal
            candidate_left = centers[index] - section_normal * width / 2.0
            candidate_right = centers[index] + section_normal * width / 2.0
            inside_ratio = _section_inside_ratio(candidate_left, candidate_right, analysis_corridor)
            endpoint_shift = max(float(np.linalg.norm(candidate_left - next_left[index])), float(np.linalg.norm(candidate_right - next_right[index]))) / max(width, 1.0)
            if inside_ratio < float(p.auxiliary_min_section_inside_ratio) or endpoint_shift > 0.4:
                rejected_here += 1
                continue
            next_left[index] = candidate_left
            next_right[index] = candidate_right
            refined_here += 1
        if refined_here:
            previous_angle = fused
        total_refined += refined_here
        total_rejected += rejected_here
        segments.append({'station_start': int(start), 'station_end': int(end - 1), 'angle_degrees': float(fused), 'angle_delta_from_global_deg': float(signed_axial_angle_difference(fused, base_angle_degrees)), 'evidence_weight': evidence_weight, 'angle_consistency': float(consistency), 'auxiliary_angle_degrees': float(auxiliary_angle), 'auxiliary_angle_consistency': float(auxiliary_consistency), 'road_local_angle_degrees': float(road_local_angle), 'road_auxiliary_angle_delta_deg': float(road_aux_delta), 'joint_local_support': bool(joint_local_support), 'joint_local_angle_consistency': float(local_consistency), 'source_families': sorted({str(item['source_family']) for item in active}), 'refined_sections': int(refined_here), 'rejected_sections': int(rejected_here)})
    if total_refined:
        output.update({'status': 'piecewise_applied', 'mode': 'auxiliary_piecewise_global_cross_sections', 'segments': segments, 'refined_sections': int(total_refined), 'rejected_sections': int(total_rejected)})
    elif evidence:
        output.update({'status': 'associated_but_geometry_rejected', 'segments': segments, 'rejected_sections': int(total_rejected)})
    return (next_left, next_right, output)

# =============================================================================
# 17. 多角度、多軌跡候選與主要道路選擇
# =============================================================================
# 對連續角度與多個種子建立道路候選，綜合 Road 支持、追蹤長度、中心及
# 寬度平順性、缺口與輔助方向先驗評分，選出同一條主要道路。
# 【候選評分】評估逐帶軌跡的 Road 支持、連續性、平順性與方向先驗。
def observation_track_score(observations: list[dict], frame_shape: tuple[int, int], angle_degrees: float, preferred_angle: float, p: Parameters, auxiliary_prior: float=0.0) -> float:
    if len(observations) < 2:
        return -1000000000.0
    bands = np.asarray([item['band'] for item in observations], dtype=np.float64)
    centers = np.asarray([(item['left_x'] + item['right_x']) / 2.0 for item in observations], dtype=np.float64)
    widths = np.asarray([item['right_x'] - item['left_x'] + 1.0 for item in observations], dtype=np.float64)
    band_span = max(1.0, float(bands[-1] - bands[0] + 1.0))
    coverage = min(1.0, band_span * p.row_band_height / max(1, frame_shape[0]))
    density = min(1.0, len(observations) / band_span)
    longitudinal_span = band_span * p.row_band_height
    aspect = min(3.0, longitudinal_span / max(float(np.median(widths)), 1.0)) / 3.0
    width_cv = float(np.std(widths) / max(float(np.mean(widths)), 1.0))
    quarter = max(1, len(widths) // 4)
    endpoint_taper = min(1.0, abs(float(np.median(widths[:quarter])) - float(np.median(widths[-quarter:]))) / max(float(np.median(widths)), 1.0))
    if len(bands) >= 3:
        fitted = np.polyval(np.polyfit(bands, centers, 1), bands)
        center_residual = min(1.0, float(np.median(np.abs(centers - fitted))) / max(float(np.median(widths)), 1.0))
    else:
        center_residual = 0.0
    direction_prior = max(0.0, 1.0 - angular_distance_degrees(angle_degrees, preferred_angle) / 90.0)
    return float(1.8 * coverage + 0.8 * density + 0.55 * aspect + 0.55 * endpoint_taper + 1.6 * direction_prior + float(p.auxiliary_initial_coarse_bonus) * float(auxiliary_prior) - 0.35 * width_cv - 0.45 * center_residual)

# 【座標工具】把道路座標中的走廊、遮罩與診斷資料轉回影像座標。
def transform_corridor_from_road_coordinates(corridor: dict, frame: dict, road_mask: np.ndarray, obstacles: np.ndarray, p: Parameters) -> dict:
    if not corridor.get('ok'):
        return corridor
    transformed = dict(corridor)
    for key in ('raw_centerline', 'centerline', 'raw_left', 'raw_right', 'left', 'right'):
        transformed[key] = points_from_road_coordinates(corridor[key], frame)
    transformed_geometry = []
    for item in corridor.get('one_sided_geometry', []):
        converted = dict(item)
        for key in ('centerline', 'left', 'right'):
            converted[key] = points_from_road_coordinates(np.asarray(item.get(key, []), dtype=np.float64), frame).tolist()
        transformed_geometry.append(converted)
    transformed['one_sided_geometry'] = transformed_geometry
    transformed['selected_direction_angle_deg'] = float(frame['angle_degrees'])
    transformed['observation_mode'] = 'continuous_angle_tracked'
    geometry_mode = str(p.direction_geometry_mode)
    if geometry_mode == 'local_tangent_cross_sections':
        transformed = local_tangent_refine_corridor(transformed, road_mask, obstacles, p)
        cross_section_method = 'local_tangent_paired_boundary_correspondence'
    elif geometry_mode == 'global_angle_cross_sections':
        transformed['local_direction_refined'] = False
        transformed['local_direction_refined_sections'] = 0
        cross_section_method = 'global_angle_paired_boundary_correspondence'
    else:
        raise ValueError(f'unsupported direction geometry mode: {geometry_mode}')
    transformed['direction_geometry_mode'] = geometry_mode
    transformed['cross_section_method'] = cross_section_method
    left = np.asarray(transformed['left'], dtype=np.float64)
    right = np.asarray(transformed['right'], dtype=np.float64)
    analysis = corridor_polygon_mask(road_mask.shape, left, right)
    internal_gap = mask_from_boundary_sources(road_mask.shape, left, right, list(transformed.get('band_source_types', [])), {'interpolated'} if p.include_unknown_internal_gaps else set(), line_width=max(2, int(p.row_band_height))) & ~road_mask
    inferred_occlusion = analysis & obstacles & ~road_mask
    if p.enable_v24_obstacle_driven_reconstruction:
        obstacle_supported = mask_from_boundary_sources(road_mask.shape, left, right, list(transformed.get('band_source_types', [])), {'interior_occlusion_inferred', 'one_side_inferred', 'extrapolated'}, line_width=max(2, int(p.row_band_height))) & analysis & ~road_mask
    else:
        obstacle_supported = np.zeros_like(road_mask, dtype=bool)
    full = road_mask | internal_gap | inferred_occlusion | obstacle_supported
    transformed['analysis_geometry_corridor'] = analysis
    transformed['inferred_geometry_envelope'] = analysis
    transformed['inferred_internal_gap_mask'] = internal_gap
    transformed['inferred_occlusion_mask'] = inferred_occlusion
    transformed['obstacle_supported_reconstruction_mask'] = obstacle_supported
    transformed['full_road_corridor'] = full
    transformed['corridor_mask'] = full
    return transformed

# 【候選評分】綜合粗篩分數與完整重建品質，計算主要道路最終分數。
def final_track_score(corridor: dict, diagonal: float, coarse_score: float) -> float:
    if not corridor.get('ok'):
        return -1000000000.0
    centerline = np.asarray(corridor['centerline'], dtype=np.float64)
    length_score = min(1.5, polyline_length(centerline) / max(diagonal, 1.0))
    source_counts = corridor.get('source_counts', {})
    span = max(1, int(corridor.get('span_bands', len(centerline))))
    direct_ratio = float(source_counts.get('direct', 0)) / span
    inference_penalty = 0.25 * float(corridor.get('interpolation_ratio', 0.0)) + 0.45 * float(corridor.get('extrapolation_ratio', 0.0))
    if len(centerline) >= 3:
        steps = np.diff(centerline, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        unit = steps / np.maximum(norms[:, None], 1e-06)
        curvature = float(np.mean(np.linalg.norm(np.diff(unit, axis=0), axis=1)))
        tangents = np.gradient(centerline, axis=0)
        cross_vectors = np.asarray(corridor['right'], dtype=np.float64) - np.asarray(corridor['left'], dtype=np.float64)
        tangent_norms = np.linalg.norm(tangents, axis=1)
        cross_norms = np.linalg.norm(cross_vectors, axis=1)
        orthogonality_error = float(np.median(np.abs(np.sum(tangents * cross_vectors, axis=1) / np.maximum(tangent_norms * cross_norms, 1e-06))))
    else:
        curvature = 1.0
        orthogonality_error = 1.0
    return float(coarse_score + 1.15 * length_score + 0.75 * direct_ratio - inference_penalty - 0.35 * min(1.0, curvature) - 1.8 * min(1.0, orthogonality_error))

# 【方向工具】由 Road 主成分及跨度選出連續角度搜尋的起始提示。
def continuous_angle_hint(road_mask: np.ndarray, orientation_evidence: dict, p: Parameters) -> float:
    base_angle = 0.0 if orientation_evidence.get('preferred_axis') == 'horizontal_x' else 90.0
    frame = road_coordinate_frame(road_mask.shape, base_angle)
    oriented = mask_to_road_coordinates(road_mask, frame)
    observations = tracked_road_band_observations(oriented, p, seed_rank=0)
    if len(observations) < 2:
        return base_angle
    band_span = max(1.0, float(observations[-1]['band'] - observations[0]['band'] + 1))
    longitudinal_span = band_span * max(2, int(p.row_band_height))
    median_width = float(np.median([item['right_x'] - item['left_x'] + 1.0 for item in observations]))
    observation_density = len(observations) / band_span
    if longitudinal_span / max(median_width, 1.0) < 2.25 or observation_density < 0.55:
        return base_angle
    oriented_centers = np.asarray([[(item['left_x'] + item['right_x']) / 2.0, item['y']] for item in observations], dtype=np.float64)
    points = points_from_road_coordinates(oriented_centers, frame)
    centered = points - np.median(points, axis=0, keepdims=True)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    if np.linalg.norm(tangent) < 1e-06:
        return base_angle
    return float(math.degrees(math.atan2(tangent[1], tangent[0])) % 180.0)

def continuous_direction_corridor(road_mask: np.ndarray, obstacles: np.ndarray, orientation_evidence: dict, p: Parameters, auxiliary_masks: dict[str, np.ndarray] | None=None) -> dict:
    # 【主要功能：從多角度、多種子候選中選出主要道路】
    # 先在縮圖快速評分角度與軌跡，再於分析解析度重建前幾名；比較 Road
    # 支持、連續長度、中心／寬度平順性、缺口及輔助方向後回傳最佳候選。
    height, width = road_mask.shape
    long_edge = max(height, width)
    scale = min(1.0, float(p.direction_coarse_long_edge) / max(1, long_edge))
    coarse_road = resize_boolean_mask(road_mask, scale)
    coarse_obstacles = resize_boolean_mask(obstacles, scale)
    preferred_angle = continuous_angle_hint(coarse_road, orientation_evidence, p)
    auxiliary_hypotheses = independent_auxiliary_direction_hypotheses(auxiliary_masks, road_mask.shape, p, road_mask=road_mask) if p.enable_auxiliary_direction_evidence else []
    has_paired_side_geometry = any(('roadside_pair_center' in item.get('source_families', []) for item in auxiliary_hypotheses))
    for hypothesis in auxiliary_hypotheses:
        families = set(hypothesis.get('source_families', []))
        if 'roadside_pair_center' in families:
            delta = angular_distance_degrees(float(hypothesis['angle_degrees']), preferred_angle)
            if delta > 45.0:
                hypothesis['confidence'] = float(0.25 * float(hypothesis['confidence']))
        if has_paired_side_geometry and 'roadside_boundary' in families and ('road_line' not in families):
            hypothesis['confidence'] = float(0.35 * float(hypothesis['confidence']))
    auxiliary_hypotheses.sort(key=lambda item: (item.get('confidence', 0.0), item.get('support_weight', 0.0)), reverse=True)
    diagonal = max(1.0, math.hypot(height, width))
    global_auxiliary_hypotheses: list[dict] = []
    for hypothesis in auxiliary_hypotheses:
        span_ratio = float(hypothesis.get('longitudinal_union_coverage', float(hypothesis.get('maximum_span_px', 0.0)) / diagonal))
        families = set(hypothesis.get('source_families', []))
        minimum_span_ratio = float(p.v21_global_roadline_min_span_diagonal_ratio if 'road_line' in families else p.v21_global_aux_min_span_diagonal_ratio)
        eligible = bool(float(hypothesis.get('confidence', 0.0)) >= float(p.v21_global_aux_min_confidence) and float(hypothesis.get('angle_consistency', 0.0)) >= float(p.v21_global_aux_min_angle_consistency) and (span_ratio >= minimum_span_ratio))
        if bool(p.enable_v31_multi_aux_pretrack_direction) and bool(hypothesis.get('v31_strong_consensus', False)):
            eligible = bool(float(hypothesis.get('angle_consistency', 0.0)) >= 0.8 and span_ratio >= float(p.v31_min_union_coverage))
        hypothesis['global_selection_span_ratio'] = span_ratio
        hypothesis['global_selection_min_span_ratio'] = minimum_span_ratio
        hypothesis['global_selection_eligible'] = eligible
        if eligible:
            global_auxiliary_hypotheses.append(hypothesis)
    step = max(3.0, float(p.direction_angle_step_degrees))
    angles = list(np.arange(0.0, 180.0, step))
    angles.extend([preferred_angle] + [float(item['angle_degrees']) for item in global_auxiliary_hypotheses])
    angles = sorted({round(float(value) % 180.0, 6) for value in angles})
    coarse_candidates: list[dict] = []
    for angle in angles:
        auxiliary_prior = auxiliary_hypothesis_prior(angle, global_auxiliary_hypotheses, float(p.auxiliary_initial_candidate_tolerance_deg))
        frame = road_coordinate_frame(coarse_road.shape, angle)
        oriented_road = mask_to_road_coordinates(coarse_road, frame)
        _ = coarse_obstacles
        for seed_rank in range(max(1, int(p.direction_seed_tracks_per_angle))):
            observations = tracked_road_band_observations(oriented_road, p, seed_rank=seed_rank)
            score = observation_track_score(observations, frame['shape'], angle, preferred_angle, p, auxiliary_prior=0.0)
            if score <= -100000000.0:
                continue
            coarse_candidates.append({'angle_degrees': angle, 'seed_rank': seed_rank, 'coarse_score': score, 'auxiliary_prior_score': float(auxiliary_prior)})
    if not coarse_candidates:
        return {'ok': False, 'reason': 'no_direction_track_candidates'}
    coarse_candidates.sort(key=lambda item: item['coarse_score'], reverse=True)
    shortlisted = list(coarse_candidates[:max(1, int(p.direction_top_k))])
    preferred_candidate = min(coarse_candidates, key=lambda item: (angular_distance_degrees(item['angle_degrees'], preferred_angle), -item['coarse_score']))
    if preferred_candidate not in shortlisted:
        shortlisted.append(preferred_candidate)
    for hypothesis in global_auxiliary_hypotheses:
        hypothesis_angle = float(hypothesis['angle_degrees'])
        auxiliary_candidate = min(coarse_candidates, key=lambda item: (angular_distance_degrees(float(item['angle_degrees']), hypothesis_angle), -float(item['coarse_score'])))
        if auxiliary_candidate not in shortlisted:
            shortlisted.append(auxiliary_candidate)
    unique_shortlisted: list[dict] = []
    seen_shortlisted: set[tuple[float, int]] = set()
    for candidate in shortlisted:
        key = (float(candidate['angle_degrees']), int(candidate['seed_rank']))
        if key in seen_shortlisted:
            continue
        seen_shortlisted.add(key)
        unique_shortlisted.append(candidate)
    shortlisted = unique_shortlisted
    evaluated: list[tuple[float, dict, dict]] = []
    full_frames: dict[float, tuple[dict, np.ndarray, np.ndarray]] = {}
    for track_id, candidate in enumerate(shortlisted):
        angle = float(candidate['angle_degrees'])
        if angle not in full_frames:
            frame = road_coordinate_frame(road_mask.shape, angle)
            full_frames[angle] = (frame, mask_to_road_coordinates(road_mask, frame), mask_to_road_coordinates(obstacles, frame))
        frame, oriented_road, oriented_obstacles = full_frames[angle]
        rebuilt = reconstruct_road_corridor(oriented_road, oriented_obstacles, p, observation_mode='tracked', seed_rank=int(candidate['seed_rank']), track_id=track_id)
        if not rebuilt.get('ok'):
            continue
        rebuilt = transform_corridor_from_road_coordinates(rebuilt, frame, road_mask, obstacles, p)
        score = final_track_score(rebuilt, math.hypot(height, width), float(candidate['coarse_score']))
        geometry_score = float(score)
        reliable_auxiliary_hypothesis = max((float(item.get('confidence', 0.0)) for item in global_auxiliary_hypotheses), default=0.0) >= float(p.v21_global_aux_min_confidence)
        road_prior_weight = 0.85 if reliable_auxiliary_hypothesis else 2.5
        road_direction_prior = max(0.0, 1.0 - angular_distance_degrees(angle, preferred_angle) / 90.0)
        score += road_prior_weight * road_direction_prior
        associated = associate_auxiliary_direction_evidence(auxiliary_masks, rebuilt, road_mask.shape, p) if auxiliary_hypotheses else {'status': 'no_masks', 'evidence': [], 'source_counts': {}}
        selection_family_weights = {'road_line': 1.0, 'roadside_boundary': 0.45, 'cone_sequence': 0.35}
        associated_quality = float(sum((float(item.get('quality_weight', 0.0)) * float(selection_family_weights.get(str(item.get('source_family', '')), 0.35)) for item in associated.get('evidence', []))))
        associated_quality_score = float(np.clip(associated_quality, 0.0, 1.0))
        auxiliary_prior_score = float(candidate.get('auxiliary_prior_score', 0.0))
        matching_strong_hypotheses = [item for item in global_auxiliary_hypotheses if bool(item.get('v31_strong_consensus', False)) and angular_distance_degrees(angle, float(item.get('angle_degrees', angle))) <= float(p.v31_consensus_angle_deg)]
        v31_independent_consensus = float(max((float(item.get('confidence', 0.0)) * float(item.get('angle_consistency', 0.0)) * min(1.0, float(item.get('longitudinal_union_coverage', 0.0)) / max(float(p.v31_min_union_coverage), 1e-06)) for item in matching_strong_hypotheses), default=0.0))
        paired_center_prior_score = auxiliary_hypothesis_prior(angle, [item for item in global_auxiliary_hypotheses if 'roadside_pair_center' in item.get('source_families', []) and angular_distance_degrees(float(item['angle_degrees']), preferred_angle) <= 35.0], 35.0)
        eligible_associated_quality = associated_quality_score if auxiliary_prior_score > 0.0 else 0.0
        if bool(p.enable_v31_multi_aux_pretrack_direction) and v31_independent_consensus > 0.0:
            joint_auxiliary_score = float(np.clip(0.55 * v31_independent_consensus + 0.25 * auxiliary_prior_score + 0.2 * eligible_associated_quality, 0.0, 1.0))
        else:
            joint_auxiliary_score = float(np.clip(0.2 * auxiliary_prior_score + 0.8 * eligible_associated_quality, 0.0, 1.0))
        associated_source_count = len(associated.get('source_counts', {}))
        spatially_confirmed_paired_score = float(paired_center_prior_score * associated_quality_score)
        unassociated_auxiliary_penalty = float(0.9 * auxiliary_prior_score if auxiliary_prior_score >= 0.5 and associated_quality_score <= 1e-06 else 0.0)
        score += float(p.v21_global_aux_final_bonus) * joint_auxiliary_score + float(p.v31_pretrack_prior_weight) * v31_independent_consensus + 0.25 * min(2, associated_source_count) + 1.25 * spatially_confirmed_paired_score - unassociated_auxiliary_penalty
        score -= 0.25 * int(candidate['seed_rank'])
        summary = {'track_id': int(track_id), 'angle_degrees': angle, 'seed_rank': int(candidate['seed_rank']), 'coarse_score': float(candidate['coarse_score']), 'final_score': float(score), 'geometry_score': geometry_score, 'centerline_length_px': polyline_length(rebuilt['centerline']), 'observed_bands': int(rebuilt.get('observed_bands', 0)), 'span_bands': int(rebuilt.get('span_bands', 0)), 'road_direction_prior': float(road_direction_prior), 'auxiliary_prior_score': auxiliary_prior_score, 'auxiliary_associated_quality': associated_quality, 'auxiliary_associated_source_count': int(associated_source_count), 'joint_auxiliary_score': joint_auxiliary_score, 'v31_independent_consensus': v31_independent_consensus, 'v31_strong_hypothesis_count': int(len(matching_strong_hypotheses)), 'paired_center_prior_score': float(paired_center_prior_score), 'spatially_confirmed_paired_score': spatially_confirmed_paired_score, 'unassociated_auxiliary_penalty': unassociated_auxiliary_penalty, 'auxiliary_status': str(associated.get('status', 'no_masks'))}
        evaluated.append((score, rebuilt, summary))
    if not evaluated:
        return {'ok': False, 'reason': 'direction_track_reconstruction_failed'}
    evaluated.sort(key=lambda item: item[0], reverse=True)
    geometry_best_index = max(range(len(evaluated)), key=lambda index: float(evaluated[index][2].get('geometry_score', -float('inf'))))
    geometry_best = evaluated[geometry_best_index]
    selected_geometry_score = float(evaluated[0][2].get('geometry_score', -float('inf')))
    best_geometry_score = float(geometry_best[2].get('geometry_score', -float('inf')))
    if best_geometry_score - selected_geometry_score > float(p.v31_pretrack_max_geometry_deficit if bool(p.enable_v31_multi_aux_pretrack_direction) and float(evaluated[0][2].get('v31_independent_consensus', 0.0)) > 0.0 else p.v21_global_aux_max_road_score_deficit) and angular_distance_degrees(float(evaluated[0][2].get('angle_degrees', 0.0)), float(geometry_best[2].get('angle_degrees', 0.0))) >= 15.0 and (float(evaluated[0][2].get('spatially_confirmed_paired_score', 0.0)) <= 0.0) and (float(evaluated[0][2].get('joint_auxiliary_score', 0.0)) < float(p.v30_strong_joint_direction_score)) and (float(evaluated[0][2].get('v31_independent_consensus', 0.0)) <= 0.0):
        evaluated.insert(0, evaluated.pop(geometry_best_index))
        evaluated[0][1]['axis_selection_reason'] = 'road_geometry_guarded_auxiliary_selection'
    preferred_index = min(range(len(evaluated)), key=lambda index: angular_distance_degrees(evaluated[index][2]['angle_degrees'], preferred_angle))
    preferred_evaluation = evaluated[preferred_index]
    selected_joint = float(evaluated[0][2].get('joint_auxiliary_score', 0.0))
    preferred_joint = float(preferred_evaluation[2].get('joint_auxiliary_score', 0.0))
    if evaluated[0][0] - preferred_evaluation[0] <= 0.25 and selected_joint <= preferred_joint + 0.05 and (float(evaluated[0][2].get('v31_independent_consensus', 0.0)) <= float(preferred_evaluation[2].get('v31_independent_consensus', 0.0)) + 0.05):
        evaluated.insert(0, evaluated.pop(preferred_index))
    best_score, best, _ = evaluated[0]
    best = dict(best)
    best['track_hypotheses'] = [item[2] for item in evaluated]
    best['track_score_margin'] = float(best_score - evaluated[1][0] if len(evaluated) > 1 else best_score)
    best['selected_track_id'] = int(best.get('track_id', 0))
    selected_summary = evaluated[0][2]
    best['road_direction_hint_deg'] = float(preferred_angle)
    best['auxiliary_direction_hypotheses'] = auxiliary_hypotheses
    best['selected_joint_auxiliary_score'] = float(selected_summary.get('joint_auxiliary_score', 0.0))
    best['axis_selection_reason'] = str(best.get('axis_selection_reason') or ('road_auxiliary_joint_multi_track_selection' if global_auxiliary_hypotheses else 'continuous_angle_multi_track_selection'))
    return best

# =============================================================================
# 18. 橫斷面、自由通道、阻塞率與判定信心
# =============================================================================
# 將最終左右邊界轉成連續橫斷面，扣除道路內障礙物後取得最大連續自由
# 寬度，計算阻塞率、連續阻塞區段與來源信心。
# 【阻塞工具】找出橫斷面上未被障礙物占用的連續像素區間。
def free_runs(blocked: np.ndarray) -> list[tuple[int, int]]:
    return contiguous_true_runs(~np.asarray(blocked, dtype=bool))

# 【阻塞工具】將自由區間轉成像素索引、端點與寬度等診斷資訊。
def describe_free_intervals(blocked: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> list[dict]:
    count = len(blocked)
    denominator = max(1, count - 1)
    intervals = []
    for start, end in free_runs(blocked):
        width = end - start + 1
        intervals.append({'start_index': int(start), 'end_index': int(end), 'width_px': int(width), 'width_ratio': float(width / max(1, count)), 'start_fraction': float(start / denominator), 'end_fraction': float(end / denominator), 'center_fraction': float((start + end) / 2.0 / denominator), 'left_x': int(xs[start]), 'left_y': int(ys[start]), 'right_x': int(xs[end]), 'right_y': int(ys[end])})
    return intervals

# 【橫斷面】直接連接一組已配對左右邊界，計算完整寬度與阻塞率。
def cross_section_between_boundaries(center: np.ndarray, left: np.ndarray, right: np.ndarray, obstacle_mask: np.ndarray, station_index: int, min_width: int, source_type: str='direct') -> CrossSection | None:
    height, width = obstacle_mask.shape
    vector = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    distance = float(np.linalg.norm(vector))
    if distance + 1.0 < min_width:
        return None
    sample_count = max(2, int(math.ceil(distance)) + 1)
    xs = np.rint(np.linspace(left[0], right[0], sample_count)).astype(np.int32)
    ys = np.rint(np.linspace(left[1], right[1], sample_count)).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = (xs[valid], ys[valid])
    if not len(xs):
        return None
    keep = np.ones(len(xs), dtype=bool)
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    xs, ys = (xs[keep], ys[keep])
    road_width = len(xs)
    if road_width < min_width:
        return None
    blocked = obstacle_mask[ys, xs]
    intervals = describe_free_intervals(blocked, xs, ys)
    free_width = max((item['width_px'] for item in intervals), default=0)
    free_ratio = free_width / road_width
    return CrossSection(station_index=station_index, center_x=float(center[0]), center_y=float(center[1]), left_x=int(xs[0]), left_y=int(ys[0]), right_x=int(xs[-1]), right_y=int(ys[-1]), road_width_px=int(road_width), max_free_width_px=int(free_width), free_width_ratio=float(free_ratio), blockage_ratio=float(1.0 - free_ratio), free_intervals=intervals, source_type=source_type, source_weight=float(SOURCE_WEIGHTS.get(source_type, 0.35)))

# 【橫斷面】由中心與切線建立法向量，搜尋走廊左右交點並量測阻塞。
def cross_section_at(center: np.ndarray, tangent: np.ndarray, road_mask: np.ndarray, obstacle_mask: np.ndarray, station_index: int, max_gap: int, obstacle_link_gap: int, min_width: int, expand_with_obstacles: bool=True) -> CrossSection | None:
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-06:
        return None
    tx, ty = tangent / norm
    normal = np.asarray([-ty, tx], dtype=np.float64)
    height, width = road_mask.shape
    max_distance = int(math.ceil(math.hypot(height, width)))
    offsets = np.arange(-max_distance, max_distance + 1, dtype=np.float64)
    xs = np.rint(center[0] + offsets * normal[0]).astype(np.int32)
    ys = np.rint(center[1] + offsets * normal[1]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, offsets = (xs[valid], ys[valid], offsets[valid])
    if len(xs) == 0:
        return None
    keep = np.ones(len(xs), dtype=bool)
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    xs, ys, offsets = (xs[keep], ys[keep], offsets[keep])
    road_line = road_mask[ys, xs]
    obstacle_line = obstacle_mask[ys, xs]
    road_indices = np.flatnonzero(road_line)
    if len(road_indices) == 0:
        return None
    center_index = int(np.argmin(np.abs(offsets)))
    anchor = int(road_indices[np.argmin(np.abs(road_indices - center_index))])
    allowed_anchor_gap = obstacle_link_gap + 2 if expand_with_obstacles else max_gap + 2
    if abs(anchor - center_index) > allowed_anchor_gap:
        return None

    def expand_through_evidence(evidence: np.ndarray, start: int, allowed_gap: int) -> tuple[int, int]:
        left = start
        gap = 0
        for index in range(start - 1, -1, -1):
            if evidence[index]:
                left, gap = (index, 0)
            else:
                gap += 1
                if gap > allowed_gap:
                    break
        right = start
        gap = 0
        for index in range(start + 1, len(evidence)):
            if evidence[index]:
                right, gap = (index, 0)
            else:
                gap += 1
                if gap > allowed_gap:
                    break
        return (left, right)
    base_left, base_right = expand_through_evidence(road_line, anchor, max_gap)
    if expand_with_obstacles:
        evidence = road_line | obstacle_line
        left, _ = expand_through_evidence(evidence, base_left, obstacle_link_gap)
        _, right = expand_through_evidence(evidence, base_right, obstacle_link_gap)
    else:
        left, right = (base_left, base_right)
    road_width = right - left + 1
    if road_width < min_width:
        return None
    blocked = obstacle_line[left:right + 1]
    intervals = describe_free_intervals(blocked, xs[left:right + 1], ys[left:right + 1])
    free_width = max((item['width_px'] for item in intervals), default=0)
    free_ratio = free_width / road_width
    blockage = 1.0 - free_ratio
    return CrossSection(station_index=station_index, center_x=float(center[0]), center_y=float(center[1]), left_x=int(xs[left]), left_y=int(ys[left]), right_x=int(xs[right]), right_y=int(ys[right]), road_width_px=int(road_width), max_free_width_px=int(free_width), free_width_ratio=float(free_ratio), blockage_ratio=float(blockage), free_intervals=intervals)

# 【品質工具】判斷相鄰橫斷面是否在非端點位置不合理交叉。
def _v21_cross_sections_properly_intersect(first: CrossSection, second: CrossSection, endpoint_margin: float=0.02) -> bool:
    a = np.asarray([first.left_x, first.left_y], dtype=np.float64)
    b = np.asarray([first.right_x, first.right_y], dtype=np.float64)
    c = np.asarray([second.left_x, second.left_y], dtype=np.float64)
    d = np.asarray([second.right_x, second.right_y], dtype=np.float64)
    first_vector = b - a
    second_vector = d - c
    denominator = float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
    if abs(denominator) < 1e-09:
        return False
    offset = c - a
    first_fraction = float((offset[0] * second_vector[1] - offset[1] * second_vector[0]) / denominator)
    second_fraction = float((offset[0] * first_vector[1] - offset[1] * first_vector[0]) / denominator)
    margin = float(np.clip(endpoint_margin, 0.0, 0.49))
    return bool(margin < first_fraction < 1.0 - margin and margin < second_fraction < 1.0 - margin)

# 【品質工具】檢查候選橫斷面是否與前面局部橫斷面交叉。
def _v21_section_crosses_previous(candidate: CrossSection, previous: list[CrossSection], local_window: int | None=None) -> bool:
    relevant = previous
    if local_window is not None and int(local_window) > 0:
        relevant = previous[-int(local_window):]
    return any((_v21_cross_sections_properly_intersect(existing, candidate) for existing in relevant))

# 【回退工具】依期望方向與全域方向產生少量橫斷切線候選。
def _v21_tangent_candidates(desired: np.ndarray, global_tangent: np.ndarray) -> list[np.ndarray]:
    desired = np.asarray(desired, dtype=np.float64)
    global_tangent = np.asarray(global_tangent, dtype=np.float64)
    desired /= max(float(np.linalg.norm(desired)), 1e-06)
    global_tangent /= max(float(np.linalg.norm(global_tangent)), 1e-06)
    if float(desired @ global_tangent) < 0.0:
        global_tangent *= -1.0
    candidates: list[np.ndarray] = []
    for local_fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
        tangent = local_fraction * desired + (1.0 - local_fraction) * global_tangent
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-06:
            continue
        tangent /= norm
        if not candidates or abs(float(candidates[-1] @ tangent)) < 0.999999:
            candidates.append(tangent)
    return candidates

# 【自由通道】跨站點連接相互重疊的自由區間，選出主要連續通道。
def connect_free_lane_intervals(sections: list[CrossSection]) -> dict:
    previous_section: CrossSection | None = None
    previous_interval: dict | None = None
    segment_id = -1
    segment_lengths: dict[int, int] = {}
    selected_count = 0
    for section in sorted(sections, key=lambda item: item.station_index):
        intervals = section.free_intervals
        if not intervals:
            previous_section = None
            previous_interval = None
            continue
        chosen_index: int | None = None
        if previous_section is not None and previous_interval is not None and (section.station_index == previous_section.station_index + 1):
            candidates = []
            for index, interval in enumerate(intervals):
                overlap = max(0.0, min(previous_interval['end_fraction'], interval['end_fraction']) - max(previous_interval['start_fraction'], interval['start_fraction']))
                center_shift = abs(previous_interval['center_fraction'] - interval['center_fraction'])
                if overlap > 0.0 or center_shift <= 0.25:
                    score = 2.0 * overlap + interval['width_ratio'] - 0.5 * center_shift
                    candidates.append((score, index))
            if candidates:
                chosen_index = max(candidates)[1]
        if chosen_index is None:
            segment_id += 1
            chosen_index = max(range(len(intervals)), key=lambda index: intervals[index]['width_px'])
        section.selected_free_interval_index = int(chosen_index)
        section.free_lane_segment_id = int(segment_id)
        selected_count += 1
        segment_lengths[segment_id] = segment_lengths.get(segment_id, 0) + 1
        previous_section = section
        previous_interval = intervals[chosen_index]
    longest = max(segment_lengths.values(), default=0)
    return {'selected_sections': selected_count, 'segment_count': len(segment_lengths), 'longest_segment_sections': longest}

# 【自由通道】將選中的連續自由區間畫成 free_lane_corridor 遮罩。
def free_lane_mask_from_sections(shape: tuple[int, int], sections: list[CrossSection]) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    selected = []
    for section in sorted(sections, key=lambda item: item.station_index):
        index = section.selected_free_interval_index
        if index is None or index >= len(section.free_intervals):
            continue
        interval = section.free_intervals[index]
        left = (int(interval['left_x']), int(interval['left_y']))
        right = (int(interval['right_x']), int(interval['right_y']))
        draw.line((*left, *right), fill=1, width=2)
        selected.append((section, interval, left, right))
    for first, second in zip(selected, selected[1:]):
        first_section, _, first_left, first_right = first
        second_section, _, second_left, second_right = second
        if second_section.station_index != first_section.station_index + 1 or second_section.free_lane_segment_id != first_section.free_lane_segment_id:
            continue
        draw.polygon([first_left, first_right, second_right, second_left], fill=1)
    return np.asarray(canvas, dtype=bool)

# 【分類工具】計算數值序列連續高於門檻的最長長度。
def longest_threshold_run(values: Iterable[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        if value >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best

# 【分類工具】回傳最長連續門檻區段的實際索引，供信心計算。
def longest_threshold_run_indices(values: list[float], threshold: float) -> list[int]:
    best: list[int] = []
    current: list[int] = []
    for index, value in enumerate(values):
        if value >= threshold:
            current.append(index)
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []
    return best

def decision_confidence_from_sections(prediction: int, ordered_ratios: list[float], sections: list[CrossSection], p: Parameters) -> dict:
    # 【主要功能：分類完成後估計判定信心】
    # 只平均真正支撐該分類之橫斷面的幾何來源權重；不修改阻塞率、P90、
    # 連續區段或 full／half／small_or_clear 分類本身。
    by_index = {section.station_index: section for section in sections}
    if prediction == 2:
        decision_indices = longest_threshold_run_indices(ordered_ratios, p.full_threshold)
        basis = 'longest_full_run'
    elif prediction == 1:
        decision_indices = longest_threshold_run_indices(ordered_ratios, p.partial_threshold)
        basis = 'longest_partial_run'
    else:
        decision_indices = sorted(by_index)
        basis = 'all_valid_low_blockage_sections'
    decision_sections = [by_index[index] for index in decision_indices if index in by_index]
    if not decision_sections:
        return {'label': 'insufficient', 'score': 0.0, 'basis': basis, 'section_indices': [], 'source_counts': {}}
    weights = [section.source_weight for section in decision_sections]
    score = float(np.mean(weights))
    if score >= 0.8:
        label = 'high'
    elif score >= 0.55:
        label = 'medium'
    else:
        label = 'low'
    counts = {source: int(sum((section.source_type == source for section in decision_sections))) for source in SOURCE_WEIGHTS}
    return {'label': label, 'score': score, 'basis': basis, 'section_indices': decision_indices, 'source_counts': counts}

# =============================================================================
# 19. V40 最終分類
# =============================================================================
# 使用 P90、連續高／部分阻塞橫斷面與有效比例輸出四種結果。
def road_aligned_method(road: np.ndarray, obstacles: np.ndarray, p: Parameters, auxiliary_masks: dict[str, np.ndarray] | None=None) -> dict:
    # 【主要功能：單張影像的 V40 完整判定入口】
    # 依序執行主要道路選擇、道路／導航整合、橫斷面建立、自由通道計算、
    # P90 與連續阻塞分類，最後加入來源信心與所有視覺化所需診斷資料。
    height, width = road.shape
    image_area = height * width
    road_area_ratio = float(road.sum() / image_area)
    if road_area_ratio < p.min_road_area_ratio:
        return invalid_result('road_too_small', road_area_ratio)
    closed = binary_close(road, p.close_radius)
    main_road, main_ratio = largest_component(closed)
    if not np.any(main_road):
        return invalid_result('no_road', road_area_ratio, main_ratio)
    orientation_evidence = road_component_orientation_evidence(closed)
    diagonal = math.hypot(height, width)
    if p.enable_continuous_direction_tracking:
        track_selection_auxiliary_masks = None if bool(p.enable_v32_authoritative_navigation_direction) else auxiliary_masks
        corridor = continuous_direction_corridor(closed, obstacles, orientation_evidence, p, auxiliary_masks=track_selection_auxiliary_masks)
        tracked_length = polyline_length(corridor['centerline']) if corridor.get('ok') else 0.0
        axis_selection_reason = corridor.get('axis_selection_reason', 'continuous_angle_tracking_failed')
    else:
        corridor = reconstruct_road_corridor(closed, obstacles, p, observation_mode='tracked')
        tracked_length = polyline_length(corridor['centerline']) if corridor.get('ok') else 0.0
        axis_selection_reason = orientation_evidence['selection_basis']
        should_try_transposed = orientation_evidence['preferred_axis'] == 'horizontal_x'
        if should_try_transposed:
            transposed = reconstruct_road_corridor(closed.T, obstacles.T, p, observation_mode='tracked')
            if transposed.get('ok'):
                transposed = transpose_corridor_to_original_coordinates(transposed)
                transposed_length = polyline_length(transposed['centerline'])
                if not corridor.get('ok') or transposed_length > tracked_length * 1.05:
                    corridor = transposed
                    tracked_length = transposed_length
                    axis_selection_reason = 'strong_horizontal_border_evidence'
    if not corridor.get('ok') or tracked_length / diagonal < p.min_path_diagonal_ratio:
        fallback = reconstruct_road_corridor(closed, obstacles, p, observation_mode='envelope')
        fallback_length = polyline_length(fallback['centerline']) if fallback.get('ok') else 0.0
        if fallback.get('ok') and fallback_length > tracked_length:
            corridor = fallback
            tracked_length = fallback_length
            axis_selection_reason = 'debug_envelope_fallback'
    corridor['orientation_evidence'] = orientation_evidence
    corridor['axis_selection_reason'] = axis_selection_reason
    if orientation_evidence['selection_basis'] == 'bottom_or_vertical_component_support':
        corridor['warnings'] = [warning for warning in corridor.get('warnings', []) if warning != 'road_may_be_horizontal']
    if not corridor.get('ok'):
        return invalid_result(corridor.get('reason', 'corridor_reconstruction_failed'), road_area_ratio, main_ratio)
    auxiliary_summary = associate_auxiliary_direction_evidence(auxiliary_masks, corridor, road.shape, p)
    corridor['auxiliary_direction_evidence'] = auxiliary_summary
    if p.enable_v21_decoupled_geometry or p.enable_v20_v12_base_v19_direction:
        corridor = apply_v20_v12_base_v19_direction(road, obstacles, corridor, auxiliary_masks, p)
    else:
        corridor = apply_unified_evidence_reconstruction(road, obstacles, corridor, auxiliary_masks, p)
    corridor['auxiliary_direction_evidence'] = auxiliary_summary
    raw_path = corridor['raw_centerline']
    geometry_centerline = np.asarray(corridor['centerline'], dtype=np.float64)
    smooth_path = geometry_centerline.copy()
    analysis_geometry_corridor = corridor['analysis_geometry_corridor']
    inferred_occlusion_mask = np.asarray(corridor.get('inferred_occlusion_mask', analysis_geometry_corridor & obstacles & ~road), dtype=bool) & ~road
    inferred_internal_gap_mask = np.asarray(corridor.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool) & ~road
    obstacle_supported_reconstruction_mask = np.asarray(corridor.get('obstacle_supported_reconstruction_mask', np.zeros_like(road)), dtype=bool) & ~road
    full_road_corridor = road | inferred_occlusion_mask | inferred_internal_gap_mask | obstacle_supported_reconstruction_mask
    geometry_metrics = corridor_geometry_metrics(geometry_centerline, np.asarray(corridor['left'], dtype=np.float64), np.asarray(corridor['right'], dtype=np.float64), corridor.get('v13_central_roadline_mask'), inferred_internal_gap_mask, analysis_geometry_corridor)
    corridor['geometry_metrics'] = geometry_metrics
    valid_obstacles = obstacles & full_road_corridor
    branch_clusters = 0
    raw_length = polyline_length(raw_path)
    fitted_length = polyline_length(smooth_path)
    if len(raw_path) < 2 or fitted_length / diagonal < p.min_path_diagonal_ratio:
        return invalid_result('centerline_too_short', road_area_ratio, main_ratio, branch_clusters)
    quality_warnings = list(corridor.get('warnings', []))
    if main_ratio < p.min_main_component_ratio:
        quality_warnings.append('fragmented_road_mask')
    stations, left_stations, right_stations = resample_corridor_geometry(smooth_path, np.asarray(corridor['left'], dtype=np.float64), np.asarray(corridor['right'], dtype=np.float64), p.sample_spacing)
    station_sources = resample_corridor_sources(smooth_path, corridor.get('band_source_types', []), p.sample_spacing)
    native_center_band_mask = np.asarray(corridor.get('v35_native_center_authority_mask', []), dtype=bool)
    if len(native_center_band_mask) == len(smooth_path):
        native_center_labels = ['roadline_center' if value else 'outside' for value in native_center_band_mask]
        native_center_station_labels = resample_corridor_sources(smooth_path, native_center_labels, p.sample_spacing)
        native_center_station_mask = np.asarray([label == 'roadline_center' for label in native_center_station_labels], dtype=bool)
    else:
        native_center_station_mask = np.zeros(len(stations), dtype=bool)
    v40_frozen_pair_band_mask = np.asarray(corridor.get('v40_frozen_pair_valid_mask', []), dtype=bool)
    if len(v40_frozen_pair_band_mask) == len(smooth_path):
        v40_pair_labels = ['valid' if value else 'invalid' for value in v40_frozen_pair_band_mask]
        v40_pair_station_labels = resample_corridor_sources(smooth_path, v40_pair_labels, p.sample_spacing)
        v40_frozen_pair_station_mask = np.asarray([label == 'valid' for label in v40_pair_station_labels], dtype=bool)
    else:
        v40_frozen_pair_station_mask = np.ones(len(stations), dtype=bool)
    center_source_band_labels = list(corridor.get('v36_center_source_labels', []))
    if len(center_source_band_labels) == len(smooth_path):
        center_source_station_labels = resample_corridor_sources(smooth_path, center_source_band_labels, p.sample_spacing)
    else:
        center_source_station_labels = ['v12'] * len(stations)
    station_core_states = [reconstruction_state_for_source(source) for source in station_sources]
    if len(stations) < 3:
        return invalid_result('too_few_stations', road_area_ratio, main_ratio, branch_clusters)
    selected_curve_reference = corridor.get('v18_selected_curve_reference')
    if selected_curve_reference is not None:
        selected_role = str(selected_curve_reference.get('role', 'shape_shifted'))
        auxiliary_application = {'status': 'selected_auxiliary_curve_applied', 'mode': 'v21_selected_auxiliary_navigation_center', 'segments': [{'source_family': str(selected_curve_reference.get('source_family', 'auxiliary')), 'source_category': str(selected_curve_reference.get('source_category', 'auxiliary')), 'role': selected_role, 'selection_reason': str(selected_curve_reference.get('selection_reason', 'reliable_shape'))}], 'refined_sections': int(min(len(stations), corridor.get('v13_reconstruction_summary', {}).get('v17_guided_center_bands', 0))), 'rejected_sections': 0, 'terminal_extension_used': False, 'accepted_curve_count': 1}
    elif bool(p.enable_v18_auxiliary_curve_roles):
        auxiliary_application = {'status': str(auxiliary_summary.get('status', 'no_masks')) if not auxiliary_summary.get('evidence', []) else 'no_reliable_auxiliary_curve_road_geometry_kept', 'mode': 'global_angle_cross_sections', 'segments': [], 'refined_sections': 0, 'rejected_sections': int(len(auxiliary_summary.get('evidence', []))), 'terminal_extension_used': False, 'accepted_curve_count': 0}
    else:
        left_stations, right_stations, auxiliary_application = apply_auxiliary_piecewise_sections(stations, left_stations, right_stations, station_sources, analysis_geometry_corridor, auxiliary_summary, float(corridor.get('selected_direction_angle_deg', 90.0)), p)
    corridor['auxiliary_direction_application'] = auxiliary_application
    if auxiliary_application['refined_sections']:
        corridor['direction_geometry_mode'] = auxiliary_application['mode']
        if auxiliary_application['mode'] == 'v21_selected_auxiliary_navigation_center':
            corridor['cross_section_method'] = 'navigation_tangent_fixed_corridor_intersection'
        else:
            corridor['cross_section_method'] = 'auxiliary_partial_spline_paired_boundary_resection' if auxiliary_application['mode'] == 'auxiliary_partial_spline_local_tangent' else 'auxiliary_piecewise_paired_boundary_resection'
    sections: list[CrossSection] = []
    valid_flags = []
    fixed_section_corridor = np.asarray(analysis_geometry_corridor | road, dtype=bool)
    section_tangents = v21_width_safe_cross_section_tangents(stations, left_stations, right_stations, p)
    guided_band_mask = np.asarray(corridor.get('v21_guided_center_mask', []), dtype=bool)
    selected_navigation_curve_active = bool(selected_curve_reference is not None and auxiliary_application.get('status') == 'selected_auxiliary_curve_applied' and (auxiliary_application.get('mode') == 'v21_selected_auxiliary_navigation_center'))
    if selected_navigation_curve_active and len(guided_band_mask) == len(smooth_path):
        guidance_labels = ['guided' if value else 'v12_global' for value in guided_band_mask]
        station_guidance = resample_corridor_sources(smooth_path, guidance_labels, p.sample_spacing)
    else:
        station_guidance = ['v12_global'] * len(stations)
    global_radians = math.radians(float(corridor.get('selected_direction_angle_deg', 90.0)))
    global_tangent = np.asarray([math.cos(global_radians), math.sin(global_radians)], dtype=np.float64)
    v31_band_tangents = np.asarray(corridor.get('v31_navigation_tangents', []), dtype=np.float64)
    v31_station_tangents = _v31_resample_tangent_field(np.asarray(smooth_path, dtype=np.float64), v31_band_tangents, p.sample_spacing)
    v32_band_tangents = np.asarray(corridor.get('v32_navigation_tangents', []), dtype=np.float64)
    v32_station_tangents = _v31_resample_tangent_field(np.asarray(smooth_path, dtype=np.float64), v32_band_tangents, p.sample_spacing)
    v32_direction_field_applied = bool(p.enable_v32_authoritative_navigation_direction and len(v32_station_tangents) == len(section_tangents) and (len(v32_station_tangents) == len(stations)))
    v31_direction_field_applied = bool(p.enable_v31_multi_aux_pretrack_direction and (not v32_direction_field_applied) and (len(v31_station_tangents) == len(section_tangents)) and (len(v31_station_tangents) == len(stations)))
    if v32_direction_field_applied:
        section_tangents = v32_station_tangents.copy()
        navigation_summary = corridor.get('v32_navigation_direction_summary', {})
        auxiliary_application['status'] = str(navigation_summary.get('status', 'v32_direction_applied'))
        auxiliary_application['mode'] = 'v32_authoritative_navigation_direction'
        auxiliary_application['accepted_curve_count'] = int(navigation_summary.get('accepted_curve_count', 0))
        auxiliary_application['refined_sections'] = int(navigation_summary.get('authoritative_bands', 0))
        auxiliary_application['trusted_core_sections'] = int(navigation_summary.get('authoritative_bands', 0))
        corridor['direction_geometry_mode'] = 'v32_authoritative_auxiliary_or_road_long_window'
    elif v31_direction_field_applied:
        section_tangents = v31_station_tangents.copy()
        auxiliary_application['mode'] = 'v31_multi_aux_navigation_direction'
        auxiliary_application['accepted_curve_count'] = int(corridor.get('v31_navigation_direction_summary', {}).get('accepted_curve_count', 0))
        auxiliary_application['refined_sections'] = int(corridor.get('v31_navigation_direction_summary', {}).get('supported_bands', 0))
        corridor['direction_geometry_mode'] = 'v31_multi_aux_local_direction_field'
    else:
        for index, label in enumerate(station_guidance[:len(section_tangents)]):
            if label == 'v12_global':
                section_tangents[index] = global_tangent
    for index in range(1, len(section_tangents)):
        if float(section_tangents[index] @ section_tangents[index - 1]) < 0.0:
            section_tangents[index] *= -1.0
    section_angles = np.unwrap(np.arctan2(section_tangents[:, 1], section_tangents[:, 0]))
    section_widths = np.linalg.norm(right_stations - left_stations, axis=1)
    if not v32_direction_field_applied:
        for index in range(1, len(section_angles)):
            spacing = max(1.0, float(np.linalg.norm(stations[index] - stations[index - 1])))
            half_width = max(float(p.min_road_width), 0.25 * float(section_widths[index - 1] + section_widths[index]))
            maximum_delta = float(p.v21_cross_section_curvature_safety) * math.atan2(spacing, half_width)
            delta = float(section_angles[index] - section_angles[index - 1])
            section_angles[index] = section_angles[index - 1] + float(np.clip(delta, -maximum_delta, maximum_delta))
    section_tangents = np.column_stack((np.cos(section_angles), np.sin(section_angles)))
    corridor['v21_local_tangent_section_count'] = int(sum((label == 'guided' for label in station_guidance)))
    corridor['v31_direction_field_applied'] = bool(v31_direction_field_applied)
    corridor['v32_direction_field_applied'] = bool(v32_direction_field_applied)
    v40_native_measurement = bool(p.enable_v40_v38_roadline_center_sections and corridor.get('v40_navigation_mode') == 'roadline_center' and (not corridor.get('v40_roadline_native_fallback_to_v38', False)))
    if v40_native_measurement:
        frozen_widths = np.linalg.norm(right_stations - left_stations, axis=1)
        frozen_valid = np.asarray(v40_frozen_pair_station_mask, dtype=bool) & np.all(np.isfinite(left_stations), axis=1) & np.all(np.isfinite(right_stations), axis=1) & (frozen_widths >= float(p.min_road_width))
        frozen_indices = np.arange(len(stations), dtype=np.int32)
        v37_pairing = {'left': np.asarray(left_stations, dtype=np.float64).copy(), 'right': np.asarray(right_stations, dtype=np.float64).copy(), 'valid': frozen_valid, 'sources': np.where(frozen_valid, 'v40_frozen_roadline_native_pair', 'unpaired').astype(object), 'left_indices': frozen_indices.copy(), 'right_indices': frozen_indices.copy(), 'success_ratio': float(frozen_valid.mean()) if len(frozen_valid) else 0.0, 'crossing_rejections': 0, 'backtrack_rejections': 0, 'left_progress': frozen_indices.astype(np.float64), 'right_progress': frozen_indices.astype(np.float64), 'left_branch_ids': np.zeros(len(stations), dtype=np.int32), 'right_branch_ids': np.zeros(len(stations), dtype=np.int32), 'left_boundary_branch_count': 1, 'right_boundary_branch_count': 1}
    elif bool(p.enable_v37_paired_cross_sections):
        v37_pairing = _v37_pair_boundaries_monotone(stations, section_tangents, left_stations, right_stations, fixed_section_corridor, p, anchor_indices=np.arange(len(stations), dtype=np.int32), road_mask=road)
    else:
        v37_pairing = {'left': np.full_like(stations, np.nan), 'right': np.full_like(stations, np.nan), 'valid': np.zeros(len(stations), dtype=bool), 'sources': np.full(len(stations), 'unpaired', dtype=object), 'left_indices': np.full(len(stations), -1, dtype=np.int32), 'right_indices': np.full(len(stations), -1, dtype=np.int32), 'success_ratio': 0.0, 'crossing_rejections': 0, 'backtrack_rejections': 0}
    corridor['v37_boundary_pairing'] = {'enabled': bool(p.enable_v37_paired_cross_sections), 'success_ratio': float(v37_pairing['success_ratio']), 'source_counts': {source: int(np.sum(np.asarray(v37_pairing['sources']) == source)) for source in ('paired_boundaries', 'neighbor_interpolated', 'v40_frozen_roadline_native_pair', 'unpaired')}, 'crossing_rejections': int(v37_pairing['crossing_rejections']), 'backtrack_rejections': int(v37_pairing['backtrack_rejections']), 'left_indices': np.asarray(v37_pairing['left_indices'], dtype=np.int32).tolist(), 'right_indices': np.asarray(v37_pairing['right_indices'], dtype=np.int32).tolist(), 'left_progress': np.asarray(v37_pairing.get('left_progress', []), dtype=np.float64).tolist(), 'right_progress': np.asarray(v37_pairing.get('right_progress', []), dtype=np.float64).tolist(), 'left_branch_ids': np.asarray(v37_pairing.get('left_branch_ids', []), dtype=np.int32).tolist(), 'right_branch_ids': np.asarray(v37_pairing.get('right_branch_ids', []), dtype=np.int32).tolist(), 'left_boundary_branch_count': int(v37_pairing.get('left_boundary_branch_count', 0)), 'right_boundary_branch_count': int(v37_pairing.get('right_boundary_branch_count', 0))}
    intersection_fallback_count = 0
    intersection_rejected_count = 0
    for index, (center, left, right, source_type) in enumerate(zip(stations, left_stations, right_stations, station_sources)):
        desired_tangent = section_tangents[index]
        section = None
        used_tangent = desired_tangent
        native_center_station = bool(index < len(native_center_station_mask) and native_center_station_mask[index])
        if bool(p.enable_v37_paired_cross_sections and index < len(v37_pairing['valid']) and v37_pairing['valid'][index]):
            section = cross_section_between_boundaries(center, np.asarray(v37_pairing['left'][index], dtype=np.float64), np.asarray(v37_pairing['right'][index], dtype=np.float64), valid_obstacles, index, p.min_road_width, source_type=source_type)
            if section is not None and _v21_section_crosses_previous(section, sections, local_window=1):
                if v40_native_measurement:
                    intersection_rejected_count += 1
                section = None
            if section is not None:
                section.geometry_source = str(v37_pairing['sources'][index])
        if v40_native_measurement and section is None:
            valid_flags.append(False)
            continue
        if section is None and bool(p.enable_v37_paired_cross_sections):
            proposed_v12 = cross_section_between_boundaries(center, left, right, valid_obstacles, index, p.min_road_width, source_type=source_type)
            if proposed_v12 is not None and (not _v21_section_crosses_previous(proposed_v12, sections, local_window=1)):
                proposed_v12.geometry_source = 'v12_boundary_pair'
                section = proposed_v12
        if native_center_station:
            tangent_candidates = [desired_tangent]
        elif v32_direction_field_applied:
            tangent_candidates = []
            base_angle = math.atan2(float(desired_tangent[1]), float(desired_tangent[0]))
            for delta_degrees in (0.0, 3.0, -3.0, 6.0, -6.0, 10.0, -10.0):
                angle = base_angle + math.radians(delta_degrees)
                tangent_candidates.append(np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64))
        else:
            tangent_candidates = [desired_tangent] if bool(p.enable_v29_continuous_cross_sections) else _v21_tangent_candidates(desired_tangent, global_tangent)
        for candidate_index, tangent in enumerate(tangent_candidates):
            if section is not None:
                break
            proposed = cross_section_at(center, tangent, fixed_section_corridor, valid_obstacles, index, max_gap=0, obstacle_link_gap=0, min_width=p.min_road_width, expand_with_obstacles=False)
            if proposed is None or _v21_section_crosses_previous(proposed, sections, local_window=p.v29_cross_section_crossing_history if bool(p.enable_v29_continuous_cross_sections) else None):
                continue
            section = proposed
            section.geometry_source = 'image_normal_fallback'
            used_tangent = tangent
            if candidate_index > 0:
                intersection_fallback_count += 1
            break
        section_tangents[index] = used_tangent
        if section is not None:
            section.source_type = source_type
            section.source_weight = float(SOURCE_WEIGHTS.get(source_type, 0.35))
        elif not native_center_station and (not v32_direction_field_applied or bool(p.enable_v33_physical_roadline_tracks)):
            proposed = cross_section_between_boundaries(center, left, right, valid_obstacles, index, p.min_road_width, source_type=source_type)
            if proposed is not None and (not _v21_section_crosses_previous(proposed, sections, local_window=1 if bool(p.enable_v33_physical_roadline_tracks) else p.v29_cross_section_crossing_history if bool(p.enable_v29_continuous_cross_sections) else None)):
                section = proposed
                section.geometry_source = 'v12_boundary_pair'
            else:
                intersection_rejected_count += 1
        valid_flags.append(section is not None)
        if section is not None:
            sections.append(section)
    corridor['v21_cross_section_intersection_fallback_count'] = int(intersection_fallback_count)
    corridor['v21_cross_section_intersection_rejected_count'] = int(intersection_rejected_count)
    if v40_native_measurement:
        measured_widths = np.asarray([section.road_width_px for section in sections], dtype=np.float64)
        if len(measured_widths) >= 2:
            width_steps = np.abs(np.diff(measured_widths)) / np.maximum(np.minimum(measured_widths[:-1], measured_widths[1:]), 1.0)
            width_step_p90 = float(np.percentile(width_steps, 90))
            width_smoothness = float(1.0 / (1.0 + width_step_p90))
        else:
            width_step_p90 = float('inf')
            width_smoothness = 0.0
        native_quality = dict(corridor.get('v40_native_pairing_quality', {}))
        native_quality.update({'final_valid_ratio': float(len(sections) / max(1, len(stations))), 'final_crossing_count': int(intersection_rejected_count), 'final_width_step_p90': width_step_p90, 'final_width_smoothness': width_smoothness})
        corridor['v40_native_pairing_quality'] = native_quality
    valid_ratio = len(sections) / len(stations)
    global_fallback_used = False
    if valid_ratio < p.min_valid_slice_ratio and (not v40_native_measurement):
        fallback_sections: list[CrossSection] = []
        fallback_flags: list[bool] = []
        for index, (center, left, right, source_type) in enumerate(zip(stations, left_stations, right_stations, station_sources)):
            section = cross_section_at(center, global_tangent, fixed_section_corridor, valid_obstacles, index, max_gap=0, obstacle_link_gap=0, min_width=p.min_road_width, expand_with_obstacles=False)
            if section is None:
                section = cross_section_between_boundaries(center, left, right, valid_obstacles, index, p.min_road_width, source_type=source_type)
            if section is not None:
                section.source_type = source_type
                section.source_weight = float(SOURCE_WEIGHTS.get(source_type, 0.35))
                fallback_sections.append(section)
            fallback_flags.append(section is not None)
        fallback_ratio = len(fallback_sections) / len(stations)
        if not bool(p.enable_v29_continuous_cross_sections) and fallback_ratio > valid_ratio:
            sections = fallback_sections
            valid_flags = fallback_flags
            valid_ratio = fallback_ratio
            global_fallback_used = True
    corridor['v21_cross_section_global_fallback_used'] = bool(global_fallback_used)
    corridor['cross_section_method'] = 'v40_roadline_native_frozen_boundary_correspondence' if v40_native_measurement else 'v37_monotone_paired_boundaries_perspective_sections' if bool(p.enable_v37_paired_cross_sections) else 'v32_authoritative_direction_frozen_corridor_intersection' if v32_direction_field_applied else 'v31_navigation_direction_fixed_corridor_intersection' if v31_direction_field_applied else 'v29_continuous_tangent_boundary_pair_fallback' if bool(p.enable_v29_continuous_cross_sections) else 'global_tangent_fixed_corridor_intersection_fallback' if global_fallback_used else 'navigation_tangent_fixed_corridor_intersection'
    if valid_ratio < p.min_valid_slice_ratio:
        return invalid_result('insufficient_boundaries', road_area_ratio, main_ratio, branch_clusters, raw_length, len(stations), len(sections), valid_ratio, raw_path, smooth_path, stations, sections, metadata={'quality_warnings': quality_warnings, 'left_boundary': corridor['left'].tolist(), 'right_boundary': corridor['right'].tolist(), 'raw_left_boundary': corridor['raw_left'].tolist(), 'raw_right_boundary': corridor['raw_right'].tolist(), 'row_missing_band_ratio': corridor['missing_ratio'], 'row_observed_bands': corridor['observed_bands'], 'row_span_bands': corridor['span_bands'], 'row_median_width_px': corridor['median_width'], 'accepted_occlusion_gaps': corridor['accepted_gaps'], 'one_sided_extrapolations': corridor['one_sided_extrapolations'], 'one_sided_geometry': corridor['one_sided_geometry'], 'one_side_inferences': corridor['one_side_inferences'], 'interior_occlusion_inferences': corridor['interior_occlusion_inferences'], 'corridor_source_counts': corridor['source_counts'], 'reconstruction_pipeline': corridor.get('reconstruction_pipeline', 'unified_three_state'), 'reconstruction_core_state_counts': corridor.get('core_state_counts', {}), 'reconstruction_evidence_counts': corridor.get('support_evidence_counts', {}), 'station_reconstruction_states': station_core_states, 'discarded_terminal_fragment_bands': corridor['discarded_terminal_fragment_bands'], 'row_interpolation_ratio': corridor['interpolation_ratio'], 'row_extrapolation_ratio': corridor['extrapolation_ratio'], 'curve_fit_method': corridor['curve_fit_method'], 'corridor_observation_mode': corridor['observation_mode'], 'cross_section_method': corridor.get('cross_section_method', 'paired_boundary_correspondence'), 'direction_geometry_mode': corridor.get('direction_geometry_mode', p.direction_geometry_mode), 'orientation_evidence': orientation_evidence, 'axis_selection_reason': axis_selection_reason, 'selected_direction_angle_deg': corridor.get('selected_direction_angle_deg'), 'selected_track_id': corridor.get('selected_track_id'), 'track_hypotheses': corridor.get('track_hypotheses', []), 'track_score_margin': corridor.get('track_score_margin'), 'road_direction_hint_deg': corridor.get('road_direction_hint_deg'), 'auxiliary_direction_hypotheses': corridor.get('auxiliary_direction_hypotheses', []), 'selected_joint_auxiliary_score': corridor.get('selected_joint_auxiliary_score', 0.0), 'local_direction_refined': corridor.get('local_direction_refined', False), 'local_direction_refined_sections': corridor.get('local_direction_refined_sections', 0), 'auxiliary_direction_status': auxiliary_application['status'], 'auxiliary_direction_mode': auxiliary_application['mode'], 'auxiliary_direction_evidence_count': len(auxiliary_summary.get('evidence', [])), 'auxiliary_direction_source_counts': auxiliary_summary.get('source_counts', {}), 'auxiliary_direction_segments': auxiliary_application['segments'], 'auxiliary_direction_refined_sections': auxiliary_application['refined_sections'], 'auxiliary_direction_rejected_sections': auxiliary_application['rejected_sections'], 'v32_navigation_direction_summary': corridor.get('v32_navigation_direction_summary', {}), 'v32_authoritative_curves': corridor.get('v32_authoritative_curves', []), 'v32_navigation_axis': corridor.get('v32_navigation_axis', []), 'v32_navigation_tangents': section_tangents.tolist(), 'v33_physical_tracks': corridor.get('v33_physical_tracks', []), 'v33_navigation_axis': corridor.get('v33_navigation_axis', []), 'v33_navigation_tangents': section_tangents.tolist(), 'v35_native_center_summary': corridor.get('v35_native_center_summary', {}), 'v35_native_center_station_count': int(native_center_station_mask.sum()), 'v36_center_source_labels': center_source_station_labels, 'v36_transition_station_count': int(sum((value == 'transition' for value in center_source_station_labels))), 'v37_boundary_pairing': corridor.get('v37_boundary_pairing', {}), 'v37_post_navigation_pairing': corridor.get('v37_post_navigation_pairing', {}), 'navigation_mode': corridor.get('v40_navigation_mode', 'v12_global'), 'central_roadline_used': bool(corridor.get('v40_central_roadline_used', False)), 'central_roadline_name': corridor.get('v40_central_roadline_name', 'none'), 'central_roadline_coverage': float(corridor.get('v40_central_roadline_coverage', 0.0)), 'roadline_native_pairing_quality': corridor.get('v40_native_pairing_quality', {}), 'roadline_native_fallback_to_v38': bool(corridor.get('v40_roadline_native_fallback_to_v38', False)), 'frozen_navigation_preserved': bool(corridor.get('v40_frozen_navigation_preserved', False)), 'v32_direction_field_applied': bool(corridor.get('v32_direction_field_applied', False)), 'auxiliary_terminal_extension_used': bool(corridor.get('v13_reconstruction_summary', {}).get('terminal_extension_count', 0)), 'auxiliary_direction_calibrated': False, 'v13_reconstruction_status': corridor.get('v13_reconstruction_status', 'not_applied'), 'v13_reconstruction_summary': corridor.get('v13_reconstruction_summary', {}), 'full_road_corridor_area_ratio': float(full_road_corridor.mean()), 'analysis_geometry_corridor_area_ratio': float(analysis_geometry_corridor.mean()), 'reconstructed_corridor_area_ratio': float(full_road_corridor.mean()), 'inferred_occlusion_area_ratio': float(inferred_occlusion_mask.mean()), 'inferred_internal_gap_area_ratio': float(inferred_internal_gap_mask.mean()), 'visible_road_in_full_ratio': float(np.logical_and(road, full_road_corridor).sum() / max(1, int(road.sum()))), 'valid_obstacle_pixel_ratio': float(valid_obstacles.sum() / max(1, int(obstacles.sum())))})
    by_index = {section.station_index: section.blockage_ratio for section in sections}
    ordered_ratios = [by_index.get(index, -1.0) for index in range(len(stations))]
    valid_ratios = np.asarray([section.blockage_ratio for section in sections], dtype=np.float64)
    p90 = float(np.percentile(valid_ratios, 90))
    maximum = float(valid_ratios.max())
    full_run = longest_threshold_run(ordered_ratios, p.full_threshold)
    partial_run = longest_threshold_run(ordered_ratios, p.partial_threshold)
    min_free_ratio = float(min((section.free_width_ratio for section in sections)))
    required = p.required_consecutive_slices
    if p90 >= p.full_threshold and full_run >= required:
        prediction = 2
    elif p90 >= p.partial_threshold and partial_run >= required:
        prediction = 1
    else:
        prediction = 0
    free_lane_stats = connect_free_lane_intervals(sections)
    free_lane_corridor = free_lane_mask_from_sections(road.shape, sections)
    free_lane_stats['area_ratio'] = float(free_lane_corridor.mean())
    free_lane_stats['continuity_ratio'] = float(free_lane_stats['longest_segment_sections'] / max(1, len(stations)))
    decision_confidence = decision_confidence_from_sections(prediction, ordered_ratios, sections, p)
    changed_pixels = int(np.logical_xor(road, closed).sum())
    repair_ratio = changed_pixels / max(1, int(road.sum()))
    if corridor.get('v13_reconstruction_summary', {}).get('terminal_extension_count', 0):
        status = 'auxiliary_supported_extrapolated'
    elif corridor['one_sided_extrapolations']:
        status = 'one_sided_extrapolated'
    elif corridor['interior_occlusion_inferences']:
        status = 'interior_occlusion_inferred'
    elif corridor['interpolation_ratio'] > 0.0:
        status = 'interpolated'
    else:
        status = 'direct'
    if quality_warnings:
        status += '_review'
    return {'prediction': prediction, 'centerline_status': status, 'quality_warnings': quality_warnings, 'failure_reason': None, 'road_area_ratio': road_area_ratio, 'main_component_ratio': main_ratio, 'branch_clusters': branch_clusters, 'centerline_scale': 1.0, 'centerline_method': 'direction_adaptive_boundary_pair_corridor', 'left_boundary': corridor['left'].tolist(), 'right_boundary': corridor['right'].tolist(), 'raw_left_boundary': corridor['raw_left'].tolist(), 'raw_right_boundary': corridor['raw_right'].tolist(), 'row_missing_band_ratio': corridor['missing_ratio'], 'row_observed_bands': corridor['observed_bands'], 'row_span_bands': corridor['span_bands'], 'row_median_width_px': corridor['median_width'], 'accepted_occlusion_gaps': corridor['accepted_gaps'], 'one_sided_extrapolations': corridor['one_sided_extrapolations'], 'one_sided_geometry': corridor['one_sided_geometry'], 'one_side_inferences': corridor['one_side_inferences'], 'interior_occlusion_inferences': corridor['interior_occlusion_inferences'], 'corridor_source_counts': corridor['source_counts'], 'reconstruction_pipeline': corridor.get('reconstruction_pipeline', 'unified_three_state'), 'reconstruction_core_state_counts': corridor.get('core_state_counts', {}), 'reconstruction_evidence_counts': corridor.get('support_evidence_counts', {}), 'station_reconstruction_states': station_core_states, 'discarded_terminal_fragment_bands': corridor['discarded_terminal_fragment_bands'], 'row_interpolation_ratio': corridor['interpolation_ratio'], 'row_extrapolation_ratio': corridor['extrapolation_ratio'], 'curve_fit_method': corridor['curve_fit_method'], 'corridor_observation_mode': corridor['observation_mode'], 'cross_section_method': corridor.get('cross_section_method', 'paired_boundary_correspondence'), 'v21_cross_section_intersection_fallback_count': corridor.get('v21_cross_section_intersection_fallback_count', 0), 'v21_cross_section_intersection_rejected_count': corridor.get('v21_cross_section_intersection_rejected_count', 0), 'v21_cross_section_global_fallback_used': corridor.get('v21_cross_section_global_fallback_used', False), 'direction_geometry_mode': corridor.get('direction_geometry_mode', p.direction_geometry_mode), 'orientation_evidence': orientation_evidence, 'axis_selection_reason': axis_selection_reason, 'selected_direction_angle_deg': corridor.get('selected_direction_angle_deg'), 'selected_track_id': corridor.get('selected_track_id'), 'track_hypotheses': corridor.get('track_hypotheses', []), 'track_score_margin': corridor.get('track_score_margin'), 'road_direction_hint_deg': corridor.get('road_direction_hint_deg'), 'auxiliary_direction_hypotheses': corridor.get('auxiliary_direction_hypotheses', []), 'selected_joint_auxiliary_score': corridor.get('selected_joint_auxiliary_score', 0.0), 'local_direction_refined': corridor.get('local_direction_refined', False), 'local_direction_refined_sections': corridor.get('local_direction_refined_sections', 0), 'auxiliary_direction_status': auxiliary_application['status'], 'auxiliary_direction_mode': auxiliary_application['mode'], 'auxiliary_direction_evidence_count': len(auxiliary_summary.get('evidence', [])), 'auxiliary_direction_source_counts': auxiliary_summary.get('source_counts', {}), 'auxiliary_direction_evidence': auxiliary_summary.get('evidence', []), 'auxiliary_direction_segments': auxiliary_application['segments'], 'auxiliary_direction_refined_sections': auxiliary_application['refined_sections'], 'auxiliary_direction_rejected_sections': auxiliary_application['rejected_sections'], 'auxiliary_direction_accepted_curve_count': auxiliary_application.get('accepted_curve_count', 0), 'auxiliary_direction_trusted_core_sections': auxiliary_application.get('trusted_core_sections', 0), 'auxiliary_direction_transition_sections': auxiliary_application.get('transition_sections', 0), 'v32_navigation_direction_summary': corridor.get('v32_navigation_direction_summary', {}), 'v32_authoritative_curves': corridor.get('v32_authoritative_curves', []), 'v32_navigation_axis': corridor.get('v32_navigation_axis', []), 'v32_navigation_tangents': section_tangents.tolist(), 'v33_physical_tracks': corridor.get('v33_physical_tracks', []), 'v33_navigation_axis': corridor.get('v33_navigation_axis', []), 'v33_navigation_tangents': section_tangents.tolist(), 'v35_native_center_summary': corridor.get('v35_native_center_summary', {}), 'v35_native_center_station_count': int(native_center_station_mask.sum()), 'v36_center_source_labels': center_source_station_labels, 'v36_transition_station_count': int(sum((value == 'transition' for value in center_source_station_labels))), 'v37_boundary_pairing': corridor.get('v37_boundary_pairing', {}), 'v37_post_navigation_pairing': corridor.get('v37_post_navigation_pairing', {}), 'navigation_mode': corridor.get('v40_navigation_mode', 'v12_global'), 'central_roadline_used': bool(corridor.get('v40_central_roadline_used', False)), 'central_roadline_name': corridor.get('v40_central_roadline_name', 'none'), 'central_roadline_coverage': float(corridor.get('v40_central_roadline_coverage', 0.0)), 'roadline_native_pairing_quality': corridor.get('v40_native_pairing_quality', {}), 'roadline_native_pair_success_ratio': float(corridor.get('v40_native_pairing_quality', {}).get('success_ratio', 0.0)), 'roadline_inside_sections_ratio': float(corridor.get('v40_native_pairing_quality', {}).get('inside_ratio', 0.0)), 'cross_section_crossing_count': int(corridor.get('v40_native_pairing_quality', {}).get('final_crossing_count', 0)), 'cross_section_monotonic_ratio': float(corridor.get('v40_native_pairing_quality', {}).get('monotonic_ratio', 0.0)), 'cross_section_width_smoothness': float(corridor.get('v40_native_pairing_quality', {}).get('final_width_smoothness', 0.0)), 'roadline_native_fallback_to_v38': bool(corridor.get('v40_roadline_native_fallback_to_v38', False)), 'lateral_reconstruction_bands': int(corridor.get('v37_post_navigation_pairing', {}).get('lateral_reconstruction', {}).get('modified_station_count', 0)), 'terminal_reconstruction_bands': int(corridor.get('v37_post_navigation_pairing', {}).get('terminal_reconstruction', {}).get('new_terminal_bands', 0)), 'frozen_navigation_preserved': bool(corridor.get('v40_frozen_navigation_preserved', False)), 'v32_direction_field_applied': bool(corridor.get('v32_direction_field_applied', False)), 'auxiliary_terminal_extension_used': bool(corridor.get('v13_reconstruction_summary', {}).get('terminal_extension_count', 0)), 'auxiliary_direction_calibrated': False, 'v13_reconstruction_status': corridor.get('v13_reconstruction_status', 'not_applied'), 'v13_reconstruction_summary': corridor.get('v13_reconstruction_summary', {}), 'auxiliary_road_envelope_polygons': corridor.get('auxiliary_road_envelope_polygons', []), 'auxiliary_road_envelope_area_ratio': float(np.asarray(corridor.get('auxiliary_road_envelope_mask', np.zeros_like(road)), dtype=bool).mean()), 'geometry_metrics': geometry_metrics, 'centerline_roadline_distance_median_px': geometry_metrics.get('centerline_roadline_distance_median_px'), 'centerline_roadline_distance_p90_px': geometry_metrics.get('centerline_roadline_distance_p90_px'), 'centerline_roadline_sample_count': geometry_metrics.get('centerline_roadline_sample_count', 0), 'boundary_turn_p95_deg': geometry_metrics.get('boundary_turn_p95_deg'), 'boundary_continuity_score': geometry_metrics.get('boundary_continuity_score'), 'internal_gap_recovered_pixels': geometry_metrics.get('internal_gap_recovered_pixels', 0), 'internal_gap_recovered_ratio_of_corridor': geometry_metrics.get('internal_gap_recovered_ratio_of_corridor'), 'full_road_corridor_area_ratio': float(full_road_corridor.mean()), 'analysis_geometry_corridor_area_ratio': float(analysis_geometry_corridor.mean()), 'reconstructed_corridor_area_ratio': float(full_road_corridor.mean()), 'inferred_occlusion_area_ratio': float(inferred_occlusion_mask.mean()), 'inferred_internal_gap_area_ratio': float(inferred_internal_gap_mask.mean()), 'visible_road_in_full_ratio': float(np.logical_and(road, full_road_corridor).sum() / max(1, int(road.sum()))), 'valid_obstacle_pixel_ratio': float(valid_obstacles.sum() / max(1, int(obstacles.sum()))), 'tangent_window_px': None, 'raw_centerline_length_px': raw_length, 'fitted_centerline_length_px': fitted_length, 'planned_sections': len(stations), 'valid_sections': len(sections), 'valid_section_ratio': valid_ratio, 'repair_ratio': repair_ratio, 'p90_blockage': p90, 'max_blockage': maximum, 'min_free_width_ratio': min_free_ratio, 'longest_partial_run': partial_run, 'longest_full_run': full_run, 'decision_confidence': decision_confidence['label'], 'decision_confidence_score': decision_confidence['score'], 'decision_confidence_basis': decision_confidence['basis'], 'decision_section_indices': decision_confidence['section_indices'], 'decision_section_source_counts': decision_confidence['source_counts'], 'free_lane_corridor_area_ratio': free_lane_stats['area_ratio'], 'free_lane_continuity_ratio': free_lane_stats['continuity_ratio'], 'free_lane_segment_count': free_lane_stats['segment_count'], 'free_lane_longest_segment_sections': free_lane_stats['longest_segment_sections'], 'raw_path': raw_path.tolist(), 'smooth_path': smooth_path.tolist(), 'stations': stations.tolist(), 'sections': [asdict(section) for section in sections]}

# 【結果工具】建立格式完整的 insufficient 結果，保留失敗原因與既有診斷。
def invalid_result(reason: str, road_area_ratio: float, main_component_ratio: float=0.0, branch_clusters: int=0, raw_length: float=0.0, planned: int=0, valid: int=0, valid_ratio: float=0.0, raw_path: np.ndarray | None=None, smooth_path: np.ndarray | None=None, stations: np.ndarray | None=None, sections: list[CrossSection] | None=None, metadata: dict | None=None) -> dict:
    result = {'prediction': -1, 'centerline_status': 'invalid', 'quality_warnings': [], 'failure_reason': reason, 'road_area_ratio': road_area_ratio, 'main_component_ratio': main_component_ratio, 'branch_clusters': branch_clusters, 'centerline_scale': None, 'centerline_method': 'direction_adaptive_boundary_pair_corridor', 'left_boundary': [], 'right_boundary': [], 'raw_left_boundary': [], 'raw_right_boundary': [], 'row_missing_band_ratio': None, 'row_observed_bands': 0, 'row_span_bands': 0, 'row_median_width_px': None, 'accepted_occlusion_gaps': [], 'one_sided_extrapolations': [], 'one_sided_geometry': [], 'one_side_inferences': [], 'interior_occlusion_inferences': [], 'corridor_source_counts': {}, 'reconstruction_pipeline': 'unified_three_state', 'reconstruction_core_state_counts': {}, 'reconstruction_evidence_counts': {}, 'station_reconstruction_states': [], 'discarded_terminal_fragment_bands': 0, 'row_interpolation_ratio': None, 'row_extrapolation_ratio': None, 'curve_fit_method': None, 'corridor_observation_mode': None, 'direction_geometry_mode': None, 'cross_section_method': None, 'orientation_evidence': None, 'axis_selection_reason': None, 'selected_direction_angle_deg': None, 'selected_track_id': None, 'track_hypotheses': [], 'track_score_margin': None, 'local_direction_refined': False, 'local_direction_refined_sections': 0, 'auxiliary_direction_status': 'not_evaluated', 'auxiliary_direction_mode': 'global_angle_cross_sections', 'auxiliary_direction_evidence_count': 0, 'auxiliary_direction_source_counts': {}, 'auxiliary_direction_evidence': [], 'auxiliary_direction_segments': [], 'auxiliary_direction_refined_sections': 0, 'auxiliary_direction_rejected_sections': 0, 'auxiliary_terminal_extension_used': False, 'auxiliary_direction_calibrated': False, 'navigation_mode': 'insufficient', 'central_roadline_used': False, 'central_roadline_name': 'none', 'central_roadline_coverage': 0.0, 'roadline_native_pairing_quality': {}, 'roadline_native_pair_success_ratio': 0.0, 'roadline_inside_sections_ratio': 0.0, 'cross_section_crossing_count': 0, 'cross_section_monotonic_ratio': 0.0, 'cross_section_width_smoothness': 0.0, 'roadline_native_fallback_to_v38': False, 'lateral_reconstruction_bands': 0, 'terminal_reconstruction_bands': 0, 'frozen_navigation_preserved': False, 'geometry_metrics': {}, 'centerline_roadline_distance_median_px': None, 'centerline_roadline_distance_p90_px': None, 'centerline_roadline_sample_count': 0, 'boundary_turn_p95_deg': None, 'boundary_continuity_score': None, 'internal_gap_recovered_pixels': 0, 'internal_gap_recovered_ratio_of_corridor': None, 'reconstructed_corridor_area_ratio': None, 'full_road_corridor_area_ratio': None, 'analysis_geometry_corridor_area_ratio': None, 'inferred_occlusion_area_ratio': None, 'inferred_internal_gap_area_ratio': None, 'visible_road_in_full_ratio': None, 'valid_obstacle_pixel_ratio': None, 'tangent_window_px': None, 'raw_centerline_length_px': raw_length, 'fitted_centerline_length_px': raw_length, 'planned_sections': planned, 'valid_sections': valid, 'valid_section_ratio': valid_ratio, 'repair_ratio': None, 'p90_blockage': None, 'max_blockage': None, 'min_free_width_ratio': None, 'longest_partial_run': 0, 'longest_full_run': 0, 'decision_confidence': 'insufficient', 'decision_confidence_score': 0.0, 'decision_confidence_basis': None, 'decision_section_indices': [], 'decision_section_source_counts': {}, 'free_lane_corridor_area_ratio': None, 'free_lane_continuity_ratio': None, 'free_lane_segment_count': 0, 'free_lane_longest_segment_sections': 0, 'raw_path': [] if raw_path is None else raw_path.tolist(), 'smooth_path': [] if smooth_path is None else smooth_path.tolist(), 'stations': [] if stations is None else stations.tolist(), 'sections': [] if sections is None else [asdict(section) for section in sections]}
    if metadata:
        result.update(metadata)
    return result

# =============================================================================
# 20. COCO segmentation 解碼、人工答案與分類指標
# =============================================================================
# 將 polygon／RLE 標註轉成 Road、障礙物及輔助遮罩，讀取人工答案與排除
# 原因，並計算 confusion table、Accuracy、Precision、Recall 與 Macro-F1。
# 【COCO 工具】解碼 COCO 壓縮 RLE 的 counts 字串。
def decode_compressed_rle_counts(text: str | bytes) -> list[int]:
    if isinstance(text, bytes):
        text = text.decode('ascii')
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        more = True
        last_chunk = 0
        while more:
            last_chunk = ord(text[position]) - 48
            position += 1
            value |= (last_chunk & 31) << 5 * shift
            more = bool(last_chunk & 32)
            shift += 1
            if not more and last_chunk & 16:
                value |= -1 << 5 * shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts

# 【COCO 工具】在沒有 pycocotools 時將 RLE segmentation 還原成遮罩。
def decode_rle_without_pycocotools(segmentation: dict, height: int, width: int) -> np.ndarray:
    counts = segmentation.get('counts', [])
    if isinstance(counts, (str, bytes)):
        counts = decode_compressed_rle_counts(counts)
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    foreground = False
    for run in counts:
        run = int(run)
        end = position + run
        if run < 0 or end > flat.size:
            raise ValueError('COCO RLE 長度超出影像範圍。')
        if foreground:
            flat[position:end] = 1
        position = end
        foreground = not foreground
    if position != flat.size:
        raise ValueError(f'COCO RLE 總長度 {position} 與影像像素數 {flat.size} 不符。')
    return flat.reshape((height, width), order='F').astype(bool)

# 【COCO 工具】將 COCO polygon 座標填充成二值遮罩。
def decode_polygon_without_pycocotools(segmentation: list, height: int, width: int) -> np.ndarray:
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    polygons = segmentation
    if polygons and polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    for polygon in polygons:
        if len(polygon) >= 6:
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)

# 【COCO 工具】依 segmentation 格式選擇 polygon、RLE 或 pycocotools 解碼。
def decode_segmentation(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get('segmentation')
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        if isinstance(segmentation, dict):
            return decode_rle_without_pycocotools(segmentation, height, width)
        if isinstance(segmentation, list):
            return decode_polygon_without_pycocotools(segmentation, height, width)
        return np.zeros((height, width), dtype=bool)
    else:
        if isinstance(segmentation, list):
            if not segmentation:
                return np.zeros((height, width), dtype=bool)
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = dict(segmentation)
            if isinstance(rle.get('counts'), str):
                rle['counts'] = rle['counts'].encode('utf-8')
        else:
            return np.zeros((height, width), dtype=bool)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

# 【COCO 工具】合併同一影像、同一類別族的所有 annotation 遮罩。
def combine_masks(annotations: list[dict], height: int, width: int) -> np.ndarray:
    combined = np.zeros((height, width), dtype=bool)
    for annotation in annotations:
        combined |= decode_segmentation(annotation, height, width)
    return combined

# 【答案工具】從 COCO image extra 欄位讀取人工阻塞標準答案。
def gt_status(image: dict) -> int | None:
    extra = image.get('extra', {}) or {}
    is_uavg = 'uavg_ground_truth' in extra or 'uavg_ground_truth_status' in extra or 'uavg_original_file_name' in extra
    if is_uavg:
        label = str(extra.get('uavg_ground_truth', '')).strip().lower()
        return {'small_or_clear': 0, 'half': 1, 'full': 2}.get(label)
    tags = set(extra.get('user_tags', []))
    if 'RoadFullyBlocked' in tags:
        return 2
    if 'RoadHalfBlocked' in tags:
        return 1
    return 0

# 【答案工具】取得影像不可評估或缺少答案的排除原因。
def gt_exclusion_reason(image: dict) -> str | None:
    extra = image.get('extra', {}) or {}
    is_uavg = 'uavg_ground_truth' in extra or 'uavg_ground_truth_status' in extra or 'uavg_original_file_name' in extra
    if not is_uavg:
        return None
    label = str(extra.get('uavg_ground_truth', '')).strip().lower()
    if label == 'not_evaluable':
        return 'not_evaluable'
    if label not in {'small_or_clear', 'half', 'full'}:
        return 'missing_ground_truth'
    return None

# 【評估工具】由 confusion matrix 計算 Accuracy、各類 Precision／Recall／F1 與 Macro-F1。
def classification_metrics(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else None
    recalls, precisions, f1s = ([], [], [])
    for index in range(3):
        tp = int(confusion[index, index])
        actual = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    return {'evaluated_images': total, 'accuracy': accuracy, 'macro_precision': float(np.mean(precisions)), 'macro_recall': float(np.mean(recalls)), 'macro_f1': float(np.mean(f1s)), 'per_class_recall': {STATUS_NAMES[i]: recalls[i] for i in range(3)}, 'per_class_precision': {STATUS_NAMES[i]: precisions[i] for i in range(3)}}

# =============================================================================
# 21. 視覺化、資料集評估、報告輸出與命令列入口
# =============================================================================
# draw_*／overlay_* 只產生 D、D_aux_debug 與統計圖，不參與分類；
# evaluate 負責批次推論、彙整 224 張結果並輸出 CSV／JSON／文字摘要。
# 【視覺化】輸出單張道路、障礙物、中心線與橫斷面的基本疊圖。
def draw_overlay(image_path: Path, road: np.ndarray, obstacles: np.ndarray, result: dict, output: Path):
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    if image.size != (road.shape[1], road.shape[0]):
        image = image.resize((road.shape[1], road.shape[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy()
    array[road] = array[road] * 0.7 + np.asarray([0, 180, 70]) * 0.3
    array[obstacles] = array[obstacles] * 0.55 + np.asarray([235, 40, 35]) * 0.45
    canvas = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    path = result.get('smooth_path', [])
    if len(path) >= 2:
        draw.line([tuple(point) for point in path], fill=(255, 230, 0), width=3)
    for section in result.get('sections', []):
        ratio = section['blockage_ratio']
        color = (220, 35, 35) if ratio >= 0.75 else (255, 150, 0) if ratio >= 0.3 else (30, 210, 255)
        draw.line([(section['left_x'], section['left_y']), (section['right_x'], section['right_y'])], fill=color, width=2)
    text = f"status={result['centerline_status']}  pred={STATUS_NAMES[result['prediction']]}  valid={result['valid_section_ratio']:.2f}" if result['prediction'] != -1 else f"status=invalid  reason={result['failure_reason']}"
    draw.rectangle((5, 5, min(canvas.width - 5, 10 + len(text) * 7), 27), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)

# 【視覺化工具】載入可用字型，失敗時回退 Pillow 預設字型。
def diagnostic_font(size: int):
    candidates = [Path('C:\\Windows\\Fonts\\msjh.ttc'), Path('C:\\Windows\\Fonts\\mingliu.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()

# 【視覺化】在原圖疊加 Road 與障礙物 segmentation。
def segmentation_overlay(image: Image.Image, road: np.ndarray, obstacles: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    array[road] = array[road] * 0.65 + np.asarray([0, 210, 80]) * 0.35
    array[obstacles] = array[obstacles] * 0.5 + np.asarray([245, 45, 35]) * 0.5
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

# 【視覺化工具】從結果重建平滑分析走廊遮罩。
def inferred_geometry_envelope_mask(result: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    left_boundary = result.get('left_boundary', [])
    right_boundary = result.get('right_boundary', [])
    if len(left_boundary) >= 2 and len(right_boundary) >= 2:
        polygon = [tuple(map(float, point)) for point in left_boundary]
        polygon += [tuple(map(float, point)) for point in reversed(right_boundary)]
        draw.polygon(polygon, fill=1)
        return np.asarray(canvas, dtype=bool)
    sections = sorted(result.get('sections', []), key=lambda item: int(item['station_index']))
    for section in sections:
        draw.line((int(section['left_x']), int(section['left_y']), int(section['right_x']), int(section['right_y'])), fill=1, width=3)
    for first, second in zip(sections, sections[1:]):
        if int(second['station_index']) - int(first['station_index']) != 1:
            continue
        polygon = [(int(first['left_x']), int(first['left_y'])), (int(first['right_x']), int(first['right_y'])), (int(second['right_x']), int(second['right_y'])), (int(second['left_x']), int(second['left_y']))]
        draw.polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)

# 【視覺化工具】取出輔助物支持的推測道路包絡遮罩。
def auxiliary_road_envelope_mask_from_result(result: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in result.get('auxiliary_road_envelope_polygons', []):
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points = []
        for point in polygon:
            if not isinstance(point, list) or len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)

# 【視覺化工具】由橫斷面自由區間重建主要自由通道遮罩。
def inferred_free_lane_mask(result: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    canvas = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    selected = []
    for section in sorted(result.get('sections', []), key=lambda item: int(item['station_index'])):
        index = section.get('selected_free_interval_index')
        intervals = section.get('free_intervals', [])
        if index is None or not 0 <= int(index) < len(intervals):
            continue
        interval = intervals[int(index)]
        left = (int(interval['left_x']), int(interval['left_y']))
        right = (int(interval['right_x']), int(interval['right_y']))
        draw.line((*left, *right), fill=1, width=2)
        selected.append((section, left, right))
    for first, second in zip(selected, selected[1:]):
        first_section, first_left, first_right = first
        second_section, second_left, second_right = second
        if int(second_section['station_index']) != int(first_section['station_index']) + 1 or second_section.get('free_lane_segment_id') != first_section.get('free_lane_segment_id'):
            continue
        draw.polygon([first_left, first_right, second_right, second_left], fill=1)
    return np.asarray(canvas, dtype=bool)

# 【視覺化】疊加完整道路、分析走廊、自由通道與推測遮擋區。
def road_inclusion_overlay(image: Image.Image, road: np.ndarray, obstacles: np.ndarray, full_corridor: np.ndarray, analysis_corridor: np.ndarray, free_lane: np.ndarray | None=None, inferred_occlusion: np.ndarray | None=None) -> Image.Image:
    array = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    array[analysis_corridor] = array[analysis_corridor] * 0.68 + np.asarray([35, 105, 245]) * 0.32
    accepted = obstacles & full_corridor
    excluded = obstacles & ~full_corridor
    array[excluded] = array[excluded] * 0.6 + np.asarray([185, 45, 35]) * 0.4
    array[accepted] = array[accepted] * 0.35 + np.asarray([245, 20, 210]) * 0.65
    if free_lane is not None:
        array[free_lane] = array[free_lane] * 0.58 + np.asarray([35, 245, 95]) * 0.42
    if np.any(analysis_corridor):
        analysis_interior = binary_box_count(analysis_corridor, 1) == 9
        analysis_boundary = analysis_corridor & ~analysis_interior
        array[analysis_boundary] = np.asarray([35, 95, 255])
    if np.any(road):
        road_interior = binary_box_count(road, 1) == 9
        road_boundary = road & ~road_interior
        array[road_boundary] = np.asarray([0, 235, 245])
    if inferred_occlusion is not None and np.any(inferred_occlusion):
        inferred_interior = binary_box_count(inferred_occlusion, 1) == 9
        inferred_boundary = inferred_occlusion & ~inferred_interior
        array[inferred_boundary] = np.asarray([255, 155, 0])
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

# 【視覺化工具】整理已採用與未採用的輔助 segmentation 顯示遮罩。
def auxiliary_visualization_masks(auxiliary_masks: dict[str, np.ndarray] | None, result: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    masks = validate_auxiliary_masks(auxiliary_masks, shape)
    read_mask = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        read_mask |= mask
    used_mask = np.zeros(shape, dtype=bool)
    if result.get('auxiliary_direction_status') != 'piecewise_applied':
        return (read_mask, used_mask)
    applied_segments = [item for item in result.get('auxiliary_direction_segments', []) if int(item.get('refined_sections', 0)) > 0]
    family_to_mask = {'road_line': 'road_line', 'roadside_boundary': 'roadside_boundary', 'cone_sequence': 'cone'}
    height, width = shape
    for evidence in result.get('auxiliary_direction_evidence', []):
        source_family = str(evidence.get('source_family', ''))
        mask_family = family_to_mask.get(source_family)
        bbox = evidence.get('component_bbox_xyxy')
        if mask_family is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        evidence_start = int(evidence.get('station_start', -1))
        evidence_end = int(evidence.get('station_end', -1))
        participates = any((source_family in segment.get('source_families', []) and evidence_end >= int(segment.get('station_start', 0)) and (evidence_start <= int(segment.get('station_end', -1))) for segment in applied_segments))
        if not participates:
            continue
        x0, y0, x1, y1 = (int(value) for value in bbox)
        x0, x1 = (max(0, x0), min(width - 1, x1))
        y0, y1 = (max(0, y0), min(height - 1, y1))
        if x0 <= x1 and y0 <= y1:
            used_mask[y0:y1 + 1, x0:x1 + 1] |= masks[mask_family][y0:y1 + 1, x0:x1 + 1]
    used_mask &= read_mask
    return (read_mask & ~used_mask, used_mask)

# 【視覺化】在診斷圖疊加各類道路方向輔助物遮罩。
def overlay_auxiliary_direction_masks(panel: Image.Image, auxiliary_masks: dict[str, np.ndarray] | None, result: dict) -> Image.Image:
    shape = (panel.height, panel.width)
    read_only, used = auxiliary_visualization_masks(auxiliary_masks, result, shape)
    array = np.asarray(panel.convert('RGB'), dtype=np.float32).copy()
    if np.any(read_only):
        array[read_only] = array[read_only] * 0.48 + np.asarray([150, 70, 240]) * 0.52
        interior = binary_box_count(read_only, 1) == 9
        array[read_only & ~interior] = np.asarray([175, 105, 255])
    if np.any(used):
        array[used] = array[used] * 0.25 + np.asarray([205, 255, 35]) * 0.75
        interior = binary_box_count(used, 1) == 9
        array[used & ~interior] = np.asarray([225, 255, 55])
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
AUXILIARY_CURVE_COLORS = {'road_line': (255, 65, 235), 'curb': (0, 245, 255), 'guardrail': (45, 125, 255), 'barricade': (255, 75, 45), 'roadside_wall': (80, 255, 95), 'cone': (255, 255, 255), 'roadside_boundary': (70, 210, 255)}

# 【視覺化工具】為未進入實體軌跡的輔助元件建立簡易走向曲線。
def auxiliary_component_direction_curve(points: np.ndarray, p: Parameters) -> list[list[float]]:
    values = np.asarray(points, dtype=np.float64)
    geometry = axial_angle_from_points(values)
    if geometry is None or geometry['span_px'] < float(p.auxiliary_min_component_span_px):
        return []
    axis = np.asarray(geometry['axis_vector_xy'], dtype=np.float64)
    center = values.mean(axis=0)
    projections = (values - center) @ axis
    span = float(projections.max() - projections.min())
    bin_count = int(np.clip(math.ceil(span / 12.0), 4, 48))
    edges = np.linspace(projections.min(), projections.max(), bin_count + 1)
    curve_points: list[np.ndarray] = []
    for index in range(bin_count):
        if index == bin_count - 1:
            keep = (projections >= edges[index]) & (projections <= edges[index + 1])
        else:
            keep = (projections >= edges[index]) & (projections < edges[index + 1])
        if int(keep.sum()) < 2:
            continue
        curve_points.append(np.median(values[keep], axis=0))
    if len(curve_points) < 2:
        return []
    curve = np.asarray(curve_points, dtype=np.float64)
    order = np.argsort((curve - center) @ axis)
    curve = curve[order]
    if len(curve) >= 4:
        weights = np.ones(len(curve), dtype=np.float64)
        curve = np.column_stack([regularized_curve_fit(curve[:, coordinate], weights, 0.8, max(1, int(p.curve_robust_iterations))) for coordinate in range(2)])
    return curve.tolist()

# 【視覺化工具】整理可顯示的輔助候選曲線、類別、角色與採用狀態。
def extract_auxiliary_direction_curves(auxiliary_masks: dict[str, np.ndarray] | None, result: dict, shape: tuple[int, int], p: Parameters) -> list[dict]:
    if not auxiliary_masks:
        return []
    track_package = build_auxiliary_direction_tracks(auxiliary_masks, shape, p)
    canonical_masks = track_package['raw_masks']
    curves: list[dict] = []
    centers = np.asarray(result.get('smooth_path', []), dtype=np.float64)
    left = np.asarray(result.get('left_boundary', []), dtype=np.float64)
    right = np.asarray(result.get('right_boundary', []), dtype=np.float64)
    geometry_count = min(len(centers), len(left), len(right))

    def roadline_role(curve: list[list[float]], family: str) -> str:
        if family != 'road_line' or geometry_count < 2:
            return 'direction'
        curve_values = np.asarray(curve, dtype=np.float64)
        path = centers[:geometry_count]
        distances = np.linalg.norm(curve_values[:, None, :] - path[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        widths = np.linalg.norm(right[:geometry_count] - left[:geometry_count], axis=1)
        fractions = distances[np.arange(len(curve_values)), nearest] / np.maximum(widths[nearest], 1.0)
        return 'center' if float(np.median(fractions)) <= float(p.v13_roadline_single_center_fraction) * 1.35 else 'edge'

    def component_curve(points: np.ndarray) -> list[list[float]]:
        curve = auxiliary_component_direction_curve(points, p)
        if len(curve) >= 2:
            return curve
        geometry = axial_angle_from_points(points)
        if geometry is None:
            return []
        return [geometry['endpoint_a_xy'], geometry['endpoint_b_xy']]
    for track in track_package['accepted']:
        family = str(track['category'])
        curve = component_curve(np.asarray(track['points'], dtype=np.float64))
        if len(curve) < 2:
            continue
        curves.append({'family': family, 'role': roadline_role(curve, family), 'angle_degrees': float(track['angle_degrees']), 'points': curve, 'accepted': True, 'status': str(track['status']), 'fragment_count': int(track['component_count'])})
    for track in track_package['rejected']:
        components = track.get('components') or [track.get('points')]
        for component in components:
            if component is None:
                continue
            curve = component_curve(np.asarray(component, dtype=np.float64))
            if len(curve) < 2:
                continue
            curves.append({'family': str(track['category']), 'role': 'ignored', 'angle_degrees': None, 'points': curve, 'accepted': False, 'status': str(track['status']), 'fragment_count': 1})
    cone_mask = canonical_masks.get('cone')
    if cone_mask is not None and np.any(cone_mask):
        components = connected_component_point_sets(cone_mask, max(2, int(p.auxiliary_min_component_pixels // 2)))
        if len(components) >= 3:
            centroids = np.asarray([component.mean(axis=0) for component in components])
            geometry = axial_angle_from_points(centroids)
            curve: list[list[float]] = []
            if geometry is not None:
                axis = np.asarray(geometry['axis_vector_xy'], dtype=np.float64)
                center = centroids.mean(axis=0)
                order = np.argsort((centroids - center) @ axis)
                curve = centroids[order].tolist()
            if len(curve) >= 2:
                curves.append({'family': 'cone', 'role': 'sequence', 'angle_degrees': None, 'points': curve, 'accepted': True, 'status': 'sequence', 'fragment_count': int(len(components))})
    return curves

# 【視覺化】以類別顏色及線型畫出所有輔助走向曲線。
def overlay_auxiliary_direction_curves(panel: Image.Image, auxiliary_masks: dict[str, np.ndarray] | None, result: dict, p: Parameters) -> Image.Image:
    draw = ImageDraw.Draw(panel, 'RGBA')
    curves = extract_auxiliary_direction_curves(auxiliary_masks, result, (panel.height, panel.width), p)
    font = diagnostic_font(12)
    for curve in curves:
        points = [tuple((int(round(value)) for value in point)) for point in curve['points']]
        if len(points) < 2:
            continue
        accepted = bool(curve.get('accepted', True))
        color = AUXILIARY_CURVE_COLORS.get(curve['family'], (255, 255, 255)) if accepted else (155, 155, 155)
        width = 6 if accepted and curve['family'] == 'road_line' and (curve['role'] == 'center') else 4 if accepted else 2
        draw.line(points, fill=(0, 0, 0, 210), width=width + 3, joint='curve')
        draw.line(points, fill=(*color, 255 if accepted else 205), width=width, joint='curve')
        fragment_count = int(curve.get('fragment_count', 1))
        merge_label = f' merged({fragment_count})' if fragment_count >= 2 else ''
        label = f"{curve['family']}:{curve['role']}{merge_label}" if accepted else f"{curve['family']}:ignored"
        anchor = points[len(points) // 2]
        draw.text((anchor[0] + 5, anchor[1] - 15), label, font=font, fill=(*color, 255), stroke_width=2, stroke_fill=(0, 0, 0, 230))
    return panel

# 【視覺化工具】替診斷子圖加入黑底標題列。
def titled_panel(panel: Image.Image, title: str, title_height: int=42) -> Image.Image:
    result = Image.new('RGB', (panel.width, panel.height + title_height), (25, 25, 28))
    result.paste(panel, (0, title_height))
    draw = ImageDraw.Draw(result)
    draw.text((10, 8), title, font=diagnostic_font(20), fill=(255, 255, 255))
    return result

# 【視覺化工具】沿折線弧長繪製固定實線與空白長度的虛線。
def _draw_dashed_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: tuple[int, int, int, int], width: int=2, dash: float=9.0, gap: float=6.0) -> None:
    for first, second in zip(points[:-1], points[1:]):
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length <= 1e-06:
            continue
        direction = vector / length
        position = 0.0
        while position < length:
            end = min(length, position + dash)
            start_point = a + position * direction
            end_point = a + end * direction
            draw.line((tuple(start_point), tuple(end_point)), fill=fill, width=width)
            position += dash + gap

# 【視覺化工具】補畫有 segmentation 但未進入正式軌跡列表的輔助曲線。
def _v34_unrepresented_auxiliary_debug_tracks(auxiliary_masks: dict[str, np.ndarray] | None, result: dict, p: Parameters, existing_tracks: list[dict]) -> list[dict]:
    if not auxiliary_masks:
        return []
    curves = extract_auxiliary_direction_curves(auxiliary_masks, result, next(iter(auxiliary_masks.values())).shape, p)
    diagonal = max(1.0, math.hypot(next(iter(auxiliary_masks.values())).shape[1], next(iter(auxiliary_masks.values())).shape[0]))
    match_distance = max(6.0, 0.004 * diagonal)

    def already_represented(category: str, points: np.ndarray) -> bool:
        if len(points) < 2:
            return True
        take = np.linspace(0, len(points) - 1, min(80, len(points))).astype(np.int32)
        sampled = points[take, :2]
        for track in existing_tracks:
            track_category = str(track.get('source_category', ''))
            track_family = str(track.get('source_family', ''))
            if category not in {track_category, track_family}:
                continue
            target = np.asarray(track.get('curve', []), dtype=np.float64)
            if target.ndim != 2 or target.shape[1] < 2 or len(target) < 2:
                continue
            target_take = np.linspace(0, len(target) - 1, min(120, len(target))).astype(np.int32)
            distances = np.linalg.norm(sampled[:, None, :] - target[target_take, :2][None, :, :], axis=2)
            if float(np.median(np.min(distances, axis=1))) <= match_distance:
                return True
        return False
    output: list[dict] = []
    for index, curve in enumerate(curves):
        category = str(curve.get('family', 'auxiliary'))
        points = np.asarray(curve.get('points', []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
            continue
        if already_represented(category, points):
            continue
        source_role = str(curve.get('role', 'direction'))
        role = {'center': 'C', 'edge': 'E', 'ignored': 'X'}.get(source_role, 'D')
        adoption = 'candidate' if bool(curve.get('accepted', False)) else 'rejected'
        output.append({'track_id': f'Debug-only auxiliary {index + 1}', 'source_family': category, 'source_category': category, 'role': role, 'adoption': adoption, 'fragment_count': int(curve.get('fragment_count', 1)), 'observed_segments': [points[:, :2].tolist()], 'curve': points[:, :2].tolist(), 'debug_only': True})
    return output

# 【視覺化】依 C／E／D／X 與 USED／CANDIDATE／REJECTED 畫實體軌跡。
def overlay_v33_physical_tracks(panel: Image.Image, tracks: list[dict], debug: bool) -> Image.Image:
    draw = ImageDraw.Draw(panel, 'RGBA')
    font = diagnostic_font(13 if debug else 11)
    label_boxes: list[tuple[int, int, int, int]] = []
    for track in tracks:
        adoption = str(track.get('adoption', 'candidate')).lower()
        role = str(track.get('role', 'D')).upper()
        used = adoption == 'used'
        rejected = adoption == 'rejected' or role == 'X'
        if not debug and (not used):
            continue
        category = str(track.get('source_category', 'auxiliary'))
        base_color = AUXILIARY_CURVE_COLORS.get(category, (255, 255, 255))
        color = (145, 145, 145) if rejected else base_color
        alpha = 110 if not used and (not rejected) else 205 if rejected else 255
        width = 6 if used and role == 'C' else 5 if used else 2
        curve = [tuple((int(round(value)) for value in point[:2])) for point in track.get('curve', []) if len(point) >= 2]
        if len(curve) < 2:
            continue
        if used:
            draw.line(curve, fill=(0, 0, 0, 150), width=width + 2, joint='curve')
            _draw_dashed_polyline(draw, curve, (*color, min(alpha, 210)), width=max(2, width - 2), dash=7.0, gap=5.0)
            observed_segments = track.get('observed_segments', [])
            for segment in observed_segments:
                observed = [tuple((int(round(value)) for value in point[:2])) for point in segment if len(point) >= 2]
                if len(observed) >= 2:
                    draw.line(observed, fill=(*color, 255), width=width, joint='curve')
        else:
            _draw_dashed_polyline(draw, curve, (*color, alpha), width=width, dash=4.0 if rejected else 9.0, gap=5.0 if rejected else 7.0)
        anchor = curve[len(curve) // 2]
        if debug:
            status = 'USED' if used else 'REJECTED' if rejected else 'CANDIDATE'
            source_name = {'road_line': 'roadline', 'curb': 'curb', 'guardrail': 'guardrail', 'barricade': 'barricade', 'roadside_wall': 'roadside_wall', 'cone': 'cone_sequence'}.get(category, category.lower().replace(' ', '_'))
            role_name = {'C': 'center', 'E': 'edge', 'D': 'direction', 'X': 'rejected'}.get(role, 'direction')
            first_line = f'{source_name}_{role_name}' if category == 'road_line' else source_name
            group_name = track.get('roadline_group')
            group_order = track.get('roadline_group_order')
            group_line = f'{group_name} order={int(group_order) + 1}' if category == 'road_line' and group_name is not None and (group_order is not None) else ''
            track_id = str(track.get('track_id', 'Auxiliary Track'))
            label = first_line + '\n' + f'{track_id} · {role} · {status}'
            if group_line:
                label += '\n' + group_line
            label_x = int(anchor[0] + 12)
            label_y = int(anchor[1] - 22)
            bounds = draw.multiline_textbbox((label_x, label_y), label, font=font, stroke_width=2, spacing=2)
            attempts = 0
            while any((not (bounds[2] < box[0] or bounds[0] > box[2] or bounds[3] < box[1] or (bounds[1] > box[3])) for box in label_boxes)) and attempts < 10:
                label_y += 18 if attempts % 2 == 0 else -36
                bounds = draw.multiline_textbbox((label_x, label_y), label, font=font, stroke_width=2, spacing=2)
                attempts += 1
            label_boxes.append(tuple((int(value) for value in bounds)))
            draw.multiline_text((label_x, label_y), label, font=font, fill=(*color, 255), stroke_width=2, stroke_fill=(0, 0, 0, 235), spacing=2)
    return panel

# 【視覺化】繪製 V40 的道路面、中心、邊界、橫斷面與輔助軌跡。
def draw_new_geometry_panel(image: Image.Image, road: np.ndarray, obstacles: np.ndarray, new: dict, p: Parameters, full_corridor: np.ndarray, analysis_corridor: np.ndarray, free_lane: np.ndarray | None=None, inferred_occlusion: np.ndarray | None=None, auxiliary_masks: dict[str, np.ndarray] | None=None, auxiliary_debug: bool=False) -> Image.Image:
    panel = road_inclusion_overlay(image, road, obstacles, full_corridor, analysis_corridor, free_lane=free_lane, inferred_occlusion=inferred_occlusion)
    panel = overlay_auxiliary_direction_masks(panel, auxiliary_masks, new)
    v33_tracks = list(new.get('v33_physical_tracks', []))
    if v33_tracks:
        display_tracks = list(v33_tracks)
        if auxiliary_debug:
            display_tracks.extend(_v34_unrepresented_auxiliary_debug_tracks(auxiliary_masks, new, p, v33_tracks))
        panel = overlay_v33_physical_tracks(panel, display_tracks, debug=bool(auxiliary_debug))
    else:
        panel = overlay_auxiliary_direction_curves(panel, auxiliary_masks, new, p)
    draw = ImageDraw.Draw(panel, 'RGBA')
    for curve in [] if v33_tracks else new.get('v32_authoritative_curves', []):
        points = [tuple((int(round(value)) for value in point)) for point in curve.get('points', [])]
        if len(points) < 2:
            continue
        category = str(curve.get('source_category', 'auxiliary'))
        family = str(curve.get('source_family', category))
        color = AUXILIARY_CURVE_COLORS.get(category, AUXILIARY_CURVE_COLORS.get(family, (255, 255, 255)))
        draw.line(points, fill=(0, 0, 0, 235), width=11, joint='curve')
        draw.line(points, fill=(*color, 255), width=7, joint='curve')
    left_boundary = new.get('left_boundary', [])
    right_boundary = new.get('right_boundary', [])
    if len(left_boundary) >= 2:
        draw.line([tuple(point) for point in left_boundary], fill=(255, 255, 255, 150), width=2)
    if len(right_boundary) >= 2:
        draw.line([tuple(point) for point in right_boundary], fill=(255, 255, 255, 150), width=2)
    for key in ('raw_left_boundary', 'raw_right_boundary'):
        for x, y in new.get(key, []):
            radius = 2
            draw.ellipse((int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius), fill=(80, 255, 105, 235))
    path = new.get('smooth_path', [])
    v32_direction_display = bool(new.get('v32_direction_field_applied', False))
    navigation_stations = np.asarray(new.get('stations', []), dtype=np.float64)
    center_source_labels = list(new.get('v36_center_source_labels', []))
    v36_center_display = len(navigation_stations) >= 2 and len(center_source_labels) == len(navigation_stations)
    if v36_center_display:
        run_start = 0
        for run_end in range(1, len(center_source_labels) + 1):
            if run_end < len(center_source_labels) and center_source_labels[run_end] == center_source_labels[run_start]:
                continue
            lo = max(0, run_start - (1 if run_start else 0))
            points = [tuple((float(value) for value in point)) for point in navigation_stations[lo:run_end]]
            label = center_source_labels[run_start]
            if len(points) >= 2:
                if label == 'roadline_center':
                    draw.line(points, fill=(0, 0, 0, 175), width=8, joint='curve')
                    draw.line(points, fill=(255, 235, 0, 255), width=6, joint='curve')
                elif label == 'transition':
                    draw.line(points, fill=(0, 0, 0, 155), width=7, joint='curve')
                    draw.line(points, fill=(255, 125, 0, 255), width=5, joint='curve')
                else:
                    _draw_dashed_polyline(draw, points, (255, 235, 0, 225), width=3, dash=9.0, gap=7.0)
            run_start = run_end
        if auxiliary_debug:
            point_stride = max(1, int(math.ceil(len(navigation_stations) / 70)))
            for index in range(0, len(navigation_stations), point_stride):
                x, y = navigation_stations[index]
                label = center_source_labels[index]
                if label == 'roadline_center':
                    fill, outline, radius = ((255, 235, 0, 255), (0, 0, 0, 220), 3)
                elif label == 'transition':
                    fill, outline, radius = ((255, 125, 0, 255), (0, 0, 0, 220), 3)
                else:
                    fill, outline, radius = ((30, 30, 30, 180), (255, 245, 190, 235), 2)
                draw.ellipse((int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius), fill=fill, outline=outline, width=1)
    elif v32_direction_display:
        if len(path):
            _draw_dashed_polyline(draw, [tuple(point) for point in path], (255, 235, 0, 230), width=3, dash=9.0, gap=7.0)
    elif len(path) >= 2:
        draw.line([tuple(point) for point in path], fill=(255, 235, 0, 255), width=4)
    navigation_tangents = np.asarray(new.get('v32_navigation_tangents', []), dtype=np.float64)
    arrow_count = min(len(navigation_stations), len(navigation_tangents))
    if arrow_count:
        arrow_stride = max(1, int(math.ceil(arrow_count / 8)))
        for index in range(arrow_stride // 2, arrow_count, arrow_stride):
            origin = navigation_stations[index]
            tangent = navigation_tangents[index]
            norm = float(np.linalg.norm(tangent))
            if norm <= 1e-06:
                continue
            tangent = tangent / norm
            start = origin - 8.0 * tangent
            tip = origin + 9.0 * tangent
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            wing_a = tip - 6.0 * tangent + 4.0 * normal
            wing_b = tip - 6.0 * tangent - 4.0 * normal
            draw.line((tuple(start), tuple(tip)), fill=(255, 250, 115, 250), width=3)
            draw.polygon([tuple(tip), tuple(wing_a), tuple(wing_b)], fill=(255, 250, 115, 250))
    authority_summary = new.get('v35_native_center_summary', {})
    if bool(authority_summary.get('geometric_authority', False)):
        gate_font = diagnostic_font(12)
        for gate_name in ('start', 'end'):
            point = np.asarray(authority_summary.get(f'authority_{gate_name}_point', []), dtype=np.float64)
            tangent = np.asarray(authority_summary.get(f'authority_{gate_name}_tangent', []), dtype=np.float64)
            if point.shape != (2,) or tangent.shape != (2,):
                continue
            tangent /= max(float(np.linalg.norm(tangent)), 1e-06)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            first = point - 34.0 * normal
            second = point + 34.0 * normal
            draw.line((tuple(first), tuple(second)), fill=(255, 190, 0, 255), width=5)
            if auxiliary_debug:
                draw.text(tuple(second + np.asarray([5.0, -8.0])), f'Roadline center {gate_name.upper()}', font=gate_font, fill=(255, 210, 60, 255), stroke_width=2, stroke_fill=(0, 0, 0, 235))
    for geometry in new.get('one_sided_geometry', []):
        extrapolated_path = geometry.get('centerline', [])
        extrapolated_left = geometry.get('left', [])
        extrapolated_right = geometry.get('right', [])
        if len(extrapolated_left) >= 2:
            draw.line([tuple(point) for point in extrapolated_left], fill=(255, 120, 0, 245), width=4)
        if len(extrapolated_right) >= 2:
            draw.line([tuple(point) for point in extrapolated_right], fill=(255, 120, 0, 245), width=4)
        if len(extrapolated_path) >= 2:
            draw.line([tuple(point) for point in extrapolated_path], fill=(255, 75, 0, 255), width=6)
    sections = new.get('sections', [])
    display_stride = max(1, int(math.ceil(len(sections) / 40)))
    maximum_index = None
    if sections:
        maximum_index = max(sections, key=lambda item: float(item['blockage_ratio']))['station_index']
    for section in sections:
        if int(section['station_index']) % display_stride != 0 and section['station_index'] != maximum_index:
            continue
        ratio = float(section['blockage_ratio'])
        geometry_source = str(section.get('geometry_source', 'unknown'))
        if auxiliary_debug and geometry_source == 'paired_boundaries':
            color = (20, 215, 255, 235)
        elif auxiliary_debug and geometry_source == 'neighbor_interpolated':
            color = (20, 215, 255, 180)
        elif auxiliary_debug and geometry_source == 'v12_boundary_pair':
            color = (255, 255, 255, 205)
        elif auxiliary_debug and geometry_source == 'image_normal_fallback':
            color = (145, 145, 145, 190)
        elif ratio >= p.full_threshold:
            color = (245, 35, 35, 230)
        elif ratio >= p.partial_threshold:
            color = (255, 145, 0, 225)
        else:
            color = (20, 205, 255, 205)
        left = (int(section['left_x']), int(section['left_y']))
        right = (int(section['right_x']), int(section['right_y']))
        if auxiliary_debug and geometry_source in {'neighbor_interpolated', 'image_normal_fallback'}:
            _draw_dashed_polyline(draw, [left, right], color, width=3, dash=7.0 if geometry_source == 'neighbor_interpolated' else 4.0, gap=5.0)
        else:
            draw.line((left, right), fill=color, width=3)
        radius = 3
        for x, y in (left, right):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 245))
    return panel

# 【視覺化工具】繪製阻塞率圖中的水平虛線門檻。
def dashed_horizontal(draw: ImageDraw.ImageDraw, xy, fill, dash: int=10, width: int=2):
    x0, y, x1, _ = xy
    x = x0
    while x < x1:
        draw.line((x, y, min(x + dash, x1), y), fill=fill, width=width)
        x += dash * 2

# 【視覺化】繪製沿道路站序的阻塞率曲線及 P90／分類門檻。
def draw_blockage_curve(new: dict, size: tuple[int, int], p: Parameters) -> Image.Image:
    width, height = size
    panel = Image.new('RGB', size, (247, 247, 244))
    draw = ImageDraw.Draw(panel)
    font = diagnostic_font(16)
    small = diagnostic_font(14)
    left, right, top, bottom = (65, width - 25, 30, height - 55)
    draw.rectangle((left, top, right, bottom), outline=(60, 60, 60), width=2)

    def y_for(value: float) -> int:
        return int(round(bottom - np.clip(value, 0.0, 1.0) * (bottom - top)))
    for value, label, color in [(0.0, '0.00', (110, 110, 110)), (p.partial_threshold, '0.30', (235, 135, 0)), (p.full_threshold, '0.75', (220, 40, 40)), (1.0, '1.00', (110, 110, 110))]:
        y = y_for(value)
        dashed_horizontal(draw, (left, y, right, y), fill=color, dash=8, width=2)
        draw.text((8, y - 9), label, font=small, fill=color)
    planned = int(new.get('planned_sections', 0))
    by_index = {int(item['station_index']): float(item['blockage_ratio']) for item in new.get('sections', [])}
    points: list[tuple[int, int] | None] = []
    for index in range(planned):
        if index not in by_index:
            points.append(None)
            continue
        x = left if planned <= 1 else int(round(left + index * (right - left) / (planned - 1)))
        points.append((x, y_for(by_index[index])))
    previous = None
    for point in points:
        if point is None:
            previous = None
            continue
        if previous is not None:
            draw.line((previous, point), fill=(55, 85, 190), width=3)
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(30, 60, 170))
        previous = point
    p90 = new.get('p90_blockage')
    if p90 is not None:
        y = y_for(float(p90))
        dashed_horizontal(draw, (left, y, right, y), fill=(20, 145, 75), dash=6, width=3)
        draw.text((max(left, right - 130), max(top, y - 21)), f'P90={p90:.3f}', font=small, fill=(10, 110, 55))
    draw.text((left, bottom + 12), 's=0', font=small, fill=(50, 50, 50))
    draw.text((right - 40, bottom + 12), 's=1', font=small, fill=(50, 50, 50))
    if planned == 0:
        reason = new.get('failure_reason') or 'no valid cross-sections'
        draw.text((left + 20, top + 35), f'INSUFFICIENT: {reason}', font=font, fill=(185, 30, 30))
    return panel

# 【視覺化】將單張影像的預測、信心、幾何來源與警告排成文字面板。
def draw_statistics_panel(record: dict, selection_reason: str, size: tuple[int, int], inclusion_stats: dict | None=None) -> Image.Image:
    panel = Image.new('RGB', size, (35, 37, 42))
    draw = ImageDraw.Draw(panel)
    font = diagnostic_font(19)
    small = diagnostic_font(16)
    new = record['new']
    inclusion_stats = inclusion_stats or {}

    def fmt(value):
        return 'N/A' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)
    lines = [('Selection', selection_reason), ('Image ID', record['image_id']), ('Ground truth', record['ground_truth_name']), ('V40 prediction', STATUS_NAMES[new['prediction']]), ('Decision confidence', new.get('decision_confidence', '-')), ('Confidence score', fmt(new.get('decision_confidence_score'))), ('P90', fmt(new.get('p90_blockage'))), ('Max', fmt(new.get('max_blockage'))), ('Min free ratio', fmt(new.get('min_free_width_ratio'))), ('Free-lane continuity', fmt(new.get('free_lane_continuity_ratio'))), ('Partial run', new.get('longest_partial_run', 0)), ('Full run', new.get('longest_full_run', 0)), ('Valid sections', f"{new.get('valid_sections', 0)}/{new.get('planned_sections', 0)}"), ('Valid ratio', fmt(new.get('valid_section_ratio'))), ('Centerline', new.get('centerline_method', 'legacy_skeleton')), ('Road observation', new.get('corridor_observation_mode') or '-'), ('Axis reason', new.get('axis_selection_reason') or '-'), ('Cross-sections', new.get('cross_section_method') or '-'), ('Aux direction', new.get('auxiliary_direction_status') or '-'), ('Aux evidence', new.get('auxiliary_direction_evidence_count', 0)), ('Aux refined', new.get('auxiliary_direction_refined_sections', 0)), ('Reconstruction flow', new.get('reconstruction_pipeline', 'unified_three_state')), ('Core states', json.dumps(new.get('reconstruction_core_state_counts', {}), ensure_ascii=False)), ('Support evidence', json.dumps(new.get('reconstruction_evidence_counts', {}), ensure_ascii=False)), ('V13 reconstruction', new.get('v13_reconstruction_status', '-')), ('V13 roadline gaps', new.get('v13_reconstruction_summary', {}).get('roadline_internal_bands', 0)), ('Roadline center mode', new.get('v13_reconstruction_summary', {}).get('roadline_center_mode', 'road_geometry_fallback')), ('V13 boundary anchors', new.get('v13_reconstruction_summary', {}).get('boundary_anchor_observations', 0)), ('V13 terminal bands', new.get('v13_reconstruction_summary', {}).get('terminal_extension_bands', 0)), ('Center to Road line', fmt(new.get('centerline_roadline_distance_median_px'))), ('Boundary continuity', fmt(new.get('boundary_continuity_score'))), ('Recovered gap pixels', new.get('internal_gap_recovered_pixels', 0)), ('Center status', new.get('centerline_status', '-')), ('Interpolation ratio', fmt(new.get('row_interpolation_ratio'))), ('Extrapolation ratio', fmt(new.get('row_extrapolation_ratio'))), ('Interior occlusion bands', len(new.get('interior_occlusion_inferences', []))), ('One-sided segments', len(new.get('one_sided_extrapolations', []))), ('Discarded end fragments', new.get('discarded_terminal_fragment_bands', 0)), ('Obstacle in corridor', fmt(inclusion_stats.get('accepted_obstacle_ratio'))), ('Failure', new.get('failure_reason') or '-'), ('Warnings', ', '.join(new.get('quality_warnings', [])) or '-')]
    y = 18
    for label, value in lines:
        draw.text((20, y), f'{label}:', font=small, fill=(150, 205, 255))
        draw.text((185, y), str(value), font=font if label in {'Ground truth', 'V40 prediction'} else small, fill=(255, 255, 255))
        y += 27
        if y > size[1] - 25:
            break
    return panel

# 【案例挑選】取得診斷排序使用的主要阻塞率數值。
def diagnostic_value(record: dict) -> float:
    value = record['new'].get('p90_blockage')
    return -1.0 if value is None else float(value)

# 【案例挑選】從排序候選中均勻挑選指定數量，避免只集中在相近案例。
def spread_pick(candidates: list[dict], count: int) -> list[dict]:
    candidates = sorted(candidates, key=lambda item: (diagnostic_value(item), item['image_id']))
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count)
    return [candidates[int(round(index))] for index in indices]

# 【案例挑選】依要求案例、錯誤類型、信心與失敗原因選擇診斷影像。
def select_diagnostic_records(records: list[dict], limit: int) -> list[tuple[dict, str]]:
    limit = max(0, int(limit))
    selected: list[tuple[dict, str]] = []
    used: set[int] = set()

    def add(reason: str, candidates: list[dict], quota: int, spread: bool=True):
        available = [item for item in candidates if int(item['image_id']) not in used]
        picks = spread_pick(available, min(quota, max(0, limit - len(selected)))) if spread else available[:quota]
        for item in picks:
            if len(selected) >= limit:
                break
            selected.append((item, reason))
            used.add(int(item['image_id']))
    add('full_to_half', [r for r in records if r['ground_truth'] == 2 and r['new']['prediction'] == 1], 8)
    add('full_to_clear', [r for r in records if r['ground_truth'] == 2 and r['new']['prediction'] == 0], 5)
    insufficient = sorted([r for r in records if r['new']['prediction'] < 0], key=lambda r: (-r['ground_truth'], str(r['new'].get('failure_reason')), r['image_id']))
    add('insufficient', insufficient, 5, spread=False)
    add('clear_overestimated', [r for r in records if r['ground_truth'] == 0 and r['new']['prediction'] in (1, 2)], 4)
    add('half_misclassified', [r for r in records if r['ground_truth'] == 1 and r['new']['prediction'] >= 0 and (r['new']['prediction'] != 1)], 3)
    add('branch_warning_error', [r for r in records if 'many_skeleton_branches' in r['new'].get('quality_warnings', []) and r['new']['prediction'] >= 0 and (r['new']['prediction'] != r['ground_truth'])], 3)
    add('correct_control', [r for r in records if r['new']['prediction'] == r['ground_truth']], 2)
    add('other_misclassified', [r for r in records if r['new']['prediction'] >= 0 and r['new']['prediction'] != r['ground_truth']], limit)
    add('other_insufficient', [r for r in records if r['new']['prediction'] < 0], limit, spread=False)
    add('other_warning', [r for r in records if r['new'].get('quality_warnings')], limit)
    add('other_control', records, limit)
    return selected[:limit]

# 【視覺化】組合指定版型的 A～F 子圖並另存 D_aux_debug。
def make_diagnostic_panel(image_path: Path, road: np.ndarray, obstacles: np.ndarray, record: dict, selection_reason: str, output: Path, p: Parameters, layout: str='six', auxiliary_masks: dict[str, np.ndarray] | None=None, auxiliary_debug_output: Path | None=None):
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    expected_size = (road.shape[1], road.shape[0])
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.BILINEAR)
    new = record['new']
    analysis_corridor = inferred_geometry_envelope_mask(new, road.shape)
    inferred_occlusion = np.asarray(new.get('inferred_occlusion_mask', analysis_corridor & obstacles & ~road), dtype=bool) & ~road
    inferred_internal = np.asarray(new.get('inferred_internal_gap_mask', np.zeros_like(road)), dtype=bool) & ~road
    auxiliary_road_envelope = auxiliary_road_envelope_mask_from_result(new, road.shape) & ~road
    corridor = road | inferred_occlusion | inferred_internal
    free_lane = inferred_free_lane_mask(new, road.shape)
    accepted_pixels = int(np.logical_and(obstacles, corridor).sum())
    total_obstacle_pixels = int(obstacles.sum())
    inclusion_stats = {'accepted_obstacle_pixels': accepted_pixels, 'total_obstacle_pixels': total_obstacle_pixels, 'accepted_obstacle_ratio': accepted_pixels / total_obstacle_pixels if total_obstacle_pixels else 0.0}
    prediction_name = STATUS_NAMES[int(new['prediction'])]
    if int(new['prediction']) < 0:
        p90_text = 'n/a'
        max_text = 'n/a'
    else:
        p90_text = f"{float(new.get('p90_blockage', 0.0) or 0.0):.3f}"
        max_text = f"{float(new.get('max_blockage', 0.0) or 0.0):.3f}"
    panel_d = titled_panel(draw_new_geometry_panel(image, road, obstacles, new, p, corridor, analysis_corridor, free_lane=free_lane, inferred_occlusion=inferred_occlusion | inferred_internal, auxiliary_masks=auxiliary_masks), f"D. GT={record['ground_truth_name']} | V40={prediction_name} | P90 blockage={p90_text} | max={max_text}")
    if auxiliary_debug_output is not None:
        debug_panel = titled_panel(draw_new_geometry_panel(image, road, obstacles, new, p, corridor, analysis_corridor, free_lane=free_lane, inferred_occlusion=inferred_occlusion | inferred_internal, auxiliary_masks=auxiliary_masks, auxiliary_debug=True), 'D_aux_debug. Auxiliary direction curves and adoption status')
        auxiliary_debug_output.parent.mkdir(parents=True, exist_ok=True)
        debug_panel.save(auxiliary_debug_output, quality=94)
    panel_e = titled_panel(draw_blockage_curve(new, image.size, p), 'E. Blockage ratio along normalized road position s')
    panel_f = titled_panel(draw_statistics_panel(record, selection_reason, image.size, inclusion_stats), 'F. Decision diagnostics')
    if layout == 'd':
        panels = [panel_d]
        columns = 1
    elif layout == 'cdef':
        panels = [panel_d, panel_e, panel_f]
        columns = 2
    elif layout == 'abde':
        panels = [titled_panel(image.copy(), 'A. Original image'), titled_panel(segmentation_overlay(image, road, obstacles), 'B. Segmentation: Road=green, Obstacles=red'), panel_d, panel_e]
        columns = 2
    elif layout == 'bd':
        panels = [titled_panel(segmentation_overlay(image, road, obstacles), 'B. Original + segmentation: Road=green, Obstacles=red'), panel_d]
        columns = 2
    elif layout == 'six':
        panels = [titled_panel(image.copy(), 'A. Original image'), titled_panel(segmentation_overlay(image, road, obstacles), 'B. Segmentation: Road=green, Obstacles=red'), panel_d, panel_e, panel_f]
        columns = 3
    else:
        raise ValueError(f'Unknown diagnostic layout: {layout}')
    panel_width, panel_height = panels[0].size
    rows = int(math.ceil(len(panels) / columns))
    canvas = Image.new('RGB', (panel_width * columns, panel_height * rows), (18, 18, 20))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index % columns * panel_width, index // columns * panel_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)

def generate_diagnostics(args, coco: dict, grouped: dict[int, list[dict]], road_id: int, obstacle_ids: set[int], auxiliary_category_ids: dict[str, set[int]], records: list[dict], p: Parameters):
    # 【主要功能：依既有逐圖結果產生 D 與 D_aux_debug】
    # 重新讀取原圖及 segmentation，只負責繪圖與案例挑選，不重新執行
    # 道路分類，也不會改變 per_image_results.json 中的 prediction。
    if args.diagnostics_all:
        selected = [(record, f"{record['ground_truth_name']}_to_{STATUS_NAMES[int(record['new']['prediction'])]}") for record in records[:args.diagnostic_limit]]
        directory_name = f'diagnostics_all{len(selected)}'
    else:
        selected = select_diagnostic_records(records, args.diagnostic_limit)
        directory_name = f'diagnostics_top{len(selected)}'
    if args.diagnostic_layout in {'d', 'cdef', 'abde', 'bd'}:
        directory_name += f'_{args.diagnostic_layout}'
    diagnostic_dir = args.output / directory_name
    auxiliary_debug_dir = args.output / 'D_aux_debug'
    image_lookup = {int(item['id']): item for item in coco['images']}
    rows = []
    for rank, (record, reason) in enumerate(selected, 1):
        image_id = int(record['image_id'])
        image_info = image_lookup[image_id]
        annotations = grouped.get(image_id, [])
        road = combine_masks([item for item in annotations if int(item['category_id']) == road_id], int(image_info['height']), int(image_info['width']))
        obstacles = combine_masks([item for item in annotations if int(item['category_id']) in obstacle_ids], int(image_info['height']), int(image_info['width']))
        auxiliary_masks = {family: combine_masks([item for item in annotations if int(item['category_id']) in category_ids], int(image_info['height']), int(image_info['width'])) for family, category_ids in auxiliary_category_ids.items() if category_ids}
        analysis_shape = tuple((int(value) for value in record['new'].get('analysis_shape', road.shape)))
        if len(analysis_shape) == 2 and analysis_shape != road.shape:
            road = resize_boolean_mask_to_shape(road, analysis_shape)
            obstacles = resize_boolean_mask_to_shape(obstacles, analysis_shape)
            auxiliary_masks = {family: resize_boolean_mask_to_shape(mask, analysis_shape) for family, mask in auxiliary_masks.items()}
        filename = f'{rank:02d}_{reason}_id{image_id:05d}.jpg'
        make_diagnostic_panel(args.images_dir / image_info['file_name'], road, obstacles, record, reason, diagnostic_dir / filename, p, layout=args.diagnostic_layout, auxiliary_masks=auxiliary_masks, auxiliary_debug_output=auxiliary_debug_dir / filename)
        rows.append({'rank': rank, 'selection_reason': reason, 'image_id': image_id, 'source_filename': image_info['file_name'], 'diagnostic_filename': filename, 'ground_truth': record['ground_truth_name'], 'v40_prediction': STATUS_NAMES[record['new']['prediction']], 'p90_blockage': record['new'].get('p90_blockage'), 'max_blockage': record['new'].get('max_blockage'), 'min_free_width_ratio': record['new'].get('min_free_width_ratio'), 'failure_reason': record['new'].get('failure_reason'), 'quality_warnings': ';'.join(record['new'].get('quality_warnings', []))})
        print(f'Diagnostic: {rank}/{len(selected)} -> {filename}', flush=True)
    with (diagnostic_dir / 'diagnostic_selection.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['rank'])
        writer.writeheader()
        writer.writerows(rows)
    (diagnostic_dir / 'diagnostic_selection.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Diagnostics output: {diagnostic_dir}')

# 【輸出工具】將浮點數統一四捨五入，其他資料型別保持不變。
def safe_number(value):
    if isinstance(value, float):
        return round(value, 6)
    return value

def evaluate(args, p: Parameters):
    # 【主要功能：資料集批次評估與報告輸出】
    # 讀取 COCO、排除不可評估影像、逐張呼叫 road_aligned_method，彙整
    # confusion table、Accuracy、Macro-F1、失敗原因與幾何統計並輸出檔案。
    coco = json.loads(args.coco.read_text(encoding='utf-8-sig'))
    categories = {normalized_category_name(item['name']): item['id'] for item in coco['categories']}
    auxiliary_category_ids = resolve_auxiliary_category_ids(coco['categories'])
    if 'road' not in categories:
        raise ValueError('COCO categories 找不到 Road 類別。')
    road_id = categories['road']
    if normalized_category_name(args.obstacle_categories) == 'auto':
        obstacle_families = resolve_obstacle_category_ids(coco['categories'])
        obstacle_ids = {category_id for category_ids in obstacle_families.values() for category_id in category_ids}
        requested_obstacles = [family for family, category_ids in obstacle_families.items() if category_ids]
        if not obstacle_ids:
            raise ValueError('COCO categories 找不到任何支援的道路障礙物類別。')
    else:
        requested_obstacles = [normalized_category_name(name) for name in args.obstacle_categories.split(',') if name.strip()]
        missing = [name for name in requested_obstacles if name not in categories]
        if missing:
            raise ValueError(f'COCO categories 找不到障礙物類別：{missing}')
        obstacle_ids = {categories[name] for name in requested_obstacles}
    source_annotations = coco['annotations']
    if args.predictions:
        source_annotations = json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    grouped: dict[int, list[dict]] = {}
    for annotation in source_annotations:
        if float(annotation.get('score', 1.0)) < args.score_threshold:
            continue
        grouped.setdefault(int(annotation['image_id']), []).append(annotation)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.diagnostics_only:
        results_path = args.diagnostics_results if args.diagnostics_results is not None else args.output / 'per_image_results.json'
        if not results_path.is_file():
            raise ValueError(f'diagnostics-only 找不到既有結果：{results_path}')
        records = json.loads(results_path.read_text(encoding='utf-8-sig'))
        if args.image_ids:
            requested_ids = {int(value.strip()) for value in args.image_ids.split(',') if value.strip()}
            records = [record for record in records if int(record['image_id']) in requested_ids]
            found_ids = {int(record['image_id']) for record in records}
            missing_ids = sorted(requested_ids - found_ids)
            if missing_ids:
                raise ValueError(f'既有逐圖結果找不到 image ids：{missing_ids}')
        generate_diagnostics(args, coco, grouped, road_id, obstacle_ids, auxiliary_category_ids, records, p)
        return
    overlay_dir = args.output / 'overlays'
    new_confusion = np.zeros((3, 3), dtype=np.int64)
    new_insufficient_by_actual = np.zeros(3, dtype=np.int64)
    records = []
    images = coco['images']
    if args.image_ids:
        requested_ids = {int(value.strip()) for value in args.image_ids.split(',') if value.strip()}
        images = [item for item in images if int(item['id']) in requested_ids]
        found_ids = {int(item['id']) for item in images}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f'COCO 找不到 image ids：{missing_ids}')
    considered_image_count = len(images)
    excluded_ground_truth: list[dict] = []
    ground_truth_by_image_id: dict[int, int] = {}
    evaluable_images = []
    for image_info in images:
        actual = gt_status(image_info)
        if actual is None:
            excluded_ground_truth.append({'image_id': int(image_info['id']), 'file_name': image_info.get('file_name'), 'reason': gt_exclusion_reason(image_info) or 'missing_ground_truth', 'ground_truth': (image_info.get('extra', {}) or {}).get('uavg_ground_truth')})
            continue
        image_id = int(image_info['id'])
        ground_truth_by_image_id[image_id] = int(actual)
        evaluable_images.append(image_info)
    eligible_image_count_before_limit = len(evaluable_images)
    images = evaluable_images
    if args.limit:
        images = images[:args.limit]
    for number, image_info in enumerate(images, 1):
        image_id = int(image_info['id'])
        height, width = (int(image_info['height']), int(image_info['width']))
        annotations = grouped.get(image_id, [])
        road_annotations = [item for item in annotations if int(item['category_id']) == road_id]
        obstacle_annotations = [item for item in annotations if int(item['category_id']) in obstacle_ids]
        road = combine_masks(road_annotations, height, width)
        obstacles = combine_masks(obstacle_annotations, height, width)
        auxiliary_masks = {family: combine_masks([item for item in annotations if int(item['category_id']) in category_ids], height, width) for family, category_ids in auxiliary_category_ids.items() if category_ids}
        original_shape = (height, width)
        target_shape, analysis_scale = analysis_target_shape(original_shape, args.analysis_long_edge)
        if target_shape != original_shape:
            road = resize_boolean_mask_to_shape(road, target_shape)
            obstacles = resize_boolean_mask_to_shape(obstacles, target_shape)
            auxiliary_masks = {family: resize_boolean_mask_to_shape(mask, target_shape) for family, mask in auxiliary_masks.items()}
        analysis_parameters = scaled_parameters_for_analysis(p, analysis_scale)
        actual = ground_truth_by_image_id[image_id]
        new = road_aligned_method(road, obstacles, analysis_parameters, auxiliary_masks)
        new['original_shape'] = [height, width]
        new['analysis_shape'] = [int(road.shape[0]), int(road.shape[1])]
        new['input_analysis_scale'] = float(analysis_scale)
        if new['prediction'] >= 0:
            new_confusion[actual, new['prediction']] += 1
        else:
            new_insufficient_by_actual[actual] += 1
        record = {'image_id': image_id, 'file_name': image_info['file_name'], 'ground_truth': actual, 'ground_truth_name': STATUS_NAMES[actual], 'new': new}
        records.append(record)
        if args.save_overlays and number <= args.overlay_limit:
            draw_overlay(args.images_dir / image_info['file_name'], road, obstacles, new, overlay_dir / f"{image_id:05d}_{Path(image_info['file_name']).stem}_road_aligned.jpg")
        if number % 25 == 0 or number == len(images):
            print(f'Progress: {number}/{len(images)}', flush=True)
    new_valid = int(new_confusion.sum())
    excluded_ground_truth_counts: dict[str, int] = {}
    for item in excluded_ground_truth:
        reason = str(item['reason'])
        excluded_ground_truth_counts[reason] = excluded_ground_truth_counts.get(reason, 0) + 1
    summary = {'dataset': str(args.coco.resolve()), 'mask_source': str(args.predictions.resolve()) if args.predictions else 'COCO ground-truth segmentation', 'images': len(records), 'input_images_considered': int(considered_image_count), 'eligible_images_before_limit': int(eligible_image_count_before_limit), 'excluded_ground_truth_images': int(len(excluded_ground_truth)), 'excluded_ground_truth_counts': excluded_ground_truth_counts, 'excluded_ground_truth_records': excluded_ground_truth, 'category_ids': {'road': road_id, 'obstacles': sorted(obstacle_ids), 'auxiliary_direction': {family: sorted(ids) for family, ids in auxiliary_category_ids.items()}}, 'obstacle_category_names': requested_obstacles, 'analysis_long_edge_limit': int(args.analysis_long_edge), 'parameters': asdict(p), 'new_confusion_matrix_excluding_insufficient': new_confusion.tolist(), 'new_confusion_table_with_insufficient': [new_confusion[index].tolist() + [int(new_insufficient_by_actual[index])] for index in range(3)], 'new_metrics_excluding_insufficient': classification_metrics(new_confusion), 'new_coverage': new_valid / len(records) if records else 0.0, 'new_insufficient_images': len(records) - new_valid, 'failure_reasons': {}, 'quality_warnings': {}, 'decision_confidence_counts': {}, 'corridor_source_band_counts': {source: 0 for source in SOURCE_WEIGHTS}, 'reconstruction_core_state_counts': {'direct': 0, 'internal_reconstruction': 0, 'terminal_extrapolation': 0}, 'reconstruction_evidence_counts': {}, 'auxiliary_direction_status_counts': {}, 'auxiliary_direction_source_counts': {}, 'v13_reconstruction_status_counts': {}, 'v13_reconstruction_totals': {'center_anchor_bands': 0, 'center_fit_observation_bands': 0, 'boundary_anchor_observations': 0, 'paired_boundary_anchor_observations': 0, 'single_boundary_hint_observations': 0, 'long_single_boundary_anchor_bands': 0, 'long_single_boundary_components': 0, 'roadline_internal_bands': 0, 'roadline_support_pixels': 0, 'roadline_internal_pixels': 0, 'terminal_extension_count': 0, 'terminal_extension_bands': 0}, 'geometry_validation': {}}
    for record in records:
        reason = record['new']['failure_reason']
        if reason:
            summary['failure_reasons'][reason] = summary['failure_reasons'].get(reason, 0) + 1
        for warning in record['new'].get('quality_warnings', []):
            summary['quality_warnings'][warning] = summary['quality_warnings'].get(warning, 0) + 1
        confidence = record['new'].get('decision_confidence', 'insufficient')
        summary['decision_confidence_counts'][confidence] = summary['decision_confidence_counts'].get(confidence, 0) + 1
        for source, count in record['new'].get('corridor_source_counts', {}).items():
            summary['corridor_source_band_counts'][source] = summary['corridor_source_band_counts'].get(source, 0) + int(count)
        for state, count in record['new'].get('reconstruction_core_state_counts', {}).items():
            summary['reconstruction_core_state_counts'][state] = summary['reconstruction_core_state_counts'].get(state, 0) + int(count)
        for evidence, count in record['new'].get('reconstruction_evidence_counts', {}).items():
            summary['reconstruction_evidence_counts'][evidence] = summary['reconstruction_evidence_counts'].get(evidence, 0) + int(count)
        auxiliary_status = record['new'].get('auxiliary_direction_status', 'not_evaluated')
        summary['auxiliary_direction_status_counts'][auxiliary_status] = summary['auxiliary_direction_status_counts'].get(auxiliary_status, 0) + 1
        for source, count in record['new'].get('auxiliary_direction_source_counts', {}).items():
            summary['auxiliary_direction_source_counts'][source] = summary['auxiliary_direction_source_counts'].get(source, 0) + int(count)
        reconstruction_status = record['new'].get('v13_reconstruction_status', 'not_applied')
        summary['v13_reconstruction_status_counts'][reconstruction_status] = summary['v13_reconstruction_status_counts'].get(reconstruction_status, 0) + 1
        reconstruction_summary = record['new'].get('v13_reconstruction_summary', {})
        for key in summary['v13_reconstruction_totals']:
            summary['v13_reconstruction_totals'][key] += int(reconstruction_summary.get(key, 0))
    roadline_distances = [float(record['new']['centerline_roadline_distance_median_px']) for record in records if record['new'].get('centerline_roadline_distance_median_px') is not None]
    continuity_scores = [float(record['new']['boundary_continuity_score']) for record in records if record['new'].get('boundary_continuity_score') is not None]
    recovered_ratios = [float(record['new']['internal_gap_recovered_ratio_of_corridor']) for record in records if record['new'].get('internal_gap_recovered_ratio_of_corridor') is not None]
    fragment_gaps = [gap for record in records for gap in record['new'].get('accepted_occlusion_gaps', []) if gap.get('connection_basis') == 'geometry_fragment_pair']
    summary['geometry_validation'] = {'roadline_evaluated_images': int(len(roadline_distances)), 'centerline_roadline_distance_median_px': float(np.median(roadline_distances)) if roadline_distances else None, 'boundary_continuity_mean': float(np.mean(continuity_scores)) if continuity_scores else None, 'internal_gap_recovered_pixels': int(sum((int(record['new'].get('internal_gap_recovered_pixels', 0)) for record in records))), 'internal_gap_recovered_ratio_mean': float(np.mean(recovered_ratios)) if recovered_ratios else None, 'geometry_fragment_pair_gap_count': int(len(fragment_gaps)), 'hermite_fragment_pair_gap_count': int(sum((gap.get('curve_model') == 'cubic_hermite_boundary_pair' for gap in fragment_gaps)))}
    (args.output / 'per_image_results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'confusion_table.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ground_truth', 'pred_small_or_clear', 'pred_half', 'pred_full', 'pred_insufficient', 'total'])
        for actual in range(3):
            values = [int(value) for value in new_confusion[actual]]
            insufficient = int(new_insufficient_by_actual[actual])
            writer.writerow([STATUS_NAMES[actual], *values, insufficient, sum(values) + insufficient])
    csv_fields = ['image_id', 'file_name', 'ground_truth', 'original_shape', 'analysis_shape', 'input_analysis_scale', 'v40_prediction', 'centerline_status', 'failure_reason', 'road_area_ratio', 'main_component_ratio', 'centerline_method', 'quality_warnings', 'centerline_scale', 'row_missing_band_ratio', 'row_observed_bands', 'row_span_bands', 'row_median_width_px', 'row_interpolation_ratio', 'row_extrapolation_ratio', 'reconstruction_pipeline', 'reconstruction_core_state_counts', 'reconstruction_evidence_counts', 'accepted_occlusion_gap_count', 'one_sided_extrapolation_count', 'discarded_terminal_fragment_bands', 'curve_fit_method', 'corridor_observation_mode', 'axis_selection_reason', 'direction_geometry_mode', 'cross_section_method', 'navigation_mode', 'central_roadline_used', 'central_roadline_name', 'central_roadline_coverage', 'roadline_native_pair_success_ratio', 'roadline_inside_sections_ratio', 'cross_section_crossing_count', 'cross_section_monotonic_ratio', 'cross_section_width_smoothness', 'roadline_native_fallback_to_v38', 'lateral_reconstruction_bands', 'terminal_reconstruction_bands', 'frozen_navigation_preserved', 'selected_direction_angle_deg', 'selected_track_id', 'track_score_margin', 'track_hypothesis_count', 'local_direction_refined_sections', 'auxiliary_direction_status', 'auxiliary_direction_mode', 'auxiliary_direction_evidence_count', 'auxiliary_direction_source_counts', 'auxiliary_direction_segment_count', 'auxiliary_direction_refined_sections', 'auxiliary_direction_rejected_sections', 'auxiliary_terminal_extension_used', 'auxiliary_direction_accepted_curve_count', 'auxiliary_direction_trusted_core_sections', 'auxiliary_direction_transition_sections', 'v13_reconstruction_status', 'v13_center_anchor_bands', 'v13_center_fit_observation_bands', 'v13_boundary_anchor_observations', 'v13_paired_boundary_anchor_observations', 'v13_single_boundary_hint_observations', 'v13_long_single_boundary_anchor_bands', 'v13_long_single_boundary_components', 'v13_roadline_internal_bands', 'v13_roadline_support_pixels', 'v13_roadline_internal_pixels', 'v13_terminal_extension_count', 'v13_terminal_extension_bands', 'v19_auxiliary_road_envelope_area_ratio', 'v19_auxiliary_envelope_added_pixels', 'v19_auxiliary_envelope_blank_pixels', 'v19_auxiliary_envelope_obstacle_pixels', 'v19_auxiliary_envelope_modified_stations', 'centerline_roadline_distance_median_px', 'centerline_roadline_distance_p90_px', 'centerline_roadline_sample_count', 'boundary_turn_p95_deg', 'boundary_continuity_score', 'internal_gap_recovered_pixels', 'internal_gap_recovered_ratio_of_corridor', 'reconstructed_corridor_area_ratio', 'full_road_corridor_area_ratio', 'analysis_geometry_corridor_area_ratio', 'inferred_occlusion_area_ratio', 'inferred_internal_gap_area_ratio', 'visible_road_in_full_ratio', 'valid_obstacle_pixel_ratio', 'raw_centerline_length_px', 'fitted_centerline_length_px', 'tangent_window_px', 'valid_section_ratio', 'p90_blockage', 'max_blockage', 'min_free_width_ratio', 'longest_partial_run', 'longest_full_run', 'decision_confidence', 'decision_confidence_score', 'decision_confidence_basis', 'decision_section_source_counts', 'free_lane_corridor_area_ratio', 'free_lane_continuity_ratio', 'free_lane_segment_count', 'interior_occlusion_inference_count', 'one_side_inference_count']
    with (args.output / 'per_image_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            new = record['new']
            writer.writerow({'image_id': record['image_id'], 'file_name': record['file_name'], 'ground_truth': record['ground_truth_name'], 'original_shape': json.dumps(new.get('original_shape')), 'analysis_shape': json.dumps(new.get('analysis_shape')), 'input_analysis_scale': safe_number(new.get('input_analysis_scale', 1.0)), 'v40_prediction': STATUS_NAMES[new['prediction']], 'centerline_status': new['centerline_status'], 'failure_reason': new['failure_reason'], 'road_area_ratio': safe_number(new['road_area_ratio']), 'main_component_ratio': safe_number(new['main_component_ratio']), 'centerline_method': new.get('centerline_method'), 'quality_warnings': ';'.join(new.get('quality_warnings', [])), 'centerline_scale': safe_number(new.get('centerline_scale')), 'row_missing_band_ratio': safe_number(new.get('row_missing_band_ratio')), 'row_observed_bands': new.get('row_observed_bands'), 'row_span_bands': new.get('row_span_bands'), 'row_median_width_px': safe_number(new.get('row_median_width_px')), 'row_interpolation_ratio': safe_number(new.get('row_interpolation_ratio')), 'row_extrapolation_ratio': safe_number(new.get('row_extrapolation_ratio')), 'reconstruction_pipeline': new.get('reconstruction_pipeline', 'unified_three_state'), 'reconstruction_core_state_counts': json.dumps(new.get('reconstruction_core_state_counts', {}), ensure_ascii=False), 'reconstruction_evidence_counts': json.dumps(new.get('reconstruction_evidence_counts', {}), ensure_ascii=False), 'accepted_occlusion_gap_count': len(new.get('accepted_occlusion_gaps', [])), 'one_sided_extrapolation_count': len(new.get('one_sided_extrapolations', [])), 'discarded_terminal_fragment_bands': new.get('discarded_terminal_fragment_bands', 0), 'curve_fit_method': new.get('curve_fit_method'), 'corridor_observation_mode': new.get('corridor_observation_mode'), 'axis_selection_reason': new.get('axis_selection_reason'), 'direction_geometry_mode': new.get('direction_geometry_mode'), 'cross_section_method': new.get('cross_section_method'), 'navigation_mode': new.get('navigation_mode'), 'central_roadline_used': new.get('central_roadline_used', False), 'central_roadline_name': new.get('central_roadline_name', 'none'), 'central_roadline_coverage': safe_number(new.get('central_roadline_coverage', 0.0)), 'roadline_native_pair_success_ratio': safe_number(new.get('roadline_native_pair_success_ratio', 0.0)), 'roadline_inside_sections_ratio': safe_number(new.get('roadline_inside_sections_ratio', 0.0)), 'cross_section_crossing_count': new.get('cross_section_crossing_count', 0), 'cross_section_monotonic_ratio': safe_number(new.get('cross_section_monotonic_ratio', 0.0)), 'cross_section_width_smoothness': safe_number(new.get('cross_section_width_smoothness', 0.0)), 'roadline_native_fallback_to_v38': new.get('roadline_native_fallback_to_v38', False), 'lateral_reconstruction_bands': new.get('lateral_reconstruction_bands', 0), 'terminal_reconstruction_bands': new.get('terminal_reconstruction_bands', 0), 'frozen_navigation_preserved': new.get('frozen_navigation_preserved', False), 'selected_direction_angle_deg': safe_number(new.get('selected_direction_angle_deg')), 'selected_track_id': new.get('selected_track_id'), 'track_score_margin': safe_number(new.get('track_score_margin')), 'track_hypothesis_count': len(new.get('track_hypotheses', [])), 'local_direction_refined_sections': new.get('local_direction_refined_sections', 0), 'auxiliary_direction_status': new.get('auxiliary_direction_status'), 'auxiliary_direction_mode': new.get('auxiliary_direction_mode'), 'auxiliary_direction_evidence_count': new.get('auxiliary_direction_evidence_count', 0), 'auxiliary_direction_source_counts': json.dumps(new.get('auxiliary_direction_source_counts', {}), ensure_ascii=False), 'auxiliary_direction_segment_count': len(new.get('auxiliary_direction_segments', [])), 'auxiliary_direction_refined_sections': new.get('auxiliary_direction_refined_sections', 0), 'auxiliary_direction_rejected_sections': new.get('auxiliary_direction_rejected_sections', 0), 'auxiliary_direction_accepted_curve_count': new.get('auxiliary_direction_accepted_curve_count', 0), 'auxiliary_direction_trusted_core_sections': new.get('auxiliary_direction_trusted_core_sections', 0), 'auxiliary_direction_transition_sections': new.get('auxiliary_direction_transition_sections', 0), 'auxiliary_terminal_extension_used': new.get('auxiliary_terminal_extension_used', False), 'v13_reconstruction_status': new.get('v13_reconstruction_status', 'not_applied'), 'v13_center_anchor_bands': new.get('v13_reconstruction_summary', {}).get('center_anchor_bands', 0), 'v13_center_fit_observation_bands': new.get('v13_reconstruction_summary', {}).get('center_fit_observation_bands', 0), 'v13_boundary_anchor_observations': new.get('v13_reconstruction_summary', {}).get('boundary_anchor_observations', 0), 'v13_paired_boundary_anchor_observations': new.get('v13_reconstruction_summary', {}).get('paired_boundary_anchor_observations', 0), 'v13_single_boundary_hint_observations': new.get('v13_reconstruction_summary', {}).get('single_boundary_hint_observations', 0), 'v13_long_single_boundary_anchor_bands': new.get('v13_reconstruction_summary', {}).get('long_single_boundary_anchor_bands', 0), 'v13_long_single_boundary_components': new.get('v13_reconstruction_summary', {}).get('long_single_boundary_components', 0), 'v13_roadline_internal_bands': new.get('v13_reconstruction_summary', {}).get('roadline_internal_bands', 0), 'v13_roadline_support_pixels': new.get('v13_reconstruction_summary', {}).get('roadline_support_pixels', 0), 'v13_roadline_internal_pixels': new.get('v13_reconstruction_summary', {}).get('roadline_internal_pixels', 0), 'v13_terminal_extension_count': new.get('v13_reconstruction_summary', {}).get('terminal_extension_count', 0), 'v13_terminal_extension_bands': new.get('v13_reconstruction_summary', {}).get('terminal_extension_bands', 0), 'v19_auxiliary_road_envelope_area_ratio': safe_number(new.get('auxiliary_road_envelope_area_ratio', 0.0)), 'v19_auxiliary_envelope_added_pixels': new.get('v13_reconstruction_summary', {}).get('v19_auxiliary_road_envelope', {}).get('added_pixels', 0), 'v19_auxiliary_envelope_blank_pixels': new.get('v13_reconstruction_summary', {}).get('v19_auxiliary_road_envelope', {}).get('blank_pixels', 0), 'v19_auxiliary_envelope_obstacle_pixels': new.get('v13_reconstruction_summary', {}).get('v19_auxiliary_road_envelope', {}).get('obstacle_pixels', 0), 'v19_auxiliary_envelope_modified_stations': new.get('v13_reconstruction_summary', {}).get('v19_auxiliary_envelope_geometry', {}).get('modified_station_count', 0), 'centerline_roadline_distance_median_px': safe_number(new.get('centerline_roadline_distance_median_px')), 'centerline_roadline_distance_p90_px': safe_number(new.get('centerline_roadline_distance_p90_px')), 'centerline_roadline_sample_count': new.get('centerline_roadline_sample_count', 0), 'boundary_turn_p95_deg': safe_number(new.get('boundary_turn_p95_deg')), 'boundary_continuity_score': safe_number(new.get('boundary_continuity_score')), 'internal_gap_recovered_pixels': new.get('internal_gap_recovered_pixels', 0), 'internal_gap_recovered_ratio_of_corridor': safe_number(new.get('internal_gap_recovered_ratio_of_corridor')), 'reconstructed_corridor_area_ratio': safe_number(new.get('reconstructed_corridor_area_ratio')), 'full_road_corridor_area_ratio': safe_number(new.get('full_road_corridor_area_ratio')), 'analysis_geometry_corridor_area_ratio': safe_number(new.get('analysis_geometry_corridor_area_ratio')), 'inferred_occlusion_area_ratio': safe_number(new.get('inferred_occlusion_area_ratio')), 'inferred_internal_gap_area_ratio': safe_number(new.get('inferred_internal_gap_area_ratio')), 'visible_road_in_full_ratio': safe_number(new.get('visible_road_in_full_ratio')), 'valid_obstacle_pixel_ratio': safe_number(new.get('valid_obstacle_pixel_ratio')), 'raw_centerline_length_px': safe_number(new.get('raw_centerline_length_px')), 'fitted_centerline_length_px': safe_number(new.get('fitted_centerline_length_px')), 'tangent_window_px': safe_number(new.get('tangent_window_px')), 'valid_section_ratio': safe_number(new['valid_section_ratio']), 'p90_blockage': safe_number(new['p90_blockage']), 'max_blockage': safe_number(new['max_blockage']), 'min_free_width_ratio': safe_number(new['min_free_width_ratio']), 'longest_partial_run': new['longest_partial_run'], 'longest_full_run': new['longest_full_run'], 'decision_confidence': new.get('decision_confidence'), 'decision_confidence_score': safe_number(new.get('decision_confidence_score')), 'decision_confidence_basis': new.get('decision_confidence_basis'), 'decision_section_source_counts': json.dumps(new.get('decision_section_source_counts', {}), ensure_ascii=False), 'free_lane_corridor_area_ratio': safe_number(new.get('free_lane_corridor_area_ratio')), 'free_lane_continuity_ratio': safe_number(new.get('free_lane_continuity_ratio')), 'free_lane_segment_count': new.get('free_lane_segment_count'), 'interior_occlusion_inference_count': len(new.get('interior_occlusion_inferences', [])), 'one_side_inference_count': len(new.get('one_side_inferences', []))})
    lines = ['V40：V38 道路本體＋Roadline center 專用橫斷面', f'影像數：{len(records)}', f"遮罩來源：{summary['mask_source']}", f'道路 category id：{road_id}', f'障礙物 category ids：{sorted(obstacle_ids)}', f'方向輔助 category ids：{json.dumps({family: sorted(ids) for family, ids in auxiliary_category_ids.items()}, ensure_ascii=False)}', f"分析長邊上限：{(args.analysis_long_edge if args.analysis_long_edge > 0 else '原始解析度')}", '', f"V40 Coverage：{summary['new_coverage']:.4f}", f"V40 insufficient：{summary['new_insufficient_images']}", f"V40 Accuracy（排除 insufficient）：{summary['new_metrics_excluding_insufficient']['accuracy']:.4f}" if new_valid else 'V40 沒有可評估影像', f"V40 Macro-F1（排除 insufficient）：{summary['new_metrics_excluding_insufficient']['macro_f1']:.4f}" if new_valid else '', f"中心線失敗原因：{json.dumps(summary['failure_reasons'], ensure_ascii=False)}", f"中心線品質警告：{json.dumps(summary['quality_warnings'], ensure_ascii=False)}", f"決策信心分布：{json.dumps(summary['decision_confidence_counts'], ensure_ascii=False)}", f"走廊來源帶數：{json.dumps(summary['corridor_source_band_counts'], ensure_ascii=False)}", f"三狀態道路重建：{json.dumps(summary['reconstruction_core_state_counts'], ensure_ascii=False)}", f"道路重建支持證據：{json.dumps(summary['reconstruction_evidence_counts'], ensure_ascii=False)}", f"方向輔助狀態：{json.dumps(summary['auxiliary_direction_status_counts'], ensure_ascii=False)}", f"方向輔助證據來源：{json.dumps(summary['auxiliary_direction_source_counts'], ensure_ascii=False)}", f"V13 道路重建狀態：{json.dumps(summary['v13_reconstruction_status_counts'], ensure_ascii=False)}", f"V13 道路重建統計：{json.dumps(summary['v13_reconstruction_totals'], ensure_ascii=False)}", '', '提醒：使用人工 COCO segmentation 時，結果只評估阻塞率演算法，不代表端到端模型表現。']
    (args.output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    print('\n'.join(lines))
    print(f'Output: {args.output}')
    if args.save_diagnostics:
        generate_diagnostics(args, coco, grouped, road_id, obstacle_ids, auxiliary_category_ids, records, p)

# 【命令列】定義資料路徑、輸出、診斷模式及所有可調演算法參數。
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco', type=Path, default=DEFAULT_COCO, help='NEWS COCO ground truth JSON')
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--predictions', type=Path, help='Optional COCO result JSON; omitted means GT masks')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--obstacle-categories', default='auto', help='逗號分隔的障礙類別；預設 auto 同時支援新版 rock/fallen vegetation/debris pile 與舊 NEWS Mud/TreeBranch')
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--limit', type=int, default=0, help='0 means all images')
    parser.add_argument('--image-ids', default='', help='Comma-separated COCO image IDs; overrides --limit')
    parser.add_argument('--analysis-long-edge', type=int, default=0, help='分析用 segmentation 的最長邊像素；0 保留原始解析度。高解析度資料可設 1600，所有 Road/障礙物/輔助標籤會同步最近鄰縮放，像素制幾何門檻也會按比例調整。')
    parser.add_argument('--save-overlays', action='store_true')
    parser.add_argument('--overlay-limit', type=int, default=30)
    parser.add_argument('--save-diagnostics', action='store_true', help='Create V40 diagnostics by default; use abde for report figures or d for the reconstruction panel')
    parser.add_argument('--diagnostics-only', action='store_true', help='Use existing per_image_results.json in --output; skip recomputing all images')
    parser.add_argument('--diagnostics-results', type=Path, help='Existing per_image_results.json used with --diagnostics-only; allows diagnostics to be written to a separate output folder')
    parser.add_argument('--diagnostic-limit', type=int, default=30)
    parser.add_argument('--diagnostic-layout', choices=('d', 'cdef', 'abde', 'bd', 'six'), default='cdef', help='d saves only V40 panel D; cdef is a compatibility alias for D/E/F; abde saves original/segmentation/D/E; bd saves segmentation beside D; six saves all V40 panels')
    parser.add_argument('--diagnostics-all', action='store_true', help='Do not auto-select; create diagnostics for every evaluated/requested image up to the limit')
    parser.add_argument('--sample-spacing', type=float, default=10.0)
    parser.add_argument('--row-band-height', type=int, default=6)
    parser.add_argument('--row-short-gap-bands', type=int, default=2, help='可直接視為 segmentation 小洞的最大連續缺帶數')
    parser.add_argument('--row-max-occlusion-gap-bands', type=int, default=60, help='有障礙物證據時，容許插值重建的最大缺帶數')
    parser.add_argument('--row-max-visible-continuation-gap-bands', type=int, default=120, help='前後人工 Road 幾何連續時，容許重新接回後段道路的最大缺帶數')
    parser.add_argument('--row-occlusion-min-fraction', type=float, default=0.2, help='長缺口中至少多少比例的帶必須有走廊內障礙物證據')
    parser.add_argument('--center-curve-smoothness', type=float, default=18.0)
    parser.add_argument('--width-curve-smoothness', type=float, default=24.0)
    parser.add_argument('--one-sided-max-bands', type=int, default=72, help='大型障礙只有單側道路資訊時，最多外推多少水平帶')
    parser.add_argument('--one-sided-min-obstacle-width-fraction', type=float, default=0.08, help='V23 末端外推帶內，障礙物至少占預測道路寬度的比例')
    parser.add_argument('--disable-interior-occlusion-inference', action='store_true', help='停用前後 Road 錨點的中段障礙遮擋邊界重建，供消融比較')
    parser.add_argument('--interior-occlusion-width-ratio', type=float, default=0.78, help='可見寬度低於前後插值寬度此比例時，才檢查中段遮擋')
    parser.add_argument('--interior-occlusion-obstacle-fraction', type=float, default=0.08, help='缺失側至少多少面積須由障礙物支持，才重建該側邊界')
    parser.add_argument('--interior-occlusion-anchor-window-bands', type=int, default=30, help='中段遮擋每一側搜尋完整 Road 錨點的最大帶數')
    parser.add_argument('--tangent-window-px', type=float, default=40.0, help='局部中心線切線 PCA 的前後視窗大小（像素）')
    parser.add_argument('--disable-continuous-direction-tracking', action='store_true', help='停用 v11 任意角度、多軌跡道路座標，改跑 v10 X/Y 消融路徑')
    parser.add_argument('--direction-angle-step', type=float, default=15.0, help='全域方向候選的角度間隔；局部切線仍為連續角度')
    parser.add_argument('--direction-top-k', type=int, default=5, help='縮圖預選後在原解析度重建的候選軌跡數')
    parser.add_argument('--direction-seeds-per-angle', type=int, default=2, help='每個角度最多保留幾條具獨立身份的平行 Road 軌跡')
    parser.add_argument('--direction-geometry-mode', choices=('global_angle_cross_sections', 'local_tangent_cross_sections'), default='global_angle_cross_sections', help='橫斷面方向模式；預設 Tier A 單一全域角度，local_tangent 僅供 v11 消融比較')
    parser.add_argument('--local-direction-refine-iterations', type=int, default=2, help='以局部切線動態旋轉橫斷帶並重測 direct 邊界的次數')
    parser.add_argument('--disable-auxiliary-direction-evidence', action='store_true', help='忽略 road line、cone、curb/guardrail/barricade/roadside wall；供新版 segmentation 的消融比較')
    parser.add_argument('--auxiliary-segment-stations', type=int, default=8, help='方向輔助模式中，每個共用角度長分段包含的橫斷面站數')
    parser.add_argument('--auxiliary-max-segment-delta', type=float, default=22.5, help='輔助方向相對 Road 全域方向最多可修正的角度（度）')
    parser.add_argument('--disable-v16-partial-auxiliary-spline', action='store_true', help='停用 V16 局部輔助曲線，回退到 V15 的長分段固定角度；供消融比較')
    parser.add_argument('--disable-unknown-internal-gaps', action='store_true', help='內部缺口仍要求障礙物支持，不啟用雙 Road 錨點未知遮擋重建')
    parser.add_argument('--disable-auxiliary-road-reconstruction', action='store_true', help='停用 V13 road line 道路支持、輔助邊界、中心錨定與輔助末端延伸，供 V12 幾何消融比較')
    parser.add_argument('--v13-roadline-center-fit-weight', type=float, default=3.0, help='已確認中央 road line 擬合道路中心時的觀測權重')
    parser.add_argument('--v13-roadline-center-blend', type=float, default=0.85, help='中央 road line 擬合結果寫回中心線的比例')
    parser.add_argument('--v13-single-boundary-min-span-bands', type=int, default=12, help='單側 curb/guardrail 等要約束邊界所需的最小縱向跨距')
    parser.add_argument('--v13-single-boundary-blend', type=float, default=0.6, help='長距離單側道路設施寫回該側邊界的比例')
    parser.add_argument('--horizontal-fallback-diagonal-ratio', type=float, default=0.25, help='原方向中心線短於影像對角線此比例時，嘗試轉置追蹤水平道路')
    parser.add_argument('--close-radius', type=int, default=2)
    parser.add_argument('--valid-ratio', type=float, default=0.6)
    parser.add_argument('--required-run', type=int, default=3)
    return parser.parse_args()

# 【程式入口】驗證輸入輸出路徑、建立 Parameters，然後啟動資料集評估。
def main():
    args = parse_args()
    args.output = args.output.resolve()
    if args.output == BASE or not args.output.is_relative_to(BASE):
        raise ValueError(f'輸出必須位於程式目錄 {BASE} 的子資料夾。')
    if not args.coco.is_file():
        raise ValueError(f'找不到 COCO JSON：{args.coco}')
    if (args.save_overlays or args.save_diagnostics or args.diagnostics_only) and (not args.images_dir.is_dir()):
        raise ValueError(f'找不到影像資料夾：{args.images_dir}')
    if args.diagnostics_only:
        args.save_diagnostics = True
    p = Parameters(sample_spacing=args.sample_spacing, row_band_height=args.row_band_height, row_short_gap_bands=args.row_short_gap_bands, row_max_occlusion_gap_bands=args.row_max_occlusion_gap_bands, row_max_visible_continuation_gap_bands=args.row_max_visible_continuation_gap_bands, row_occlusion_min_fraction=args.row_occlusion_min_fraction, center_curve_smoothness=args.center_curve_smoothness, width_curve_smoothness=args.width_curve_smoothness, one_sided_max_bands=args.one_sided_max_bands, one_sided_min_obstacle_width_fraction=args.one_sided_min_obstacle_width_fraction, enable_interior_occlusion_inference=not args.disable_interior_occlusion_inference, interior_occlusion_width_ratio=args.interior_occlusion_width_ratio, interior_occlusion_obstacle_fraction=args.interior_occlusion_obstacle_fraction, interior_occlusion_anchor_window_bands=args.interior_occlusion_anchor_window_bands, tangent_window_px=args.tangent_window_px, enable_continuous_direction_tracking=not args.disable_continuous_direction_tracking, direction_angle_step_degrees=args.direction_angle_step, direction_top_k=args.direction_top_k, direction_seed_tracks_per_angle=args.direction_seeds_per_angle, direction_geometry_mode=args.direction_geometry_mode, local_direction_refine_iterations=args.local_direction_refine_iterations, enable_auxiliary_direction_evidence=not args.disable_auxiliary_direction_evidence, auxiliary_segment_stations=args.auxiliary_segment_stations, auxiliary_max_segment_delta_deg=args.auxiliary_max_segment_delta, enable_v16_partial_auxiliary_spline=not args.disable_v16_partial_auxiliary_spline, include_unknown_internal_gaps=not args.disable_unknown_internal_gaps, enable_auxiliary_road_reconstruction=not args.disable_auxiliary_road_reconstruction, v13_roadline_center_fit_weight=args.v13_roadline_center_fit_weight, v13_roadline_center_blend=args.v13_roadline_center_blend, v13_single_boundary_min_span_bands=args.v13_single_boundary_min_span_bands, v13_single_boundary_blend=args.v13_single_boundary_blend, horizontal_fallback_diagonal_ratio=args.horizontal_fallback_diagonal_ratio, close_radius=args.close_radius, min_valid_slice_ratio=args.valid_ratio, required_consecutive_slices=args.required_run)
    evaluate(args, p)
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}')
        raise SystemExit(1)
