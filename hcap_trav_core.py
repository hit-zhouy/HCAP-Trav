#!/usr/bin/env python3
"""Core training components for HCAP-Trav.

This file keeps the method components used in the paper and omits dataset
construction, experiment diagnostics, checkpoint management, and ablations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from proto_traversability.models.dense_hierarchical_surface_pico import (
    DenseHierarchicalSurfacePiCO,
)
from proto_traversability.models.prototype_conditioned_response_head_direct_v2 import (
    DirectFeaturePrototypeResponseHead,
    masked_hybrid_response_loss,
)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def cross_entropy(target: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    return -(target * probability.clamp_min(1e-8).log()).sum(dim=-1)


def normalize_probability(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-8)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def route_confidence(route: torch.Tensor) -> torch.Tensor:
    entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1)
    return (1.0 - entropy / np.log(2.0)).clamp(0.0, 1.0)


def gather_map(value: torch.Tensor, batch, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(batch.cell_ids, dtype=torch.bool)
    return value[
        batch.batch_index[mask],
        :,
        batch.rows[mask],
        batch.cols[mask],
    ]


def forward_model(model: DenseHierarchicalSurfacePiCO, batch):
    return model(
        batch.image,
        batch.points,
        batch.point_mask,
        batch.quality,
        batch.availability,
        batch.batch_index,
        batch.rows,
        batch.cols,
        batch.batch_size,
        batch.height,
        batch.width,
        traversed_evidence_mask=batch.traversed_evidence_mask,
    )


@torch.no_grad()
def balanced_sinkhorn_assignment(
    logits: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 30,
    balance_strength: float = 0.25,
) -> torch.Tensor:
    n, k = logits.shape
    if n == 0 or k == 0:
        return logits.new_zeros((n, k))

    scores = logits.float() / max(float(epsilon), 1e-4)
    scores = scores - scores.max(dim=1, keepdim=True).values
    q = torch.exp(scores).T
    q = q / q.sum().clamp_min(1e-12)

    empirical = torch.softmax(logits.float(), dim=-1).mean(dim=0)
    empirical = empirical / empirical.sum().clamp_min(1e-12)
    uniform = empirical.new_full((k,), 1.0 / k)
    row_target = balance_strength * uniform + (1.0 - balance_strength) * empirical
    row_target = row_target / row_target.sum().clamp_min(1e-12)
    column_target = 1.0 / n

    for _ in range(max(int(iterations), 1)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q * row_target[:, None]
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q * column_target

    q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return q.T.contiguous().to(logits)


def branch_weights(batch, cell_weight: torch.Tensor | None = None):
    if cell_weight is None:
        cell_weight = torch.ones_like(batch.observed, dtype=batch.quality.dtype)
    cell_weight = cell_weight.detach().to(batch.quality)
    return {
        "fused": batch.observed.to(batch.quality)
        * batch.quality[:, 5].clamp(0.05, 1.0)
        * cell_weight,
        "image": batch.camera_observed.to(batch.quality)
        * batch.quality[:, 2].clamp(0.0, 1.0)
        * cell_weight,
        "lidar": batch.lidar_observed.to(batch.quality)
        * batch.quality[:, 5].clamp(0.05, 1.0)
        * cell_weight,
    }


def coarse_group_loss(
    outputs,
    route: torch.Tensor,
    batch,
    sample_type: torch.Tensor,
    labels: dict,
    coefficients: dict,
    cell_weight: torch.Tensor,
) -> torch.Tensor:
    weights = branch_weights(batch, cell_weight)
    hard_t = sample_type == int(labels["traversed"])
    hard_r = sample_type == int(labels["negative"])
    unknown = sample_type == int(labels["unknown"])

    total = batch.quality.new_zeros(())
    scale = 0.0

    for name in ("fused", "image", "lidar"):
        coefficient = float(coefficients.get(name, 0.0))
        if coefficient <= 0:
            continue

        probability = normalize_probability(
            gather_map(outputs[name].group_probability, batch)
        )
        per_cell = cross_entropy(route.detach(), probability)
        branch_weight = weights[name]

        parts = []
        for selected in (hard_t, hard_r, unknown):
            valid = selected & (branch_weight > 0)
            if valid.any():
                parts.append(weighted_mean(per_cell[valid], branch_weight[valid]))

        if parts:
            total = total + coefficient * torch.stack(parts).mean()
            scale += coefficient

    return total / max(scale, 1e-12)


def prototype_group_loss(
    outputs,
    route: torch.Tensor,
    batch,
    sample_type: torch.Tensor,
    labels: dict,
    coefficients: dict,
    cell_weight: torch.Tensor,
) -> torch.Tensor:
    weights = branch_weights(batch, cell_weight)
    hard_t = sample_type == int(labels["traversed"])
    hard_r = sample_type == int(labels["negative"])

    total = batch.quality.new_zeros(())
    scale = 0.0

    for name in ("fused", "image", "lidar"):
        coefficient = float(coefficients.get(name, 0.0))
        if coefficient <= 0:
            continue

        probability = normalize_probability(
            gather_map(outputs[name].prototype_group_probability, batch)
        )
        per_cell = cross_entropy(route.detach(), probability)
        branch_weight = weights[name]
        parts = []

        for selected in (hard_t, hard_r):
            valid = selected & (branch_weight > 0)
            if valid.any():
                parts.append(weighted_mean(per_cell[valid], branch_weight[valid]))

        if parts:
            total = total + coefficient * torch.stack(parts).mean()
            scale += coefficient

    return total / max(scale, 1e-12)


def groupwise_swav_loss(outputs, route: torch.Tensor, batch, cfg: dict):
    zero = batch.quality.new_zeros(())
    direct = batch.camera_observed & batch.lidar_observed & batch.observed
    direct &= batch.quality[:, 2].float().ge(float(cfg.get("minimum_q_surface", 0.65)))
    direct &= batch.quality[:, 5].float().ge(float(cfg.get("minimum_q_observation", 0.20)))

    if not direct.any():
        return zero, zero

    fused = outputs["fused"]
    route = route.detach()
    minimum_route = float(cfg.get("minimum_route_confidence", 0.80))
    minimum_cells = int(cfg.get("minimum_cells_per_group", 32))
    temperature = float(cfg.get("temperature", 0.15))
    epsilon = float(cfg.get("epsilon", 0.05))
    iterations = int(cfg.get("iterations", 30))
    balance_strength = float(cfg.get("balance_strength", 0.25))

    n_t = int(fused.traversed_probability.shape[1])
    n_r = int(fused.negative_probability.shape[1])
    groups = ((0, 0, n_t), (1, n_t, n_t + n_r))

    fused_losses = []
    modal_losses = []

    for group, begin, end in groups:
        selected = direct & route[:, group].ge(minimum_route)
        if int(selected.sum()) < max(minimum_cells, end - begin):
            continue

        fused_similarity = gather_map(fused.similarities, batch, selected).float()[:, begin:end]
        assignment = balanced_sinkhorn_assignment(
            fused_similarity.detach(),
            epsilon=epsilon,
            iterations=iterations,
            balance_strength=balance_strength,
        )

        route_weight = route[selected, group].float()
        q_surface = batch.quality[selected, 2].float().clamp(0.05, 1.0)
        q_observation = batch.quality[selected, 5].float().clamp(0.05, 1.0)

        fused_probability = torch.softmax(fused_similarity / temperature, dim=-1)
        fused_losses.append(
            weighted_mean(
                cross_entropy(assignment, fused_probability),
                route_weight * q_surface * q_observation,
            )
        )

        for name, quality in (("image", q_surface), ("lidar", q_observation)):
            similarity = gather_map(outputs[name].similarities, batch, selected).float()[:, begin:end]
            probability = torch.softmax(similarity / temperature, dim=-1)
            modal_losses.append(
                weighted_mean(
                    cross_entropy(assignment, probability),
                    route_weight * quality,
                )
            )

    fused_loss = torch.stack(fused_losses).mean() if fused_losses else zero
    modal_loss = torch.stack(modal_losses).mean() if modal_losses else zero
    return fused_loss, modal_loss


def pico_group_probability(query: torch.Tensor, model, temperature: float) -> torch.Tensor:
    query = F.normalize(query.float(), dim=-1, eps=1e-6)
    centers = F.normalize(model.group_prototypes.detach().to(query).float(), dim=-1, eps=1e-6)
    return torch.softmax(query @ centers.T / max(float(temperature), 1e-4), dim=-1)


@torch.no_grad()
def update_group_state(
    route: torch.Tensor,
    prediction,
    batch,
    sample_type: torch.Tensor,
    labels: dict,
    model,
    cfg: dict,
    allow_unknown_update: bool,
) -> torch.Tensor:
    classifier = normalize_probability(
        gather_map(prediction.group_probability, batch).float()
    )
    query = F.normalize(gather_map(prediction.query, batch).float(), dim=-1, eps=1e-6)

    hard_t = sample_type == int(labels["traversed"])
    hard_r = sample_type == int(labels["negative"])
    unknown = sample_type == int(labels["unknown"])

    candidate = classifier.new_ones(classifier.shape)
    candidate[hard_t] = candidate.new_tensor([1.0, 0.0])
    candidate[hard_r] = candidate.new_tensor([0.0, 1.0])

    candidate_classifier = normalize_probability(classifier * candidate)
    classifier_confidence, pseudo_label = candidate_classifier.max(dim=-1)

    update_mask = batch.observed & (hard_t | hard_r)
    if allow_unknown_update:
        update_mask |= (
            batch.observed
            & unknown
            & classifier_confidence.ge(float(cfg.get("classifier_confidence", 0.70)))
        )

    update_mask &= batch.camera_observed & batch.lidar_observed
    update_mask &= batch.quality[:, 2].float().ge(float(cfg.get("minimum_q_surface", 0.65)))
    update_mask &= batch.quality[:, 5].float().ge(float(cfg.get("minimum_q_observation", 0.20)))

    momentum = float(cfg.get("prototype_momentum", 0.99))
    minimum_cells = int(cfg.get("minimum_cells", 8))

    for group in (0, 1):
        selected = update_mask & pseudo_label.eq(group)
        if int(selected.sum()) < minimum_cells:
            continue

        weight = (
            batch.quality[selected, 2].float().clamp(0.05, 1.0)
            * batch.quality[selected, 5].float().clamp(0.05, 1.0)
        )
        centroid = F.normalize(
            (query[selected] * weight[:, None]).sum(dim=0), dim=0, eps=1e-6
        )
        old = F.normalize(model.group_prototypes[group].float(), dim=0, eps=1e-6)
        model.group_prototypes[group].copy_(
            F.normalize(momentum * old + (1.0 - momentum) * centroid, dim=0, eps=1e-6)
        )

    proto_probability = pico_group_probability(
        query, model, float(cfg.get("temperature", 0.10))
    )
    proto_probability = normalize_probability(proto_probability * candidate)
    proto_confidence = proto_probability.max(dim=-1).values

    updated = route.clone()
    selected_unknown = (
        allow_unknown_update
        & unknown
        & batch.observed
        & proto_confidence.ge(float(cfg.get("route_confidence", 0.55)))
    )

    if selected_unknown.any():
        route_momentum = float(cfg.get("route_momentum", 0.95))
        updated[selected_unknown] = normalize_probability(
            route_momentum * route[selected_unknown]
            + (1.0 - route_momentum) * proto_probability[selected_unknown]
        )

    updated[hard_t] = updated.new_tensor([1.0, 0.0])
    updated[hard_r] = updated.new_tensor([0.0, 1.0])
    return updated


def _gradient_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    detached = value.detach()
    return detached + float(scale) * (value - detached)


def _pool_mean(
    value: torch.Tensor,
    inverse: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    out = value.new_zeros((groups, value.shape[1]))
    out.index_add_(0, inverse, value * weight[:, None])
    denom = value.new_zeros(groups)
    denom.index_add_(0, inverse, weight)
    return out / denom[:, None].clamp_min(1e-6)


def backbone_gradient_scale(epoch: int, start: int = 2, ramp: int = 3, maximum: float = 0.02) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return maximum
    ratio = min(max((epoch - start + 1) / float(ramp), 0.0), 1.0)
    return maximum * ratio


def build_response_events(
    model,
    fused,
    batch,
    targets,
    normalization: dict,
    cfg: dict,
    gradient_scale: float,
):
    condition, target, target_valid, confidence, exact, event_id = targets.batch(batch)

    selected = exact & batch.observed & batch.traversed
    selected &= confidence.ge(float(cfg.get("minimum_target_confidence", 0.50)))
    selected &= batch.camera_observed & batch.lidar_observed

    q_surface = batch.quality[:, 2].float().clamp(0.0, 1.0)
    q_observation = batch.quality[:, 5].float().clamp(0.0, 1.0)
    selected &= q_surface.ge(float(cfg.get("minimum_q_surface", 0.60)))
    selected &= q_observation.ge(float(cfg.get("minimum_q_observation", 0.20)))

    if not selected.any():
        return None

    index = torch.nonzero(selected, as_tuple=False).squeeze(1)
    unique_event, inverse = torch.unique(event_id[index].long(), sorted=True, return_inverse=True)
    groups = int(len(unique_event))

    alpha_cell = gather_map(fused.traversed_probability, batch)[index].float()
    query_cell = gather_map(fused.query, batch)[index].float()
    weight = confidence[index].float() * q_surface[index].clamp_min(0.05) * q_observation[index].clamp_min(0.05)

    alpha = normalize_probability(_pool_mean(alpha_cell, inverse, weight, groups))
    query = F.normalize(_pool_mean(query_cell, inverse, weight, groups), dim=-1, eps=1e-6)

    centers = model.normalized_prototypes[: model.traversed_prototypes]
    distance = 1.0 - query @ centers.T
    residual = query - alpha @ centers

    first = torch.stack([
        torch.nonzero(inverse.eq(i), as_tuple=False)[0, 0]
        for i in range(groups)
    ])
    first = index[first]

    condition = condition[first].float()
    target = target[first].float()
    target_valid = target_valid[first].bool()
    confidence = confidence[first].float().clamp(0.0, 1.0)

    condition = (
        condition - normalization["condition_center"]
    ) / normalization["condition_scale"].clamp_min(1e-6)
    condition = condition.clamp(-5.0, 5.0)

    target_normalized = (
        target - normalization["target_center"]
    ) / normalization["target_scale"].clamp_min(1e-6)
    target_normalized = torch.where(
        target_valid, target_normalized, torch.zeros_like(target_normalized)
    )

    return {
        "fused_feature": _gradient_scale(query, gradient_scale),
        "prototype_probability": _gradient_scale(alpha, gradient_scale),
        "prototype_distance": _gradient_scale(distance, gradient_scale),
        "prototype_residual": _gradient_scale(residual, gradient_scale),
        "condition": condition,
        "target": target,
        "target_normalized": target_normalized,
        "target_valid": target_valid,
        "confidence": confidence,
    }


def response_prediction_loss(
    model,
    head: DirectFeaturePrototypeResponseHead,
    fused,
    batch,
    targets,
    normalization: dict,
    cfg: dict,
    epoch: int,
):
    scale = backbone_gradient_scale(
        epoch,
        start=int(cfg.get("gradient_start_epoch", 2)),
        ramp=int(cfg.get("gradient_ramp_epochs", 3)),
        maximum=float(cfg.get("maximum_gradient_scale", 0.02)),
    )
    event = build_response_events(
        model, fused, batch, targets, normalization, cfg, scale
    )
    if event is None:
        return fused.query.sum() * 0.0, None

    prediction = head(
        event["fused_feature"],
        event["prototype_probability"],
        event["prototype_distance"],
        event["prototype_residual"],
        event["condition"],
    )
    loss, _ = masked_hybrid_response_loss(
        prediction,
        event["target_normalized"],
        event["target_valid"],
        sample_weight=event["confidence"],
        nll_weight=float(cfg.get("nll_weight", 1.0)),
        mean_huber_weight=float(cfg.get("huber_weight", 0.5)),
        huber_delta=float(cfg.get("huber_delta", 1.0)),
    )
    return loss, prediction


def response_to_cost(
    response_mean: torch.Tensor,
    response_center: torch.Tensor,
    response_scale: torch.Tensor,
) -> torch.Tensor:
    z = (response_mean - response_center) / response_scale.clamp_min(1e-6)
    component_cost = torch.clamp(z / 2.0, 0.0, 1.0)
    return component_cost.mean(dim=-1)


def compute_core_losses(
    model,
    response_head,
    outputs,
    batch,
    route: torch.Tensor,
    sample_type: torch.Tensor,
    labels: dict,
    cfg: dict,
    epoch: int,
    response_targets=None,
    response_normalization=None,
):
    hard_anchor = (
        (sample_type == int(labels["traversed"]))
        | (sample_type == int(labels["negative"]))
    )
    confidence = route_confidence(route)
    cell_weight = torch.where(hard_anchor, torch.ones_like(confidence), confidence)

    warmup_epochs = int(cfg.get("warmup_epochs", 3))
    if epoch < warmup_epochs:
        cell_weight = hard_anchor.to(confidence)

    group = coarse_group_loss(
        outputs,
        route,
        batch,
        sample_type,
        labels,
        cfg.get("group_branch_weights", {"fused": 1.0, "image": 0.5, "lidar": 0.5}),
        cell_weight,
    )

    zero = group.new_zeros(())
    if epoch < warmup_epochs:
        proto_group = zero
        swav_cluster = zero
        swav_modal = zero
    else:
        proto_group = prototype_group_loss(
            outputs,
            route,
            batch,
            sample_type,
            labels,
            cfg.get("group_branch_weights", {"fused": 1.0, "image": 0.5, "lidar": 0.5}),
            cell_weight,
        )
        swav_cluster, swav_modal = groupwise_swav_loss(
            outputs, route, batch, cfg.get("prototype_learning", {})
        )

    response = zero
    if response_head is not None and response_targets is not None:
        response, _ = response_prediction_loss(
            model,
            response_head,
            outputs["fused"],
            batch,
            response_targets,
            response_normalization,
            cfg.get("response", {}),
            epoch,
        )

    weights = cfg.get("loss_weights", {})
    total = (
        float(weights.get("group", 1.0)) * group
        + float(weights.get("prototype_group", 1.0)) * proto_group
        + float(weights.get("swav_cluster", 1.0)) * swav_cluster
        + float(weights.get("swav_modal", 1.0)) * swav_modal
        + float(weights.get("response", 1.0)) * response
    )

    return total, {
        "group": group,
        "prototype_group": proto_group,
        "swav_cluster": swav_cluster,
        "swav_modal": swav_modal,
        "response": response,
    }


def training_step(
    model,
    response_head,
    optimizer,
    batch,
    route: torch.Tensor,
    sample_type: torch.Tensor,
    labels: dict,
    cfg: dict,
    epoch: int,
    response_targets=None,
    response_normalization=None,
):
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_model(model, batch)

    total, losses = compute_core_losses(
        model,
        response_head,
        outputs,
        batch,
        route,
        sample_type,
        labels,
        cfg,
        epoch,
        response_targets=response_targets,
        response_normalization=response_normalization,
    )

    total.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("gradient_clip_norm", 5.0))
    )
    if response_head is not None:
        torch.nn.utils.clip_grad_norm_(
            response_head.parameters(), float(cfg.get("response_gradient_clip_norm", 5.0))
        )
    optimizer.step()

    with torch.no_grad():
        prediction = forward_model(model, batch)["fused"]
        updated_route = update_group_state(
            route,
            prediction,
            batch,
            sample_type,
            labels,
            model,
            cfg.get("partial_label", {}),
            allow_unknown_update=epoch >= int(cfg.get("pseudo_start_epoch", 3)),
        )

    return updated_route, {name: float(value.detach()) for name, value in losses.items()}
