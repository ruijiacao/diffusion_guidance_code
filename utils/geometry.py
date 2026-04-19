import jax.numpy as jnp
import numpy as np


def create_curve_bounded_region(x_left, x_right, lower_curve_func, upper_curve_func):
    """Create a region bounded by two curves y=lower(x) and y=upper(x)."""

    def is_inside_vectorized(points):
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        in_x_bounds = (x_vals >= x_left) & (x_vals <= x_right)
        y_lower_vals = lower_curve_func(x_vals)
        y_upper_vals = upper_curve_func(x_vals)
        in_y_bounds = (y_vals >= y_lower_vals) & (y_vals <= y_upper_vals)
        return in_x_bounds & in_y_bounds

    def y_lower(x):
        return lower_curve_func(x)

    def y_upper(x):
        return upper_curve_func(x)

    x_range = [x_left, x_right]
    return is_inside_vectorized, y_lower, y_upper, x_range


def create_left_crescent_constraints(center, radius_outer, radius_inner, offset_x):
    """Create a left-facing crescent (opens to the right)."""
    cx, cy = center[0], center[1]
    inner_cx = cx - offset_x
    inner_cy = cy

    def is_inside_vectorized(points):
        dist_outer_sq = jnp.sum((points - jnp.array([cx, cy])) ** 2, axis=1)
        in_outer = dist_outer_sq <= radius_outer**2
        dist_inner_sq = jnp.sum((points - jnp.array([inner_cx, inner_cy])) ** 2, axis=1)
        outside_inner = dist_inner_sq > radius_inner**2
        return in_outer & outside_inner

    def y_lower(x):
        x_rel_outer = x - cx
        under_sqrt_outer = jnp.maximum(radius_outer**2 - x_rel_outer**2, 0.0)
        return cy - jnp.sqrt(under_sqrt_outer)

    def y_upper(x):
        x_rel_outer = x - cx
        under_sqrt_outer = jnp.maximum(radius_outer**2 - x_rel_outer**2, 0.0)
        return cy + jnp.sqrt(under_sqrt_outer)

    x_range = [cx - radius_outer, cx + radius_outer]
    return is_inside_vectorized, y_lower, y_upper, x_range


def create_right_crescent_constraints(center, radius_outer, radius_inner, offset_x):
    """Create a right-facing crescent (opens to the left)."""
    cx, cy = center[0], center[1]
    inner_cx = cx + offset_x
    inner_cy = cy

    def is_inside_vectorized(points):
        dist_outer_sq = jnp.sum((points - jnp.array([cx, cy])) ** 2, axis=1)
        in_outer = dist_outer_sq <= radius_outer**2
        dist_inner_sq = jnp.sum((points - jnp.array([inner_cx, inner_cy])) ** 2, axis=1)
        outside_inner = dist_inner_sq > radius_inner**2
        return in_outer & outside_inner

    def y_lower(x):
        x_rel_outer = x - cx
        under_sqrt_outer = jnp.maximum(radius_outer**2 - x_rel_outer**2, 0.0)
        return cy - jnp.sqrt(under_sqrt_outer)

    def y_upper(x):
        x_rel_outer = x - cx
        under_sqrt_outer = jnp.maximum(radius_outer**2 - x_rel_outer**2, 0.0)
        return cy + jnp.sqrt(under_sqrt_outer)

    x_range = [cx - radius_outer, cx + radius_outer]
    return is_inside_vectorized, y_lower, y_upper, x_range


def create_circle_constraints(center, radius):
    """Create region helper functions for a disk."""
    cx, cy = center[0], center[1]

    def is_inside_vectorized(points):
        return jnp.sum((points - jnp.array(center)) ** 2, axis=1) <= radius**2

    def y_lower(x):
        delta_x = x - cx
        under_sqrt = jnp.maximum(radius**2 - delta_x**2, 0.0)
        return cy - jnp.sqrt(under_sqrt)

    def y_upper(x):
        delta_x = x - cx
        under_sqrt = jnp.maximum(radius**2 - delta_x**2, 0.0)
        return cy + jnp.sqrt(under_sqrt)

    x_range = [cx - radius, cx + radius]
    return is_inside_vectorized, y_lower, y_upper, x_range


def distance_to_disk(point, center, radius):
    """Distance from point to disk boundary (0 if inside)."""
    dist_to_center = jnp.linalg.norm(point - center)
    return jnp.maximum(0, dist_to_center - radius)


def distances_to_target_disk(endpoints, center, radius):
    """Compute target-disk distances for an endpoint array of shape (n, 2)."""
    dists = [float(distance_to_disk(jnp.array(p), center, radius)) for p in np.asarray(endpoints)]
    return np.array(dists)
