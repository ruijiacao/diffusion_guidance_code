import jax
import jax.numpy as jnp


def center_of_mass_fixed_grid(cur_pos, y_lower, y_upper, x_range, t, n_x=20, n_y=20, region=None):
    """Compute center of mass on a fixed grid, optionally over an arbitrary region mask."""
    lambda_t = jnp.exp(-t)
    sigma2_t = 1 - lambda_t**2

    if region is not None:
        x_min, x_max, y_min, y_max = region["bbox"]
        x_vals = jnp.linspace(x_min, x_max, n_x)
        y_vals = jnp.linspace(y_min, y_max, n_y)
        xx, yy = jnp.meshgrid(x_vals, y_vals, indexing="xy")
        points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

        diffs = points - cur_pos
        exponent = -0.5 * jnp.sum(diffs**2, axis=1) / sigma2_t
        phi_vals = jnp.exp(exponent) / (2 * jnp.pi * sigma2_t)
        mask = region["constraint"](points).astype(jnp.float64)
        weighted_phi = phi_vals * mask

        dx = (x_max - x_min) / jnp.maximum(n_x - 1, 1)
        dy = (y_max - y_min) / jnp.maximum(n_y - 1, 1)
        cell_area = dx * dy

        integral_phi = jnp.sum(weighted_phi) * cell_area
        integral_phi = jnp.where(integral_phi < 1e-300, 1.0, integral_phi)
        center_x = jnp.sum(points[:, 0] * weighted_phi) * cell_area / integral_phi
        center_y = jnp.sum(points[:, 1] * weighted_phi) * cell_area / integral_phi
        return jnp.array([center_x, center_y])

    x_min, x_max = x_range
    x_vals = jnp.linspace(x_min, x_max, n_x)

    def integrate_over_y(x):
        y_min = y_lower(x)
        y_max = y_upper(x)
        y_vals = jnp.linspace(y_min, y_max, n_y)

        def compute_phi_xyz(y):
            point = jnp.array([x, y])
            diff = point - cur_pos
            exponent = -0.5 * jnp.sum(diff**2) / sigma2_t
            phi_val = jnp.exp(exponent) / (2 * jnp.pi * sigma2_t)
            return jnp.array([phi_val, x * phi_val, y * phi_val])

        phi_xyz_vals = jax.vmap(compute_phi_xyz)(y_vals)
        return jnp.array([
            jax.scipy.integrate.trapezoid(phi_xyz_vals[:, 0], y_vals),
            jax.scipy.integrate.trapezoid(phi_xyz_vals[:, 1], y_vals),
            jax.scipy.integrate.trapezoid(phi_xyz_vals[:, 2], y_vals),
        ])

    integrals_x = jax.vmap(integrate_over_y)(x_vals)
    integral_phi = jax.scipy.integrate.trapezoid(integrals_x[:, 0], x_vals)
    integral_x_phi = jax.scipy.integrate.trapezoid(integrals_x[:, 1], x_vals)
    integral_y_phi = jax.scipy.integrate.trapezoid(integrals_x[:, 2], x_vals)
    integral_phi = jnp.where(integral_phi < 1e-300, 1.0, integral_phi)

    center_x = integral_x_phi / integral_phi
    center_y = integral_y_phi / integral_phi
    return jnp.array([center_x, center_y])


def lambda_func_fixed_grid(cur_pos, eta, t, y_lowers, y_uppers, x_ranges, weights, n_x=20, n_y=20, regions=None):
    """Compute lambda_eta using fixed-grid integration for each region."""
    lambda_t = jnp.exp(-t)
    sigma2_t = 1 - lambda_t**2
    n_regions = len(weights)

    def compute_integral_region(region_idx):
        if regions is not None:
            region = regions[region_idx]
            x_min, x_max, y_min, y_max = region["bbox"]
            x_vals = jnp.linspace(x_min, x_max, n_x)
            y_vals = jnp.linspace(y_min, y_max, n_y)
            xx, yy = jnp.meshgrid(x_vals, y_vals, indexing="xy")
            points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

            diffs = points - cur_pos
            exponent = -0.5 * jnp.sum(diffs**2, axis=1) / sigma2_t
            phi_vals = jnp.exp(exponent) / (2 * jnp.pi * sigma2_t)
            mask = region["constraint"](points).astype(jnp.float64)

            dx = (x_max - x_min) / jnp.maximum(n_x - 1, 1)
            dy = (y_max - y_min) / jnp.maximum(n_y - 1, 1)
            return jnp.sum(phi_vals * mask) * dx * dy

        x_min, x_max = x_ranges[region_idx]
        y_lower = y_lowers[region_idx]
        y_upper = y_uppers[region_idx]
        x_vals = jnp.linspace(x_min, x_max, n_x)

        def integrate_over_y(x):
            y_min = y_lower(x)
            y_max = y_upper(x)
            y_vals = jnp.linspace(y_min, y_max, n_y)

            def compute_phi(y):
                point = jnp.array([x, y])
                diff = point - cur_pos
                exponent = -0.5 * jnp.sum(diff**2) / sigma2_t
                return jnp.exp(exponent) / (2 * jnp.pi * sigma2_t)

            phi_vals = jax.vmap(compute_phi)(y_vals)
            return jax.scipy.integrate.trapezoid(phi_vals, y_vals)

        integrals_y = jax.vmap(integrate_over_y)(x_vals)
        return jax.scipy.integrate.trapezoid(integrals_y, x_vals)

    integrals = jnp.array([compute_integral_region(i) for i in range(n_regions)])
    weighted_integrals = weights * integrals
    integral_phi_eta = weighted_integrals[eta]
    integral_phi_total = jnp.sum(weighted_integrals)
    integral_phi_total = jnp.where(integral_phi_total < 1e-300, 1.0, integral_phi_total)
    return integral_phi_eta / integral_phi_total
