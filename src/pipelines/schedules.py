"""Placement of the classifier-guidance updates over the denoising steps."""

# schedules that can be selected in the configuration file
GUIDANCE_SCHEDULES = ("throughout", "early", "late")

# how strongly "early" and "late" are skewed: the updates are placed on T * (j/k)**WARP_EXPONENT,
# so 1.0 would reproduce "throughout" and larger values push more of them towards the start
WARP_EXPONENT = 2.0


def guidance_steps(num_inference_steps, guidance_freq, schedule="throughout", exponent=WARP_EXPONENT):
    """Return the 1-based denoising steps that receive a classifier-guidance update.

    Gamma (`guidance_freq`) sets the budget, k = T // gamma updates, for every schedule, so the
    schedules only differ in where those k updates land. The gradient is normalized to the norm of
    the latents, so every update is a step of the same relative size, and an equal budget is an
    equal total guidance magnitude: a difference between the schedules is then about placement.

    "throughout" spreads them evenly, which is the original `i % gamma == 0` rule. "early" places
    them on a power-warped grid, dense at the first steps and thinning out towards the last ones,
    and "late" is its mirror image.
    """
    if guidance_freq <= 0:
        return frozenset()
    if schedule not in GUIDANCE_SCHEDULES:
        raise ValueError(f"Unknown guidance schedule {schedule!r}. Valid options are {list(GUIDANCE_SCHEDULES)}.")

    budget = num_inference_steps // guidance_freq
    if budget == 0:
        return frozenset()

    if schedule == "throughout":
        return frozenset(range(guidance_freq, num_inference_steps + 1, guidance_freq))

    steps, previous = [], 0
    for j in range(1, budget + 1):
        target = num_inference_steps * (j / budget) ** exponent
        # at the dense end several targets round onto the same step, and dropping the duplicates
        # would spend less than the budget, so they take the next free step instead
        step = max(round(target), previous + 1)
        steps.append(step)
        previous = step

    if schedule == "late":
        # mirroring the steps of "early" makes the two schedules exact reflections of each other,
        # which warping with the reciprocal exponent would not
        steps = [num_inference_steps + 1 - step for step in steps]

    return frozenset(steps)
