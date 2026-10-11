def collision_risk(distance_cm):
    """
    Prototype collision-risk logic.

    Returns:
        status: SAFE / WARNING / CRITICAL
        score: 0.0 to 1.0
        message: human-readable explanation
    """

    if distance_cm <= 30:
        return (
            "CRITICAL",
            1.0,
            "Object is extremely close."
        )

    elif distance_cm <= 75:
        return (
            "WARNING",
            0.6,
            "Object is within the warning zone."
        )

    else:
        return (
            "SAFE",
            0.0,
            "No immediate proximity hazard detected."
        )
