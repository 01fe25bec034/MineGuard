def methane_score(ch4):
    """
    Prototype methane severity score.

    < 0.60%  -> 0.0
    0.60-1.00% -> 0.5 to 1.0
    >= 1.00% -> 1.0
    """

    if ch4 < 0.60:
        return 0.0

    if ch4 >= 1.00:
        return 1.0

    return 0.5 + (
        (ch4 - 0.60) / 0.40
    ) * 0.5


def collision_score(status):
    values = {
        "SAFE": 0.0,
        "WARNING": 0.6,
        "CRITICAL": 1.0,
        "PENDING": 0.0,
    }

    return values.get(status, 0.0)


def fuse_risk(
    ch4,
    ai_probability,
    collision_status="PENDING",
    ppe_ok=None,
    authorized=None,
):
    """
    Prototype MineGuard risk-fusion engine.

    This is a decision-support layer, not a medically/
    operationally certified safety model.
    """

    signals = []
    weights = []

    # Methane
    signals.append(
        methane_score(ch4)
    )
    weights.append(0.35)

    # Trained methane AI
    signals.append(
        max(0.0, min(1.0, float(ai_probability)))
    )
    weights.append(0.35)

    # Collision
    if collision_status != "PENDING":
        signals.append(
            collision_score(collision_status)
        )
        weights.append(0.15)

    # PPE
    if ppe_ok is not None:
        signals.append(
            0.0 if ppe_ok else 1.0
        )
        weights.append(0.10)

    # Authorization
    if authorized is not None:
        signals.append(
            0.0 if authorized else 1.0
        )
        weights.append(0.05)

    total_weight = sum(weights)

    if total_weight == 0:
        score = 0.0
    else:
        score = sum(
            s * w
            for s, w in zip(signals, weights)
        ) / total_weight

    # Safety overrides
    if ch4 >= 1.0:
        score = max(score, 0.85)

    if collision_status == "CRITICAL":
        score = max(score, 0.90)

    if ppe_ok is False:
        score = max(score, 0.55)

    if authorized is False:
        score = max(score, 0.70)

    if score >= 0.75:
        level = "CRITICAL"

    elif score >= 0.50:
        level = "HIGH"

    elif score >= 0.30:
        level = "WATCH"

    else:
        level = "SAFE"

    return score, level


def explain_risk(
    ch4,
    ai_probability,
    collision_status="PENDING",
    ppe_ok=None,
    authorized=None,
):
    reasons = []
    actions = []

    if ch4 >= 1.0:
        reasons.append(
            f"Methane is above the prototype danger threshold ({ch4:.2f}%)."
        )
        actions.append(
            "Verify the affected zone and ventilation."
        )

    elif ch4 >= 0.60:
        reasons.append(
            f"Methane is elevated ({ch4:.2f}%)."
        )
        actions.append(
            "Continue close monitoring."
        )

    if ai_probability >= 0.50:
        reasons.append(
            f"AI predicts elevated future methane risk ({ai_probability * 100:.1f}%)."
        )
        actions.append(
            "Prepare an early-warning response."
        )

    elif ai_probability >= 0.30:
        reasons.append(
            f"AI detects moderate future methane risk ({ai_probability * 100:.1f}%)."
        )

    if collision_status == "WARNING":
        reasons.append(
            "Nearby-object proximity is in the warning range."
        )
        actions.append(
            "Check vehicle/worker separation."
        )

    elif collision_status == "CRITICAL":
        reasons.append(
            "Critical proximity to a vehicle/obstacle detected."
        )
        actions.append(
            "Trigger immediate local collision alert."
        )

    if ppe_ok is False:
        reasons.append(
            "Required PPE was not detected."
        )
        actions.append(
            "Restrict/stop entry until PPE is verified."
        )

    if authorized is False:
        reasons.append(
            "Worker is not authorized for this zone."
        )
        actions.append(
            "Deny access and notify the supervisor."
        )

    if not reasons:
        reasons.append(
            "No elevated hazard signal is currently active."
        )
        actions.append(
            "Continue routine monitoring."
        )

    return reasons, actions
