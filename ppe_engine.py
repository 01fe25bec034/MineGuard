# MineGuard PPE decision interface
#
# Later, the camera/vision model will provide these two
# boolean inputs.

def check_ppe(helmet_detected, vest_detected):

    missing = []

    if not helmet_detected:
        missing.append("helmet")

    if not vest_detected:
        missing.append("hi-vis vest")

    return {
        "ok": len(missing) == 0,
        "helmet": helmet_detected,
        "vest": vest_detected,
        "missing": missing,
        "status": "PPE OK" if not missing else "PPE MISSING",
    }