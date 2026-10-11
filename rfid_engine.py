# MineGuard RFID authorization engine

WORKERS = {
    "MG001": {
        "name": "Worker 01",
        "zones": {"ZONE 1", "ZONE 2"},
    },
    "MG002": {
        "name": "Worker 02",
        "zones": {"ZONE 2", "ZONE 3"},
    },
    "MG003": {
        "name": "Supervisor",
        "zones": {"ZONE 1", "ZONE 2", "ZONE 3"},
    },
}


def check_authorization(card_id, zone):
    worker = WORKERS.get(card_id)

    if worker is None:
        return {
            "authorized": False,
            "worker": "UNKNOWN",
            "card_id": card_id,
            "zone": zone,
            "reason": "Unknown RFID card",
        }

    authorized = zone in worker["zones"]

    return {
        "authorized": authorized,
        "worker": worker["name"],
        "card_id": card_id,
        "zone": zone,
        "reason": (
            "Worker authorized for zone"
            if authorized
            else "Worker not authorized for zone"
        ),
    }