# vetting/teams_alert.py

from datetime import datetime as dt
import requests

from config import Config

SEVERITY = {
    "ok":   {"style": "good",      "color": "Good",      "label": "SUCCESS"},
    "low":  {"style": "good",      "color": "Good",      "label": "LOW SEVERITY"},
    "med":  {"style": "warning",   "color": "Warning",   "label": "MEDIUM SEVERITY"},
    "high": {"style": "attention", "color": "Attention", "label": "HIGH SEVERITY"},
}

def send_teams_alert(cfg: Config, severity: str, title: str, message: str = "", details: dict | None = None):
    """
    Send a model status alert to Microsoft Teams. To set this up, open Teams and go to the channel you want to receive
    alerts in. Go to Channel > Workflows > "Send webhook alerts" > Copy URL. Then paste the url into the .env with
    TEAMS_WEBHOOK_URL=https://...
    Do not publish the link to GitHub or share it publicly as anyone with the link can send messages to the channel.

    :param cfg: Configuration object containing environment variables set in the .env
    :param severity: Alert severity ("ok", "low", "med", "high")
    :param title: Alert title
    :param message: Text body of the alert
    :param details: Dictionary of tabulated details about the alert

    Example:
        send_teams_alert(cfg, "med", "2026-11-10 Run Succeeded with Issues",
            "Max SWE value exceeds threshold",
            details={"Domain": "PNW", "Max SWE (in)": 3280.04, "Mean SWE (in)": 5.4}
        )

    """
    s = SEVERITY[severity]

    body = [
        {
            "type": "Container",
            "style": s["style"],
            "bleed": True,
            "items": [
                {
                    "type": "TextBlock",
                    "text": title,
                    "weight": "Bolder",
                    "size": "Large",
                    "color": s["color"],
                    "wrap": True,
                },
                {
                    "type": "TextBlock",
                    "text": s["label"],
                    "isSubtle": True,
                    "spacing": "None",
                },
            ],
        }
    ]

    if message:
        body.append({"type": "TextBlock", "text": message, "wrap": True})

    facts = {"Time": dt.now().strftime("%Y-%m-%d %H:%M:%S")}
    facts.update(details or {})
    body.append({
        "type": "FactSet",
        "facts": [{"title": k, "value": str(v)} for k, v in facts.items()],
    })

    payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "msteams": {"width": "Full"},
                "body": body,
            },
        }],
    }

    r = requests.post(cfg.teams_webhook_url, json=payload, timeout=15)
    r.raise_for_status()

if __name__ == "__main__":
    import time
    cfg = Config()
    # send_teams_alert(cfg, "high", "2026-11-10 Domain Failed",
    #            "SNM domain failed to run due to too low percentage of sensors with SWE > 0",
    #            details={"Domain": "SNM", "\\# Sensors": 111, "\\# Sensors > 0": 29, "% Sensors > 0": "26.1%"})
    # send_teams_alert(cfg, "ok", "2026-11-10 Run Succeeded")
    # time.sleep(3)
    # send_teams_alert(cfg, "low", "2026-11-10 Run Succeeded with Warnings",
    #            message="Failed to download sensor data for 3 stations",
    #            details={"SNOTEL": 409, "CDEC": "MRL, PDS"}
    #            )
    # time.sleep(3)
    # send_teams_alert(cfg, "med", "2026-11-10 Run Succeeded with Issues",
    #            "Max SWE value exceeds threshold",
    #             details={"Domain": "PNW", "Max SWE (in)": 3280.04, "Mean SWE (in)": 5.4}
    #            )