import numpy as np


# -----------------------------
# CONFIDENCE CALCULATION
# -----------------------------

def calculate_confidence(scores):

    scores = np.array(scores)

    std = np.std(scores)

    confidence = 1 - std

    return max(0.1, confidence)


# -----------------------------
# DYNAMIC WEIGHTING
# -----------------------------

def dynamic_weights(weather, satellite, social):

    weather_conf = calculate_confidence([weather])
    sat_conf = calculate_confidence([satellite])
    social_conf = calculate_confidence([social])

    total = weather_conf + sat_conf + social_conf

    w_weather = weather_conf / total
    w_sat = sat_conf / total
    w_social = social_conf / total

    return w_weather, w_sat, w_social


# -----------------------------
# FUSION ENGINE
# -----------------------------

def dynamic_fusion(weather, satellite, social):

    w_weather, w_sat, w_social = dynamic_weights(weather, satellite, social)

    final_risk = (
        w_weather * weather +
        w_sat * satellite +
        w_social * social
    )

    return final_risk, (w_weather, w_sat, w_social)