def triage_decision(prediction, cfg):
    """
    Convert confidence and entropy into a clinical routing decision.
    """

    confidence = prediction["confidence"]
    entropy = prediction["entropy"]

    # Low confidence means the model is not sufficiently sure.
    if confidence < cfg.confidence_threshold:
        return "defer_for_radiologist_review"

    # High entropy means the probability distribution is too uncertain.
    if entropy > cfg.entropy_threshold:
        return "defer_for_radiologist_review"

    return "accept_model_prediction"


def format_output(prediction, class_names, cfg):
    """
    Format final inference result in a readable dictionary.
    """

    predicted_class = class_names[prediction["pred_class"]]

    return {
        "predicted_class": predicted_class,
        "confidence": round(prediction["confidence"], 4),
        "entropy": round(prediction["entropy"], 4),
        "variance": round(prediction["variance"], 4),
        "triage_decision": triage_decision(prediction, cfg),
    }
