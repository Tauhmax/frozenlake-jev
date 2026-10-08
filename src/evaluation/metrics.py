def score(action, probabilities, q_values, optimal):
    correct = int(action in optimal)
    confidence = max(probabilities)
    return {
        "optimal_action_accuracy": correct,
        "q_regret": float(max(q_values) - q_values[action]),
        "optimal_probability_mass": sum(probabilities[a] for a in optimal),
        "confidence": confidence,
        "brier": (confidence - correct) ** 2,
    }
