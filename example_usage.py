"""
Demonstration of Hidden Markov Model Viterbi Decoder Skill
"""

from client import HiddenMarkovModel

def main():
    print("=== Hidden Markov Model Viterbi Sequence Decoding ===")
    states = ["HEALTHY", "FEVER"]
    init_p = {"HEALTHY": 0.6, "FEVER": 0.4}

    # Transition matrix
    trans = {
        "HEALTHY": {"HEALTHY": 0.7, "FEVER": 0.3},
        "FEVER": {"HEALTHY": 0.4, "FEVER": 0.6}
    }

    # Emission matrix: observations = ["normal", "cold", "dizzy"]
    emissions = {
        "HEALTHY": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
        "FEVER": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6}
    }

    hmm = HiddenMarkovModel(states, init_p, trans, emissions)

    observed_stream = ["normal", "cold", "dizzy"]
    print("Observed Patient Symptoms:", observed_stream)

    path, log_prob = hmm.decode_viterbi(observed_stream)
    print(f"Decoded Optimal Hidden State Trajectory: {path}")
    print(f"Log Probability: {log_prob:.4f}")

    assert path == ["HEALTHY", "HEALTHY", "FEVER"]
    print("\nHMM Viterbi Decoder Verification PASS!")

if __name__ == "__main__":
    main()
