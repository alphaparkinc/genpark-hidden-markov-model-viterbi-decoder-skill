"""
Hidden Markov Model Viterbi Decoder Skill Client
Pure Python Standard Library implementation of the Viterbi Dynamic Programming Algorithm (Rabiner).
Computes the maximum a posteriori (MAP) path of latent states given an observation sequence.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class HiddenMarkovModel:
    def __init__(self, states: List[str], initial_probs: Dict[str, float],
                 transition_matrix: Dict[str, Dict[str, float]],
                 emission_matrix: Dict[str, Dict[str, float]]):
        self.states = states
        self.initial_probs = initial_probs
        self.transitions = transition_matrix  # P(next | curr)
        self.emissions = emission_matrix      # P(obs | state)

    def decode_viterbi(self, observations: List[str]) -> Tuple[List[str], float]:
        """
        Run Viterbi decoding using log-probabilities to prevent numerical underflow.
        Returns (optimal_state_sequence, path_log_prob).
        """
        if not observations:
            return [], 0.0

        # T1: state -> log_prob; T2: state -> backpointer state
        viterbi_trellis: List[Dict[str, float]] = []
        backpointers: List[Dict[str, str]] = []

        # Initialization step (t = 0)
        first_obs = observations[0]
        init_trellis = {}
        for s in self.states:
            pi = self.initial_probs.get(s, 0.0)
            e = self.emissions.get(s, {}).get(first_obs, 0.0)
            if pi > 0 and e > 0:
                init_trellis[s] = math.log(pi) + math.log(e)
            else:
                init_trellis[s] = float("-inf")

        viterbi_trellis.append(init_trellis)

        # Recursion step (t = 1 to T-1)
        for t in range(1, len(observations)):
            obs = observations[t]
            curr_trellis = {}
            curr_backpointer = {}

            for curr_s in self.states:
                e = self.emissions.get(curr_s, {}).get(obs, 0.0)
                e_log = math.log(e) if e > 0 else float("-inf")

                best_prob = float("-inf")
                best_prev_s = None

                for prev_s in self.states:
                    prev_prob = viterbi_trellis[t - 1][prev_s]
                    trans = self.transitions.get(prev_s, {}).get(curr_s, 0.0)
                    trans_log = math.log(trans) if trans > 0 else float("-inf")

                    candidate = prev_prob + trans_log + e_log
                    if candidate > best_prob:
                        best_prob = candidate
                        best_prev_s = prev_s

                curr_trellis[curr_s] = best_prob
                curr_backpointer[curr_s] = best_prev_s

            viterbi_trellis.append(curr_trellis)
            backpointers.append(curr_backpointer)

        # Termination: Find best final state
        last_trellis = viterbi_trellis[-1]
        best_last_state = max(last_trellis, key=last_trellis.get)
        best_final_log_prob = last_trellis[best_last_state]

        # Backtracking
        best_path = [best_last_state]
        for t in range(len(observations) - 2, -1, -1):
            prev_s = backpointers[t][best_path[-1]]
            best_path.append(prev_s)

        best_path.reverse()
        return best_path, best_final_log_prob
