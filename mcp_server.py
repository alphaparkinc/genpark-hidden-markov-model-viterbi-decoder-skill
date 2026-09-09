"""
MCP Server for Hidden Markov Model Viterbi Decoder Skill
"""

import json
import sys
from client import HiddenMarkovModel

def handle_call(name: str, args: dict) -> dict:
    if name == "decode_hmm":
        st = args.get("states", [])
        init_p = args.get("initial_probs", {})
        trans = args.get("transitions", {})
        emis = args.get("emissions", {})
        obs = args.get("observations", [])
        hmm = HiddenMarkovModel(st, init_p, trans, emis)
        path, lp = hmm.decode_viterbi(obs)
        return {"optimal_path": path, "log_prob": lp}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
