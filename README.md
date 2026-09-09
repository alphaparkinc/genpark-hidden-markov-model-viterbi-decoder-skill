# GenPark Hidden Markov Model Viterbi Decoder Skill

Viterbi dynamic programming algorithm for decoding maximum a posteriori (MAP) state sequences in Hidden Markov Models.

Check out [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph LR
    subgraph t=0
        H0[HEALTHY]
        F0[FEVER]
    end
    subgraph t=1
        H1[HEALTHY]
        F1[FEVER]
    end
    subgraph t=2
        H2[HEALTHY]
        F2[FEVER]
    end
    H0 -->|Viterbi Trellis| H1
    H1 -->|Viterbi Trellis| F2
    style H0 fill:#e8f5e9
    style H1 fill:#e8f5e9
    style F2 fill:#ffebee
```

## Features
- Dynamic programming log-space calculations avoiding underflow.
- Exact backpointer sequence path reconstruction.
- Pure Python standard library.
