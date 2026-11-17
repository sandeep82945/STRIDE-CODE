# Exact definitions of the four settings

All settings start from the same downloaded model in the same training mode:
full-parameter training by default, or fresh LoRA adapters when selected. There
is no warmup or structural reward. They share prompts, parsed output format,
answer reward and GRPO policy loss. Answer correctness does not independently
verify every reasoning step. STRIDE bonuses require at least two answer-correct,
parser-eligible responses to the same question. The union of their original
mapped token spans is unchanged across all STRIDE variants.

Let T_ij be the tokens of step j in response i and h_it the frozen reference
hidden states. The existing kernel uses normalized mean step embeddings and
Cholesky conditional variances, conditioning on other eligible responses plus
the target response's preceding steps. Its positive score is
U_ij = clip(v_ij - epsilon, 0, 1), set to zero when U_ij <= epsilon,
with epsilon=0.001. All auxiliary advantages below are added to the same
correctness advantage. They are zero outside the original mapped tokens.

| Setting | Auxiliary advantage on a mapped token |
|---|---|
| local_positive | alpha * U_ij for its own step j |
| token_uniform | alpha * sum_j(length(T_ij) * U_ij) / sum_j(length(T_ij)) |
| single_step | alpha * U_i, recomputed using one pooled vector per eligible response |
| grpo | 0 |

For single_step, first compute the mean of h_it over the union of T_ij and
normalize that one vector. Do not average normalized step embeddings; their
magnitudes have already been lost. Unmapped gaps, answer tokens and padding are
excluded. Each eligible response now contributes one block of size1 to the
same Cholesky calculation. There are no within-response preceding blocks in
this ablation. Its single U_i is broadcast over the original union of step
tokens. The parser still sees the unmodified multi-step response.

The single-step comparison changes representation and credit granularity
together. It tests the value of decomposing a response into steps as a whole.
Its bonus magnitude can change; it is not artificially matched to local STRIDE.
The uniform comparison holds per-response auxiliary mass fixed and tests where
that mass is assigned. Identical mass does not imply identical gradient norms.

Raw features and group diagnostics record pooled vectors, U_ij, U_i, per-arm
auxiliary mass, gate activation and actual local-versus-uniform/single score
differences. Training metrics also report the fraction of groups receiving a
nonzero bonus for the selected arm. These permit checking whether the ablation
was active, rather than mistaking a closed correctness gate for method parity.

`credit.py` and `production_stride.py` remain unchanged reference code.
`experiment_credit.py` selects the original three positive/zero arms and adds
single_step. Legacy centered/DQO-style outputs are not exposed by the comparison
runner. `feature_pooling.py` preserves the original whole/step computations
while adding raw-token pooling. Source identity checks include both new files;
old optimizer checkpoints from another source version cannot silently resume.
