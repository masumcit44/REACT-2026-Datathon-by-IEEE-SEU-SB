"""V8 audit: honest verification of the V7 champion plus what V8 tested."""
import os, json
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
def J(p):
    import json
    with open(p) as f: return json.load(f)
def main():
    sel=J(os.path.join(OUT,"v7_selection.json")); core=J(os.path.join(OUT,"v7_stack_results_core.json"))
    wide=J(os.path.join(OUT,"v7_stack_results_wide.json")); repro=J(os.path.join(AUD,"v7_exact_reproduction.json"))
    blends=J(os.path.join(AUD,"v8_conservative_blends.json")); abl=J(os.path.join(AUD,"v8_lofo_ablation.json"))
    cx=J(os.path.join(AUD,"v8_stack_complexity.json")); lb=J(os.path.join(AUD,"v8_leaderboard_evidence.json"))
    print("=== V8 AUDIT ===")
    print(f"champion: {sel['winner_tag']} {sel['winner_candidate']['head']} f3={sel['winner_candidate']['fold3_ap']:.6f}")
    print(f"repro LOFO f3={repro['lofo_fold3_match']} f2={repro['lofo_fold2_match']} test_exact={repro['reproduced_test_exact_match']}")
    print(f"wide rejected by coherence rule: f3={wide['chosen_candidate']['fold3_ap']:.6f}; stored test vector corrupted by expit bug (mean {wide['test_prob_mean']:.6f}), corrected audit-only copy in v8_out")
    print(f"blends verdict: {blends['verdict']} best={blends['best']}")
    print(f"C-grid best: {cx['best']['head']} f3={cx['best']['f3']:.6f} untested on LB")
    print(f"test agreement champ vs C0.01 spearman={lb['spearman']:.4f}")
    print("RECOMMENDATION: KEEP V7 champion as primary.")
if __name__=="__main__": main()
