#!/usr/bin/env python3
# N-gram (char) LM for Tiny Shakespeare with minimal deps + validation metrics
import os, math, numpy as np, hashlib, time, argparse, pdb, sys, string, collections
np.set_printoptions(precision=4, suppress=True)

def time_ms(): return int(round(time.time()*1000))

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return chars, stoi, itos

def split_train_valid(text, valid_frac):
    n = len(text); v = int(n*valid_frac)
    cut = n - v
    return text[:cut], text[cut:]

def encode(text, stoi): return np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))

def count_ngrams(ids, order):
    """
    Return:
      totals[k]: dict context->count(context)   for k in [1..order]
      nexts[k]:  dict context->Counter(next_id) for k in [1..order]
    contexts are tuples of length k-1 (empty tuple for unigram)
    """
    totals = [None]*(order+1)
    nexts  = [None]*(order+1)
    for k in range(1, order+1):
        totals[k] = collections.Counter()
        nexts[k]  = {}

    # i is the index of the NEXT symbol; context ends at i-1
    for i in range(len(ids)):
        nxt = ids[i]
        for k in range(1, order+1):
            ctx_len = k - 1
            if i - ctx_len < 0:
                continue  # not enough history yet
            if ctx_len > 0:
                ctx = tuple(ids[i-ctx_len:i])  # previous k-1 tokens
            else:
                ctx = ()  # unigram context
            totals[k][ctx] += 1
            d = nexts[k].get(ctx)
            if d is None:
                d = collections.Counter()
                nexts[k][ctx] = d
            d[nxt] += 1
    return totals, nexts

def parse_lams(order, lams_str):
    if lams_str:
        parts = [float(x) for x in lams_str.split(",")]
        if len(parts) != order:
            print(f"ERROR: need {order} lambdas (unigram..{order}-gram).", file=sys.stderr); sys.exit(1)
        s = sum(parts)
        if not (abs(s-1.0) < 1e-6):
            parts = [x/s for x in parts]  # normalize defensively
        return parts
    # default: geometric-ish taper toward higher order, sum to 1
    base = np.array([0.5**(order-k) for k in range(1, order+1)], dtype=np.float64)
    base /= base.sum()
    return base.tolist()

def prob_interpolated(next_id, hist, V, order, totals, nexts, lams, eps=0.0):
    """
    Jelinek–Mercer interpolation over orders 1..N
    P = sum_k λ_k * ML_k(next | last k-1)
    where ML_k(c|h) = count(h,c)/count(h); if unseen context, its ML_k contributes 0.
    eps can be >0 to add tiny floor to avoid p=0 under extreme sparsity; default 0.
    """
    p = 0.0
    for k in range(1, order+1):
        ctx_len = k-1
        ctx = tuple(hist[-ctx_len:]) if ctx_len>0 else ()
        denom = totals[k].get(ctx, 0)
        if denom > 0:
            cnt = nexts[k][ctx].get(next_id, 0)
            ml = cnt / denom
            p += lams[k-1] * ml
        # else zero contribution
    if p <= 0.0 and eps>0.0:
        p = eps
    return p

def evaluate(ids_val, V, order, totals, nexts, lams):
    nll = 0.0; n = 0
    # build rolling history
    for i in range(1, len(ids_val)):
        # history is up to order-1 previous ids
        start = max(0, i-(order))
        hist = ids_val[start:i]
        p = prob_interpolated(ids_val[i], hist, V, order, totals, nexts, lams, eps=0.0)
        if p <= 0.0:  # absolute backstop to avoid -inf (rare for Shakespeare with interpolation)
            p = 1e-300
        nll += -math.log(p)
        n += 1
    return (nll / max(1,n), math.exp(nll/max(1,n)), n)

def sample(V, itos, order, totals, nexts, lams, start_text="\n", length=400, seed=123):
    rng = np.random.default_rng(seed)
    # seed history from start_text (falling back to empty)
    hist = []
    for ch in start_text:
        if ch in itos.values(): pass
    out = list(start_text)
    # map char->id once
    ch2id = {v:k for k,v in itos.items()}
    hist = [ch2id.get(ch, None) for ch in out]
    hist = [h for h in hist if h is not None]
    if len(hist) == 0:
        hist = [0]
        out = [itos[0]]
    for _ in range(length):
        # construct distribution over next ids
        probs = np.zeros(V, dtype=np.float64)
        for j in range(V):
            probs[j] = prob_interpolated(j, hist[-(order-1):], V, order, totals, nexts, lams, eps=0.0)
        s = probs.sum()
        if s <= 0.0:
            probs[:] = 1.0 / V
        else:
            probs /= s
        nxt = int(rng.choice(V, p=probs))
        out.append(itos[nxt])
        hist.append(nxt)
    return "".join(out)

def main():
    ap = argparse.ArgumentParser(description="Tiny Shakespeare N-gram (char) LM")
    ap.add_argument("--txt", type=str, default="input.txt")
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=2, help="ngram order (2=bigram, 5=pentagram, ...)")
    ap.add_argument("--lams", type=str, default="", help="comma weights (unigram..N-gram) summing to 1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sample_len", type=int, default=400)
    args = ap.parse_args()

    if args.order < 1: print("ERROR: --order must be >=1", file=sys.stderr); sys.exit(1)

    t0 = time_ms()
    if not os.path.exists(args.txt):
        print(f"ERROR: file not found: {args.txt}", file=sys.stderr); sys.exit(1)
    text = load_text(args.txt)
    chars, stoi, itos = build_vocab(text)
    V = len(chars)

    train_text, val_text = split_train_valid(text, args.valid_frac)
    ids_tr = encode(train_text, stoi)
    ids_va = encode(val_text,  stoi)

    lams = parse_lams(args.order, args.lams)

    t1 = time_ms()
    totals, nexts = count_ngrams(ids_tr, args.order)
    t2 = time_ms()

    nll, ppl, n_pairs = evaluate(ids_va, V, args.order, totals, nexts, lams)
    t3 = time_ms()

    # diagnostics
    nz_by_k = []
    for k in range(1, args.order+1):
        nz = sum(len(nexts[k][ctx]) for ctx in nexts[k].keys())
        nz_by_k.append(nz)

    print("\n=== N-gram LM (character-level) ===")
    print(f"File: {args.txt}")
    print(f"Vocab size: V={V}")
    print(f"Order: {args.order} (window size = {args.order-1})")
    print(f"Lambdas (unigram..{args.order}-gram): {', '.join(f'{x:.4f}' for x in lams)}")
    print(f"Train chars: {len(ids_tr):,} | Valid chars: {len(ids_va):,} | Valid bigrams: {n_pairs:,}")
    print(f"Timings (ms): load+prep={t1-t0} | count={t2-t1} | eval={t3-t2}")
    for k in range(1, args.order+1):
        print(f"  k={k}: contexts={len(totals[k]):,} | nonzero transitions={nz_by_k[k-1]:,}")

    print(f"\nValidation NLL (nats/token): {nll:.4f}")
    print(f"Validation Perplexity:       {ppl:.4f}")

    print("\n=== SAMPLE ===")
    print(sample(V, itos, args.order, totals, nexts, lams, start_text="Wherefore art thou Romeo?\n", length=args.sample_len, seed=args.seed))

if __name__ == "__main__":
    main()
