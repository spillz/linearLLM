#!/usr/bin/env python3
# N-gram (char) LM for Tiny Shakespeare with minimal deps + validation metrics
import os, math, numpy as np, time, argparse, sys, collections
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

def encode(text, stoi):
    return np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))

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
            ctx = tuple(ids[i-ctx_len:i]) if ctx_len > 0 else ()
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
    if p <= 0.0 and eps>0.0:
        p = eps
    return p

def evaluate(ids_val, V, order, totals, nexts, lams):
    nll = 0.0; n = 0
    for i in range(1, len(ids_val)):
        # history is up to order-1 previous ids
        start = max(0, i-(order-1))
        hist = ids_val[start:i]
        p = prob_interpolated(ids_val[i], hist, V, order, totals, nexts, lams, eps=0.0)
        if p <= 0.0:  # backstop to avoid -inf under extreme sparsity
            p = 1e-300
        nll += -math.log(p)
        n += 1
    avg = nll / max(1, n)
    return (avg, math.exp(avg), n)

def sample(V, itos, stoi, order, totals, nexts, lams,
           start_text="\n", length=400, seed=123, eps=0.0):
    rng = np.random.default_rng(seed)
    # seed history from start_text
    out = list(start_text)
    hist = [stoi[ch] for ch in out if ch in stoi]
    if not hist:
        # deterministic start: pick most frequent unigram context ()
        # or fallback to 0
        if () in nexts[1] and len(nexts[1][()])>0:
            first = max(nexts[1][()].items(), key=lambda kv: kv[1])[0]
        else:
            first = 0
        hist = [first]; out = [itos[first]]
    for _ in range(length):
        # construct distribution over next ids
        probs = np.zeros(V, dtype=np.float64)
        hist_slice = hist[-(order-1):] if order > 1 else []
        for j in range(V):
            probs[j] = prob_interpolated(j, hist_slice, V, order, totals, nexts, lams, eps=eps)
        s = probs.sum()
        if s <= 0.0:
            probs[:] = 1.0 / V
        else:
            probs /= s
        nxt = int(rng.choice(V, p=probs))
        out.append(itos[nxt])
        hist.append(nxt)
    return "".join(out)

# ---------- Line-input REPL with streaming model output ----------
def stream_sample(itos, V, order, totals, nexts, lams, hist_ids,
                  max_len=200, rng=None, eps=0.0, stop_at_newline=False):
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_len):
        probs = np.zeros(V, dtype=np.float64)
        hist_slice = hist_ids[-(order-1):] if order > 1 else []
        for j in range(V):
            probs[j] = prob_interpolated(j, hist_slice, V, order, totals, nexts, lams, eps=eps)
        s = probs.sum()
        if s <= 0.0:
            probs[:] = 1.0 / V
        else:
            probs /= s
        nxt = int(rng.choice(V, p=probs))
        ch = itos[nxt]
        yield ch
        hist_ids.append(nxt)
        if stop_at_newline and ch == '\n':
            break

def repl_line_stream(itos, stoi, V, order, totals, nexts, lams,
                     *, seed=123, add_newline_after_input=True,
                     gen_len=200, stop_at_newline=False, eps=0.0):
    """
    Line input from user; model streams its continuation char-by-char.
    Commands:
      :quit            -> exit
      :clear           -> clear context
      :len N           -> set generation length
      :seed N          -> set RNG seed
      :nonl            -> toggle add_newline_after_input
      :stopnl          -> toggle stop_at_newline
    """
    rng = np.random.default_rng(seed)
    buf = []       # context chars (for display)
    buf_ids = []   # context as ids

    print("\n=== N-gram REPL (line input, streaming output) ===")
    print("Type a line and press Enter; the model will stream its continuation.")
    print("Commands: :quit, :clear, :len N, :seed N, :nonl, :stopnl\n")

    while True:
        try:
            prompt_ctx = "".join(buf[-max(0, order-1):]) if order > 1 else "∅"
            line = input(f"[ctx={repr(prompt_ctx)}] >>> ")
        except EOFError:
            print("\nExiting REPL."); break
        except KeyboardInterrupt:
            print("\nExiting REPL."); break

        # commands
        if line.strip() == ":quit":
            print("Bye."); break
        if line.strip() == ":clear":
            buf.clear(); buf_ids.clear()
            print("[cleared]")
            continue
        if line.startswith(":len "):
            try:
                gen_len = max(0, int(line.split(None, 1)[1]))
                print(f"[gen_len set to {gen_len}]")
            except Exception:
                print("[usage] :len N")
            continue
        if line.startswith(":seed "):
            try:
                seed = int(line.split(None, 1)[1])
                rng = np.random.default_rng(seed)
                print(f"[seed set to {seed}]")
            except Exception:
                print("[usage] :seed N")
            continue
        if line.strip() == ":nonl":
            add_newline_after_input = not add_newline_after_input
            print(f"[add_newline_after_input={add_newline_after_input}]")
            continue
        if line.strip() == ":stopnl":
            stop_at_newline = not stop_at_newline
            print(f"[stop_at_newline={stop_at_newline}]")
            continue

        # append user text to context (filter to vocab)
        for ch in line:
            if ch in stoi:
                buf.append(ch); buf_ids.append(stoi[ch])
        if add_newline_after_input and '\n' in stoi:
            buf.append('\n'); buf_ids.append(stoi['\n'])

        # stream model continuation
        import sys
        for ch in stream_sample(itos, V, order, totals, nexts, lams,
                                hist_ids=buf_ids, max_len=gen_len, rng=rng,
                                eps=eps, stop_at_newline=stop_at_newline):
            sys.stdout.write(ch); sys.stdout.flush()
            buf.append(ch)
        print("")  # neat line end

def main():
    ap = argparse.ArgumentParser(description="Tiny Shakespeare N-gram (char) LM")
    ap.add_argument("--txt", type=str, default="input.txt")
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=2, help="ngram order (2=bigram, 5=pentagram, ...)")
    ap.add_argument("--lams", type=str, default="", help="comma weights (unigram..N-gram) summing to 1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sample_len", type=int, default=400)
    # REPL (line-input, streaming output)
    ap.add_argument("--repl", action="store_true", help="start REPL (line input; model streams chars)")
    ap.add_argument("--repl_len", type=int, default=200, help="REPL: max chars to stream after each line")
    ap.add_argument("--repl_stopnl", action="store_true", help="REPL: stop streaming at first newline")
    ap.add_argument("--repl_addnl", action="store_true", help="REPL: append a newline after user input")
    ap.add_argument("--repl_eps", type=float, default=0.0, help="REPL: tiny prob floor (e.g., 1e-12)")
    args = ap.parse_args()

    if args.order < 1:
        print("ERROR: --order must be >=1", file=sys.stderr); sys.exit(1)

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
    print(f"Order: {args.order} (context length = {args.order-1})")
    print(f"Lambdas (unigram..{args.order}-gram): {', '.join(f'{x:.4f}' for x in lams)}")
    print(f"Train chars: {len(ids_tr):,} | Valid chars: {len(ids_va):,} | Valid pairs: {n_pairs:,}")
    print(f"Timings (ms): load+prep={t1-t0} | count={t2-t1} | eval={t3-t2}")
    for k in range(1, args.order+1):
        print(f"  k={k}: contexts={len(totals[k]):,} | nonzero transitions={nz_by_k[k-1]:,}")

    print(f"\nValidation NLL (nats/token): {nll:.4f}")
    print(f"Validation Perplexity:       {ppl:.4f}")

    print("\n=== SAMPLE ===")
    print(sample(V, itos, stoi, args.order, totals, nexts, lams,
                 start_text="Wherefore art thou Romeo?\n",
                 length=args.sample_len, seed=args.seed, eps=0.0))

    if args.repl:
        repl_line_stream(
            itos=itos, stoi=stoi, V=V, order=args.order,
            totals=totals, nexts=nexts, lams=lams,
            seed=args.seed, add_newline_after_input=args.repl_addnl,
            gen_len=args.repl_len, stop_at_newline=args.repl_stopnl,
            eps=args.repl_eps
        )

if __name__ == "__main__":
    main()
