#!/usr/bin/env python3
# Tiny Shakespeare Transformer (char-level GPT-mini) + save + REPL (streaming)
# deps: torch, numpy (+ stdlib)
import os, math, time, argparse, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Utilities
# -----------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f: return f.read()

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return chars, stoi, itos

def encode(text, stoi):
    return np.fromiter((stoi[c] for c in text), dtype=np.int64, count=len(text))

def split_train_valid(ids, valid_frac=0.1):
    n = len(ids); v = int(n*valid_frac)
    return ids[:n-v], ids[n-v:]

def get_device(arg):
    if arg == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return arg

# -----------------------
# Dataset / batching
# -----------------------
class CharDataset:
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
    def __len__(self): return len(self.ids) - self.block_size
    def get_batch(self, batch_size, device):
        n = len(self)
        ix = torch.randint(low=0, high=n, size=(batch_size,))
        x = torch.stack([self.ids[i:i+self.block_size] for i in ix])
        y = torch.stack([self.ids[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)

# -----------------------
# Model (GPT-mini)
# -----------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1,1,block_size,block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size, mlp_mult=4, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult*d_model),
            nn.GELU(),
            nn.Linear(mlp_mult*d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=256, n_layer=6, n_head=8, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, block_size, mlp_mult=4, attn_pdrop=dropout, resid_pdrop=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length > block_size"
        tok = self.tok_emb(idx)
        x = tok + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1), reduction='mean')
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens=400, temperature=1.0, top_k=None, yield_topk=0):
        """
        Stream-yield (token_id, [(id,prob),...topk]) one step at a time.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_id], dim=1)
            if yield_topk and yield_topk > 0:
                pvals, pidx = torch.topk(probs, k=min(yield_topk, probs.size(-1)), dim=-1)
                yield int(next_id[0,0].item()), list(zip(pidx[0].tolist(), pvals[0].tolist()))
            else:
                yield int(next_id[0,0].item()), None

# -----------------------
# Eval helpers
# -----------------------
@torch.no_grad()
def evaluate(model, ids_val, block_size, device):
    model.eval()
    nll_sum, n_tok = 0.0, 0
    ids = torch.tensor(ids_val, dtype=torch.long)
    for start in range(0, len(ids)-block_size-1, block_size):
        x = ids[start:start+block_size].unsqueeze(0).to(device)
        y = ids[start+1:start+block_size+1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        n = x.numel()
        nll_sum += loss.item() * n
        n_tok += n
    nll = nll_sum / max(1, n_tok)
    ppl = math.exp(nll)
    return nll, ppl

# -----------------------
# Train
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Tiny Shakespeare Transformer (char GPT-mini) with save+REPL")
    ap.add_argument("--txt", type=str, default="input.txt")
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--sample_len", type=int, default=400)
    ap.add_argument("--top_k", type=int, default=50)

    # Saving / REPL
    ap.add_argument("--load_path", type=str, default="tinyshake_transformer.pt")
    ap.add_argument("--save_path", type=str, default="tinyshake_transformer.pt")
    ap.add_argument("--force_save", action="store_true", help="overwrite save_path without asking")
    ap.add_argument("--repl_max_new", type=int, default=400, help="max new chars per REPL generation")
    ap.add_argument("--repl_temp", type=float, default=0.8)
    ap.add_argument("--repl_top_k", type=int, default=50)
    ap.add_argument("--repl_topk_debug", type=int, default=5, help="when debug on, show top-K per step")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.load_path and os.path.exists(args.load_path):
        # ----- Load pretrained checkpoint (config + vocab + weights) -----
        ckpt = torch.load(args.load_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt and "vocab" in ckpt and "config" in ckpt:
            # restore hyperparameters so the instantiated model matches the checkpoint
            cfg = ckpt["config"]
            for k in ("block_size", "d_model", "n_layer", "n_head", "dropout"):
                if k in cfg:
                    setattr(args, k, cfg[k])

            # restore vocabulary for REPL
            vocab = ckpt["vocab"]
            chars, stoi, itos = vocab["chars"], vocab["stoi"], vocab["itos"]
            V = len(chars)

            # build and load model
            model = GPTMini(
                vocab_size=V,
                block_size=args.block_size,
                d_model=args.d_model,
                n_layer=args.n_layer,
                n_head=args.n_head,
                dropout=args.dropout,
            ).to(device)
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            if missing or unexpected:
                print(f"Warning: while loading, missing={missing}, unexpected={unexpected}", file=sys.stderr)
            print(f"Loaded checkpoint from {args.load_path} (V={V}, block_size={args.block_size}, "
                f"d_model={args.d_model}, n_layer={args.n_layer}, n_head={args.n_head}, dropout={args.dropout})")
        else:
            # Fallback: try to rebuild vocab from --txt, then load raw state_dict (if provided)
            if not os.path.exists(args.txt):
                print(f"ERROR: '{args.load_path}' is not a full checkpoint and --txt not found to rebuild vocab.", file=sys.stderr)
                sys.exit(1)
            text = load_text(args.txt)
            chars, stoi, itos = build_vocab(text)
            V = len(chars)
            model = GPTMini(
                vocab_size=V,
                block_size=args.block_size,
                d_model=args.d_model,
                n_layer=args.n_layer,
                n_head=args.n_head,
                dropout=args.dropout,
            ).to(device)
            if isinstance(ckpt, dict):
                try:
                    model.load_state_dict(ckpt, strict=False)  # raw state_dict case
                    print(f"Loaded raw state_dict from {args.load_path} with vocab rebuilt from {args.txt} (V={V}).")
                except Exception as e:
                    print(f"ERROR loading state_dict from {args.load_path}: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"ERROR: Unrecognized checkpoint format in {args.load_path}.", file=sys.stderr)
                sys.exit(1)
    else:
        # Load and prep data
        if not os.path.exists(args.txt):
            print(f"ERROR: file not found: {args.txt}", file=sys.stderr); sys.exit(1)
        text = load_text(args.txt)
        chars, stoi, itos = build_vocab(text)
        V = len(chars)
        ids = encode(text, stoi)
        ids_tr, ids_va = split_train_valid(ids, args.valid_frac)

        train_ds = CharDataset(ids_tr, args.block_size)

        model = GPTMini(
            vocab_size=V, block_size=args.block_size,
            d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
            dropout=args.dropout
        ).to(device)

        # optimizer & schedule
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

        def lr_at(step):
            if step < args.warmup_steps:
                return args.lr * (step / max(1, args.warmup_steps))
            t = min(1.0, (step - args.warmup_steps)/max(1, args.max_steps - args.warmup_steps))
            minlr = args.lr * 0.1
            return minlr + 0.5*(args.lr - minlr)*(1 + math.cos(math.pi*t))

        # Train
        global_step = 0
        t0 = time.time()
        model.train()
        for epoch in range(1, args.epochs+1):
            for it in range(args.steps_per_epoch):
                x, y = train_ds.get_batch(args.batch_size, device)
                for pg in optim.param_groups:
                    pg["lr"] = lr_at(global_step)
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                global_step += 1
                if (it+1) % 100 == 0:
                    print(f"epoch {epoch} step {it+1}/{args.steps_per_epoch} | train loss {loss.item():.4f} | lr {optim.param_groups[0]['lr']:.2e}")

            # epoch-end validation
            val_nll, val_ppl = evaluate(model, ids_va, args.block_size, device)
            print(f"\n[EPOCH {epoch}] Validation NLL (nats/token): {val_nll:.4f} | Perplexity: {val_ppl:.4f}")

        print(f"\nTraining finished in {(time.time()-t0):.1f}s")

        # Final eval
        val_nll, val_ppl = evaluate(model, ids_va, args.block_size, device)
        print(f"\nFinal Validation NLL (nats/token): {val_nll:.4f}")
        print(f"Final Validation Perplexity:       {val_ppl:.4f}")

        # -----------------------
        # Save model (+ prompt if exists)
        # -----------------------
        def do_save(path):
            payload = {
                "state_dict": model.state_dict(),
                "vocab": {"chars": chars, "stoi": stoi, "itos": itos},
                "config": {
                    "block_size": args.block_size, "d_model": args.d_model,
                    "n_layer": args.n_layer, "n_head": args.n_head, "dropout": args.dropout
                }
            }
            torch.save(payload, path)

        if args.save_path:
            if os.path.exists(args.save_path) and not args.force_save:
                ans = input(f"\nFile '{args.save_path}' exists. Overwrite? [y/N]: ").strip().lower()
                if ans in ("y","yes"):
                    do_save(args.save_path)
                    print(f"Saved to {args.save_path}")
                else:
                    print("Skipped saving.")
            else:
                do_save(args.save_path)
                print(f"Saved to {args.save_path}")

    # -----------------------
    # REPL (streaming)
    # -----------------------
    model.eval()
    print("\nEntering REPL. Type a prompt and press Enter to generate.")
    print("Commands: ':q' to quit, ':d' to toggle debug (top-k stream).")
    debug = False
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if line.strip() == ":q":
            print("bye.")
            break
        if line.strip() == ":d":
            debug = not debug
            print(f"(debug {'ON' if debug else 'OFF'})")
            continue
        # build context
        ctx_ids = [stoi.get(ch, 0) for ch in line]
        if not ctx_ids:
            ctx_ids = [0]
        idx = torch.tensor([ctx_ids], dtype=torch.long, device=device)

        # stream generation
        topk_k = args.repl_topk_debug if debug else 0
        gen = model.generate_stream(
            idx, max_new_tokens=args.repl_max_new,
            temperature=args.repl_temp, top_k=args.repl_top_k,
            yield_topk=topk_k
        )

        sys.stdout.write(line)  # echo prompt as part of stream
        sys.stdout.flush()
        for next_id, topk in gen:
            ch = itos[next_id]
            sys.stdout.write(ch)
            sys.stdout.flush()
            if debug and topk is not None:
                # Print a compact probs line for this step
                desc = " ".join([f"{itos[i]!r}:{p:.2f}" for (i,p) in topk])
                sys.stdout.write(f"  ← [{desc}]\n")
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
