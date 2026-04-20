"""precompute pretrained molecular embeddings for sciplex drugs

steps:
  1 load sciplex3 anndata
  2 extract unique product_name values
  3 fetch SMILES via pertpy Compound.annotate_compounds (uses pubchempy)
  4 run MolFormer-XL on each SMILES to produce a 768 dim embedding
  5 save to artifacts/drug_embeddings/molformer_xl_sciplex.pt

the saved payload is a dict:
    {
        "embeddings": dict[drug_name -> torch.Tensor(emb_dim)],
        "smiles":     dict[drug_name -> str or None],
        "dim":        int,
        "source":     str,
    }

control entries ("(no drug)" "Vehicle" etc) are represented by zero vectors
drugs for which SMILES lookup fails are also zero vectors but tracked in the
missing list for inspection

usage:
    .venv/bin/python scripts/build_drug_embeddings.py \
        --adata-path data/sciplex3.h5ad \
        --output artifacts/drug_embeddings/molformer_xl_sciplex.pt
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import anndata as ad
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import track


MOLFORMER_CHECKPOINT = "ibm/MoLFormer-XL-both-10pct"
MOLFORMER_EMBED_DIM = 768

app = typer.Typer(add_completion=False)
console = Console()


def _is_control_name(name: str) -> bool:
    """heuristic for control/vehicle rows that should not hit pubchem"""

    lowered = str(name).strip().lower()
    return lowered in {"", "vehicle", "(no drug)", "control", "dmso", "none", "nan"}


_PAREN_CODE_RE = re.compile(r"\s*\([^)]*\)")


def _simplify_name(name: str) -> str:
    """strip parenthetical code names and question marks used in sciplex labels"""

    simple = _PAREN_CODE_RE.sub("", name).replace("?", "").strip()
    return simple


def _pubchem_lookup(names: list[str]) -> dict[str, str]:
    """query pubchem for canonical SMILES one drug at a time with progress

    pertpy's annotate_compounds does this serially with no progress output
    we call pubchempy directly so we can show a live progress bar and
    print the counts of resolved vs unresolved as we go
    """

    import pubchempy as pcp

    if not names:
        return {}

    out: dict[str, str] = {}
    n_found = 0
    n_missing = 0
    for name in track(names, description="pubchem lookup", transient=False):
        try:
            compounds = pcp.get_compounds(name, namespace="name")
        except Exception as exc:
            compounds = []
            console.print(f"[yellow]pubchempy error for {name!r}: {exc}[/]")
        smi: str | None = None
        if compounds:
            compound = compounds[0]
            # try the new property name first then fall back to the
            # deprecated canonical name for older pubchempy versions
            smi = getattr(compound, "connectivity_smiles", None) or getattr(compound, "canonical_smiles", None)
        if smi:
            out[name] = str(smi).strip()
            n_found += 1
        else:
            n_missing += 1
    console.print(f"[dim]pubchem resolved {n_found}/{len(names)} missing {n_missing}[/]")
    return out


def _fetch_smiles(drug_names: list[str]) -> dict[str, str | None]:
    """resolve SMILES for each drug name via pertpy compound annotation

    does a first pass on the raw name then a second pass with parenthetical
    code names stripped (sciplex labels often look like
    "Bisindolylmaleimide IX (Ro 31-8220 Mesylate)" which pubchem cannot
    match verbatim) returns dict mapping drug_name to SMILES or None
    """

    real_names = [n for n in drug_names if not _is_control_name(n)]
    if not real_names:
        return {name: None for name in drug_names}

    console.print(f"[dim]querying pubchem for {len(real_names)} drugs via pertpy Compound[/]")
    smiles_map: dict[str, str | None] = {name: None for name in drug_names}
    smiles_map.update({k: v for k, v in _pubchem_lookup(real_names).items()})

    missing = [n for n in real_names if smiles_map.get(n) is None]
    if missing:
        retry_pairs = [
            (original, _simplify_name(original))
            for original in missing
            if _simplify_name(original) and _simplify_name(original) != original
        ]
        if retry_pairs:
            console.print(f"[dim]retrying {len(retry_pairs)} drugs with simplified names[/]")
            retry_map = _pubchem_lookup([simple for _, simple in retry_pairs])
            n_found = 0
            for original, simple in retry_pairs:
                smi = retry_map.get(simple)
                if smi:
                    smiles_map[original] = smi
                    n_found += 1
            console.print(f"[dim]retry recovered {n_found} drugs[/]")
    return smiles_map


def _load_molformer(device: str) -> tuple[object, object]:
    """load the MolFormer-XL tokenizer and model"""

    from transformers import AutoModel, AutoTokenizer

    console.print(f"[dim]loading {MOLFORMER_CHECKPOINT}[/]")
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_CHECKPOINT, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MOLFORMER_CHECKPOINT,
        deterministic_eval=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def _embed_smiles(
    smiles_map: dict[str, str | None],
    tokenizer: object,
    model: object,
    device: str,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """embed SMILES strings with MolFormer-XL and return per-drug vectors

    drugs with None SMILES get a zero vector. tracks drugs that failed lookup.
    """

    embeddings: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    resolvable = [(name, smi) for name, smi in smiles_map.items() if smi]
    for name, smi in smiles_map.items():
        if not smi:
            embeddings[name] = torch.zeros(MOLFORMER_EMBED_DIM, dtype=torch.float32)
            if not _is_control_name(name):
                missing.append(name)

    for i in track(range(0, len(resolvable), batch_size), description="embedding SMILES"):
        chunk = resolvable[i : i + batch_size]
        names, smis = zip(*chunk)
        tokens = tokenizer(list(smis), padding=True, return_tensors="pt").to(device)
        out = model(**tokens)
        # pooled representation mean over non-padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = (summed / lengths).cpu()
        for name, vec in zip(names, pooled):
            embeddings[name] = vec.to(torch.float32)

    return embeddings, missing


@app.command()
def main(
    adata_path: Annotated[Path, typer.Option("--adata-path", help="path to sciplex3 h5ad file")] = Path(
        "data/sciplex3.h5ad"
    ),
    output: Annotated[Path, typer.Option("--output", help="output pt file")] = Path(
        "artifacts/drug_embeddings/molformer_xl_sciplex.pt"
    ),
    product_column: Annotated[str, typer.Option("--product-col", help="column with drug names")] = "product_name",
    batch_size: Annotated[int, typer.Option("--batch-size", help="MolFormer inference batch size")] = 32,
    device: Annotated[str, typer.Option("--device", help="cuda or cpu")] = "cuda",
    overrides: Annotated[
        Path | None,
        typer.Option("--overrides", help="json file mapping drug_name to SMILES for pubchem misses"),
    ] = None,
) -> None:
    """build the sciplex drug embedding artifact"""

    if not adata_path.exists():
        console.print(f"[red]adata not found at {adata_path}[/]")
        raise typer.Exit(1)

    console.print(f"[dim]reading {adata_path}[/]")
    adata = ad.read_h5ad(adata_path, backed="r")
    if product_column not in adata.obs.columns:
        console.print(f"[red]{product_column} not in adata.obs[/]")
        raise typer.Exit(1)
    drug_names = sorted(str(name) for name in adata.obs[product_column].astype(str).unique())
    console.print(f"[green]found {len(drug_names)} unique {product_column} values[/]")

    # persistent smiles cache so reruns skip the slow pubchem loop
    # 188 drugs at ~0.5 s each is several minutes of serial network i/o
    cache_path = output.parent / "smiles_cache.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if set(cached.keys()) >= set(drug_names):
            console.print(f"[green]loaded smiles cache with {sum(1 for v in cached.values() if v)} resolved drugs from {cache_path}[/]")
            smiles_map = {name: cached.get(name) for name in drug_names}
        else:
            missing_in_cache = set(drug_names) - set(cached.keys())
            console.print(
                f"[yellow]smiles cache missing {len(missing_in_cache)} drugs refetching all[/]"
            )
            smiles_map = _fetch_smiles(drug_names)
    else:
        smiles_map = _fetch_smiles(drug_names)

    # persist cache before applying overrides so overrides stay separate
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({k: v for k, v in smiles_map.items()}, indent=2, default=str))
    console.print(f"[dim]wrote smiles cache to {cache_path}[/]")

    if overrides is not None:
        if not overrides.exists():
            console.print(f"[red]overrides file not found: {overrides}[/]")
            raise typer.Exit(1)
        override_map = json.loads(overrides.read_text())
        n_overridden = 0
        for name, smi in override_map.items():
            if name in smiles_map:
                smiles_map[name] = str(smi).strip()
                n_overridden += 1
            else:
                console.print(f"[yellow]override for unknown drug ignored: {name}[/]")
        console.print(f"[green]applied {n_overridden} manual SMILES overrides[/]")

    resolved = sum(1 for smi in smiles_map.values() if smi)
    console.print(f"[green]resolved SMILES for {resolved}/{len(drug_names)} drugs[/]")

    if not torch.cuda.is_available() and device == "cuda":
        console.print("[yellow]cuda not available falling back to cpu[/]")
        device = "cpu"
    tokenizer, model = _load_molformer(device)

    embeddings, missing = _embed_smiles(smiles_map, tokenizer, model, device, batch_size)
    if missing:
        console.print(f"[yellow]{len(missing)} drugs had no SMILES and got zero vectors[/]")
        console.print(f"[dim]missing examples: {missing[:5]}[/]")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "smiles": smiles_map,
        "dim": MOLFORMER_EMBED_DIM,
        "source": MOLFORMER_CHECKPOINT,
    }
    torch.save(payload, output)
    console.print(f"[green]wrote {output}[/]")

    # sidecar json with resolution stats for quick inspection
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "n_drugs": len(drug_names),
                "n_resolved": resolved,
                "n_missing_smiles": len(missing),
                "missing_examples": missing[:20],
                "source": MOLFORMER_CHECKPOINT,
                "dim": MOLFORMER_EMBED_DIM,
            },
            indent=2,
        )
    )
    console.print(f"[green]wrote {stats_path}[/]")


if __name__ == "__main__":
    app()
