"""Granted workspaces — directories HIVE is allowed to look at.

By default HIVE can only touch the vault's HIVE folder. A grant widens that, and
grants are made explicitly, by name, from the terminal — never inferred from
speech. That separation is the point: saying "organise my project folder" out
loud should not be able to hand a misheard path to a file operation.

Two access levels:

    read   list, inspect and compare structure. The default.
    write  create directories and move files inside the grant. Requires
           `grant --write`, and every mutation is still previewed and confirmed
           before anything moves.

Grants persist in the vault so they survive a restart, and are always re-checked
against the filesystem on load — a grant to a directory that has since been
deleted or replaced by a symlink is dropped rather than trusted.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Never grantable, whatever is typed. These either hold credentials or are large
# enough that walking them would hang the terminal.
#
# Split into two sets deliberately. "/" and $HOME are denied only as exact
# targets — putting them in the ancestor set would deny every path on the system,
# since "/" is an ancestor of literally everything.
_FORBIDDEN_TREES = {
    Path.home() / ".ssh", Path.home() / ".gnupg", Path.home() / ".config",
    Path.home() / ".local" / "share" / "keyrings", Path.home() / ".pki",
    Path("/etc"), Path("/proc"), Path("/sys"), Path("/dev"), Path("/boot"),
    Path("/root"),
}
_FORBIDDEN_EXACT = {Path("/"), Path.home()}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache",
              ".obsidian", ".trash", "target", "dist", "build", ".mypy_cache"}


@dataclass
class Grant:
    path: Path
    write: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    def to_json(self) -> dict:
        return {"path": str(self.path), "write": self.write}


class Workspace:
    def __init__(self, cfg, bus, ltm):
        self.cfg, self.bus, self.ltm = cfg, bus, ltm
        self.grants: dict[str, Grant] = {}

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        data = self.ltm.load_state("workspace", {"grants": []})
        for item in data.get("grants", []):
            p = Path(item.get("path", ""))
            # Re-validate: the world may have changed since the grant was made.
            if p.is_dir() and not p.is_symlink() and self._permitted(p)[0]:
                self.grants[p.name.lower()] = Grant(p, bool(item.get("write")))
        if self.grants:
            self.bus.log(f"workspace: {len(self.grants)} folder(s) granted", "memory")

    def save(self) -> None:
        self.ltm.save_state("workspace",
                            {"grants": [g.to_json() for g in self.grants.values()]})

    # -- validation ---------------------------------------------------------
    def _permitted(self, path: Path) -> tuple[bool, str]:
        try:
            p = path.expanduser().resolve()
        except OSError as exc:
            return False, str(exc)
        if not p.is_dir():
            return False, "not a directory"
        if p in _FORBIDDEN_EXACT:
            return False, "too broad to grant — name a specific folder"
        parents = set(p.parents)
        for bad in _FORBIDDEN_TREES:
            if p == bad or bad in parents:
                return False, "that directory is off limits"
        home = Path.home().resolve()
        if home not in parents:
            return False, "outside your home directory"
        return True, ""

    # -- name resolution ----------------------------------------------------
    def find(self, name: str, limit: int = 8) -> list[Path]:
        """Locate directories by name under $HOME.

        Spoken and typed folder names rarely come with a path. Search is depth-
        limited and skips the usual dependency middens, so it stays interactive.
        """
        target = name.strip().strip("\"'").lower()
        if not target:
            return []
        direct = Path(target).expanduser()
        if direct.is_dir():
            return [direct.resolve()]

        hits: list[Path] = []
        home = Path.home()
        for root, dirs, _files in os.walk(home, topdown=True):
            depth = len(Path(root).relative_to(home).parts)
            if depth >= 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for d in dirs:
                if d.lower() == target or target in d.lower():
                    hits.append(Path(root) / d)
                    if len(hits) >= limit:
                        return hits
        return hits

    # -- grants -------------------------------------------------------------
    def grant(self, spec: str, write: bool = False) -> tuple[bool, str]:
        matches = self.find(spec)
        if not matches:
            return False, f"couldn't find a folder called {spec!r} under your home directory"
        if len(matches) > 1:
            want = spec.strip().strip("\"'").lower()
            exact = [m for m in matches if m.name.lower() == want] or matches
            if len(exact) > 1:
                # Prefer the shallowest — ~/voss over ~/DATABASE/x/y/voss. A name
                # sitting directly in $HOME is almost always the one meant, and
                # this is the difference between one clear answer and a six-line
                # "ambiguous" dump on a perfectly ordinary request.
                depth = min(len(m.parts) for m in exact)
                shallow = [m for m in exact if len(m.parts) == depth]
                if len(shallow) == 1:
                    exact = shallow
            if len(exact) == 1:
                matches = exact
            else:
                listing = "\n  ".join(str(m) for m in exact[:6])
                return False, (f"{spec!r} matches {len(exact)} folders — "
                               f"give me the full path:\n  {listing}")
        path = matches[0].resolve()
        ok, why = self._permitted(path)
        if not ok:
            return False, f"cannot grant {path}: {why}"
        self.grants[path.name.lower()] = Grant(path, write)
        self.save()
        mode = "read+write" if write else "read-only"
        return True, f"granted {mode}: {path}"

    def revoke(self, name: str) -> tuple[bool, str]:
        key = name.strip().strip("\"'").lower()
        if key in self.grants:
            path = self.grants.pop(key).path
            self.save()
            return True, f"revoked {path}"
        return False, f"no grant named {name!r}"

    def revoke_all(self) -> str:
        n = len(self.grants)
        self.grants.clear()
        self.save()
        return f"revoked {n} grant(s)"

    def get(self, name: str) -> Grant | None:
        return self.grants.get(name.strip().strip("\"'").lower())

    def allows(self, path: Path, write: bool = False) -> bool:
        try:
            p = Path(path).expanduser().resolve()
        except OSError:
            return False
        for g in self.grants.values():
            if p == g.path or g.path in p.parents:
                return g.write if write else True
        return False

    # -- inspection ---------------------------------------------------------
    def tree(self, name: str, depth: int = 3, limit: int = 200) -> tuple[bool, str]:
        g = self.get(name)
        if g is None:
            return False, f"no grant named {name!r} — grant it first"
        lines: list[str] = [str(g.path)]
        count = 0
        for root, dirs, files in os.walk(g.path, topdown=True):
            rel = Path(root).relative_to(g.path)
            level = len(rel.parts)
            if level >= depth:
                dirs[:] = []
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            if level:
                lines.append("  " * level + f"{rel.name}/")
            for f in sorted(files)[:12]:
                lines.append("  " * (level + 1) + f)
                count += 1
                if count >= limit:
                    lines.append("  ... (truncated)")
                    return True, "\n".join(lines)
        return True, "\n".join(lines)

    def structure(self, name: str) -> dict | None:
        """A shape summary used for comparing one folder against another."""
        g = self.get(name)
        if g is None:
            return None
        dirs: list[str] = []
        by_ext: dict[str, int] = {}
        files = 0
        for root, dnames, fnames in os.walk(g.path, topdown=True):
            dnames[:] = sorted(d for d in dnames if d not in _SKIP_DIRS)
            rel = Path(root).relative_to(g.path)
            if rel.parts:
                dirs.append(str(rel))
            for f in fnames:
                files += 1
                by_ext[Path(f).suffix.lower() or "(none)"] = \
                    by_ext.get(Path(f).suffix.lower() or "(none)", 0) + 1
        return {
            "name": g.path.name, "path": str(g.path),
            "dirs": sorted(dirs)[:60], "dir_count": len(dirs),
            "file_count": files,
            "extensions": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])[:12]),
            "writable": g.write,
        }

    def summary(self) -> str:
        if not self.grants:
            return "No folders granted."
        rows = []
        for g in self.grants.values():
            try:
                n = sum(1 for _ in g.path.rglob("*"))
            except OSError:
                n = -1
            mode = "rw" if g.write else "ro"
            rows.append(f"  [{mode}] {g.path}  ({n} entries)")
        return "Granted folders:\n" + "\n".join(rows)

    # -- mutation (always previewed first) ----------------------------------
    def plan_mirror(self, source: str, template: str) -> tuple[bool, str, list]:
        """Plan reorganising `source` to match `template`'s directory shape.

        Returns a preview only. Nothing moves until apply_mirror is called with
        explicit confirmation — a reorganisation is not reversible by voice.
        """
        src, tpl = self.get(source), self.get(template)
        if src is None or tpl is None:
            missing = source if src is None else template
            return False, f"no grant named {missing!r}", []
        if not src.write:
            return False, (f"{src.path.name} is granted read-only. "
                           f"Re-grant it with: grant --write {src.path.name}"), []

        tpl_dirs = [str(Path(r).relative_to(tpl.path))
                    for r, d, _ in os.walk(tpl.path)
                    if Path(r) != tpl.path]
        tpl_dirs = [d for d in tpl_dirs if not any(p in _SKIP_DIRS for p in Path(d).parts)]

        # Route each loose file in the source into the template folder whose name
        # best matches its extension or its own name.
        buckets: dict[str, list[str]] = {}
        for f in sorted(p for p in src.path.iterdir() if p.is_file()):
            ext = f.suffix.lower().lstrip(".")
            best = ""
            for d in tpl_dirs:
                leaf = Path(d).name.lower()
                if ext and (ext in leaf or leaf in ext):
                    best = d
                    break
                if leaf and leaf in f.stem.lower():
                    best = d
            if best:
                buckets.setdefault(best, []).append(f.name)

        moves = [(fn, d) for d, fns in buckets.items() for fn in fns]
        lines = [f"Plan: mirror {tpl.path.name}'s structure onto {src.path.name}",
                 f"  create {len(tpl_dirs)} director{'y' if len(tpl_dirs)==1 else 'ies'}"]
        for d in tpl_dirs[:20]:
            lines.append(f"    + {d}/")
        if len(tpl_dirs) > 20:
            lines.append(f"    ... and {len(tpl_dirs)-20} more")
        lines.append(f"  move {len(moves)} file(s)")
        for fn, d in moves[:20]:
            lines.append(f"    {fn}  ->  {d}/")
        if len(moves) > 20:
            lines.append(f"    ... and {len(moves)-20} more")
        if not moves:
            lines.append("    (no confident matches — directories only)")
        lines.append("\nType `apply` to run this, anything else to abandon it.")
        return True, "\n".join(lines), [(str(src.path), tpl_dirs, moves)]

    def apply_mirror(self, plan: list) -> tuple[bool, str]:
        if not plan:
            return False, "nothing planned"
        src_str, tpl_dirs, moves = plan[0]
        src = Path(src_str)
        if not self.allows(src, write=True):
            return False, "that folder is not granted for writing"
        made = moved = 0
        for d in tpl_dirs:
            try:
                (src / d).mkdir(parents=True, exist_ok=True)
                made += 1
            except OSError:
                continue
        for fn, d in moves:
            srcf, dstf = src / fn, src / d / fn
            try:
                if srcf.is_file() and not dstf.exists():
                    shutil.move(str(srcf), str(dstf))
                    moved += 1
            except OSError:
                continue
        return True, f"created {made} director(y/ies), moved {moved} file(s)"
