# state_of_neuro/scripts/term_normalization.py
"""Normalize terms and expand recorded singular/plural decisions.

Recovered from bozelosp/state-of-neuroscience at 61358677e2ce862dd465c99c61541782d8549a70.
Source: build_authoritative_clusters.py.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List

_DASH = re.compile(r"[‒–—−]")

_APOS = re.compile(r"[’´`]")

_WS   = re.compile(r"\s+")

_NONA = re.compile(r"[^0-9a-z]+")

def ntext(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = _DASH.sub("-", s)
    s = _APOS.sub("'", s)
    s = s.replace("'", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s

def nkey(s: str) -> str:
    return _NONA.sub("", ntext(s))

def unique_preserve_order(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in seq:
        k=nkey(x or "")
        if not k: continue
        if k not in seen:
            out.append(x); seen.add(k)
    return out

def read_json(path:str)->Any:
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

class SP:
    def __init__(self):
        self.to_sing: Dict[str,str] = {}
        self.to_plur: Dict[str,str] = {}
    def load(self, path: str):
        if not path or not os.path.exists(path): return
        data = read_json(path)
        items = []
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                items = data["items"]
            else:
                # allow dict-of-dicts {tag:{singular,plural}, ...}
                items = [{"tag":k, **(v if isinstance(v,dict) else {})} for k,v in data.items()]
        elif isinstance(data, list):
            items = data
        for it in items:
            tag  = ntext(str(it.get("tag","")))
            sing = ntext(str(it.get("singular") or it.get("tag","")))
            plur = ntext(str(it.get("plural") or ""))
            if tag:
                if sing: self.to_sing[tag]=sing
                if plur: self.to_plur[tag]=plur
            if sing and plur:
                self.to_sing[plur]=sing
                self.to_plur[sing]=plur
    def expand(self, token: str) -> List[str]:
        b = ntext(token)
        out=[]
        if b in self.to_sing: out.append(self.to_sing[b])
        if b in self.to_plur: out.append(self.to_plur[b])
        # de-dup, keep order
        u=[]; seen=set()
        for x in out:
            k=nkey(x)
            if k and k not in seen:
                u.append(x); seen.add(k)
        return u
