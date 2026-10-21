# -*- coding: utf-8 -*-
"""v13: class + data-source configuration (see class_config.json).

Pure Python (json/fnmatch only) so it can be imported by common.py and by
scripts that never import torch. Everything that used to be hard-coded in
common.py (the class lists, the raw-label maps) and everything new in v13
(per-source class scope, split roles, "uncertain" labels) comes from here.

Per-class "annotated_in" (optional, list of source globs; default = every
source): where that class was actually annotated. This is what makes adding
a class later safe -- a new class that only exists in ONE new dataset gets
annotated_in=["thatdataset*"], and every OTHER source (Gerlache, SOI,
bluefin, DCLDE...) is automatically masked for it instead of being treated as
"all background".
"""
import fnmatch
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_config.json")
_VALID_BANDS = ("lf", "hf")
_VALID_SPLITS = ("random", "train_only", "val_only", "ignore")


class ClassConfig:
    def __init__(self, d: dict, origin: str = "<dict>"):
        self.raw = d
        self.origin = origin
        classes = d.get("classes") or []
        if not classes:
            raise ValueError(f"{origin}: 'classes' is empty")
        names = [c["name"] for c in classes]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{origin}: duplicate class names {dup}")
        for c in classes:
            if c.get("band") not in _VALID_BANDS:
                raise ValueError(f"{origin}: class {c['name']!r} needs band in {_VALID_BANDS}, got {c.get('band')!r}")
        self.classes = classes
        self.lf_labels = [c["name"] for c in classes if c["band"] == "lf"]
        self.hf_labels = [c["name"] for c in classes if c["band"] == "hf"]
        self.joint_labels = self.lf_labels + self.hf_labels

        self.aplose_map, self.soi_map = {}, {}
        for c in classes:
            for raw in c.get("aplose_labels", []):
                self._put(self.aplose_map, raw, c["name"], "aplose_labels")
            for raw in c.get("soi_labels", []):
                self._put(self.soi_map, raw, c["name"], "soi_labels")
        self.uncertain_map = dict(d.get("uncertain_labels", {}))
        for raw, cls in self.uncertain_map.items():
            if cls not in names:
                raise ValueError(f"{origin}: uncertain_labels[{raw!r}] -> unknown class {cls!r}")
            if raw in self.aplose_map:
                raise ValueError(f"{origin}: {raw!r} is both a positive label and an uncertain label")

        self.stem_rules = [(r["contains"], r["source"]) for r in d.get("stem_source_rules", [])]
        self.sources = d.get("sources", {})
        self.defaults = {"annotated_classes": "all", "split": "random", "sample_weight": 1.0,
                         "val_only_classes": []}
        self.defaults.update(d.get("source_defaults", {}))
        for pat, sc in list(self.sources.items()) + [("source_defaults", self.defaults)]:
            ac = sc.get("annotated_classes", "all")
            if ac != "all":
                bad = [x for x in ac if x not in names]
                if bad:
                    raise ValueError(f"{origin}: sources[{pat!r}].annotated_classes has unknown classes {bad}")
            if sc.get("split", "random") not in _VALID_SPLITS:
                raise ValueError(f"{origin}: sources[{pat!r}].split must be one of {_VALID_SPLITS}")
            bad = [x for x in sc.get("val_only_classes", []) if x not in names]
            if bad:
                raise ValueError(f"{origin}: sources[{pat!r}].val_only_classes has unknown classes {bad}")
        self._annotated_in = {c["name"]: c.get("annotated_in") for c in classes}

    @staticmethod
    def _put(m, raw, cls, field):
        if raw in m and m[raw] != cls:
            raise ValueError(f"raw label {raw!r} in {field} maps to both {m[raw]!r} and {cls!r}")
        m[raw] = cls

    # ----------------------------------------------------------------- sources
    def source_of(self, stem: str, path: str = None) -> str:
        for sub, src in self.stem_rules:
            if sub in stem:
                return src
        if path:
            return os.path.basename(os.path.dirname(os.path.abspath(path))) or "unknown"
        return "unknown"

    def source_cfg(self, source: str) -> dict:
        out = dict(self.defaults)
        for pat, sc in self.sources.items():
            if fnmatch.fnmatchcase(source, pat):
                out.update(sc)
                break
        return out

    def scope(self, source: str, label_list) -> list:
        """bool per class in label_list: was this class annotated in this source?"""
        sc = self.source_cfg(source)
        ac = sc.get("annotated_classes", "all")
        res = []
        for lb in label_list:
            ok = ac == "all" or lb in ac
            ai = self._annotated_in.get(lb)
            if ok and ai is not None:
                ok = any(fnmatch.fnmatchcase(source, p) for p in ai)
            res.append(bool(ok))
        return res

    def val_only_classes(self, source: str) -> set:
        return set(self.source_cfg(source).get("val_only_classes", []))

    def split_role(self, source: str) -> str:
        return self.source_cfg(source).get("split", "random")

    def sample_weight(self, source: str) -> float:
        return float(self.source_cfg(source).get("sample_weight", 1.0))


def load_class_config(path_or_dict=None) -> ClassConfig:
    if isinstance(path_or_dict, dict):
        return ClassConfig(path_or_dict, origin="<dict>")
    path = path_or_dict or os.environ.get("SOMAD_CLASS_CONFIG") or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return ClassConfig(json.load(f), origin=path)
