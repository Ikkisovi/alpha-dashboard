import json
import math
import re
from typing import Dict, List, Optional

from alphagen.data.expression import Expression
from alphagen.data.expression import OutOfDataRangeError
from alphagen.utils.correlation import batch_pearsonr
from src.alpha_gfn.alpha_pool import AlphaPoolGFN
from src.alphagen.data.tree import ExpressionParser
from src.strategy.online_learning import AlphaTemplate, RollingTuner


def load_template_config(path: str) -> List[Dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    configs: List[Dict] = []
    for cfg in payload:
        cfg_copy = dict(cfg)
        param_ranges = cfg_copy.get("param_ranges") or []
        cfg_copy["param_ranges"] = [tuple(p) for p in param_ranges]
        configs.append(cfg_copy)
    return configs


class TemplateTuner:
    def __init__(self, pool: AlphaPoolGFN, use_abs_ic: bool = True):
        self.pool = pool
        self.parser = ExpressionParser()
        self.use_abs_ic = use_abs_ic

    def _score_expr(self, expr_str: str) -> float:
        try:
            expr = self.parser.parse(expr_str)
            value = self.pool._normalize_by_day(expr.evaluate(self.pool.data))
            ic = batch_pearsonr(value, self.pool.target).mean().item()
        except (OutOfDataRangeError, Exception):
            return 0.0
        if math.isnan(ic):
            return 0.0
        return abs(ic) if self.use_abs_ic else ic

    def _extract_seed_params(self, template_str: str, expr_str: str, n_params: int) -> Optional[List[float]]:
        pattern = re.escape(template_str)
        for idx in range(n_params):
            pattern = pattern.replace(r"\{" + str(idx) + r"\}", rf"(?P<p{idx}>[-+]?[\deE\.]+)")
        match = re.match(pattern + r"$", expr_str)
        if not match:
            return None
        params = []
        for idx in range(n_params):
            try:
                params.append(float(match.group(f"p{idx}")))
            except Exception:
                return None
        return params

    def tune_templates(
        self,
        template_cfgs: List[Dict],
        popsize: int = 6,
        maxiter: int = 6,
        mutation: float = 0.6,
        recombination: float = 0.7,
        resume_exprs: Optional[List[str]] = None
    ) -> List[Dict]:
        tuned: List[Dict] = []
        resume_exprs = resume_exprs or []

        for cfg in template_cfgs:
            cfg_type = cfg.get("type", "fixed")
            name = cfg.get("name", cfg.get("expr", "template"))

            if cfg_type == "fixed":
                expr_str = cfg["expr"]
                score = self._score_expr(expr_str)
                tuned.append({
                    "name": name,
                    "expr": expr_str,
                    "params": [],
                    "score": float(score),
                    "source": "fixed"
                })
                continue

            template = AlphaTemplate(
                cfg["template"],
                cfg["param_ranges"],
                param_types=cfg.get("param_types")
            )
            seed_params = cfg.get("seed_params")
            if seed_params is None:
                for expr_str in resume_exprs:
                    seed_params = self._extract_seed_params(template.template_str, expr_str, len(cfg["param_ranges"]))
                    if seed_params is not None:
                        break

            tuner = RollingTuner(
                evaluate_fn=lambda expr, _: self._score_expr(expr),
                popsize=popsize,
                maxiter=maxiter,
                mutation=(mutation, 1.0),
                recombination=recombination
            )

            best_expr, best_params, best_score = tuner.tune(template, history_data=None, prev_params=seed_params)
            tuned.append({
                "name": name,
                "expr": best_expr,
                "params": [float(p) for p in best_params],
                "score": float(best_score),
                "source": "tuned"
            })
        return tuned

    def seed_pool(self, tuned_results: List[Dict]) -> List[Dict]:
        added: List[Dict] = []
        for res in tuned_results:
            try:
                expr = self.parser.parse(res["expr"])
            except Exception:
                continue
            ic_reward, nov_reward, ssl_reward = self.pool.try_new_expr_with_ssl(expr, embedding=None)
            added.append({
                "name": res.get("name", res.get("expr")),
                "expr": res["expr"],
                "ic_reward": float(ic_reward),
                "nov_reward": float(nov_reward),
                "ssl_reward": float(ssl_reward),
            })
        if added:
            self.pool.recompute_weights()
        return added
