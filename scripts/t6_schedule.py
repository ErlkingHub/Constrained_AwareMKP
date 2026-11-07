"""T6 activated-theta schedules. No model module from either project is imported."""
from collections import defaultdict

import numpy as np
import torch
from torch import nn

CLOCKS = ("tau_step", "tau_cap", "tau_feas")
SCALARS = ("profit_exponent", "scarcity_exponent", "bottleneck_weight", "temperature",
           "opportunity_weight", "reserve_weight", "lp_primal_weight", "lp_dual_weight")
DYNAMIC = (*SCALARS, "companion_weights")
MINIMAL = ("profit_exponent", "scarcity_exponent", "lp_primal_weight", "companion_weights")
ARMS = {"S1": ("category", "tau_step"), "S2": ("category", "tau_cap"),
        "S3": ("M", "tau_cap"), "S4": ("category", "tau_cap"), "S5": ("category", "tau_cap")}


def clocks(state):
    """Pre-action, deployable clocks; no final episode length or future state."""
    n = state.profits.shape[1]
    values = {"tau_step": state.selected.sum(1).double() / n,
              "tau_cap": 1 - (state.remaining / state.capacities).mean(1),
              "tau_feas": 1 - state.feasible.sum(1).double() / n}
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()) or bool(((value < -1e-12) | (value > 1+1e-12)).any()):
            raise ValueError(f"Clock {name} is not finite in [0,1]")
        values[name] = value.clamp(0, 1)
    return values


def key(record, grouping):
    return str(record["M"]) if grouping == "M" else f"{record['M']}|{record['N']}|{record['tightness']}"


def interpolate(nodes, values, query):
    """Linear interpolation, exact knots and endpoint clamping; output value dtype."""
    nodes = torch.as_tensor(nodes, dtype=torch.float64, device=query.device)
    values = torch.as_tensor(values, device=query.device)
    query = query.double().reshape(-1)
    if (nodes.ndim != 1 or len(nodes) != values.shape[0] or len(nodes) < 1 or
            not bool(torch.isfinite(nodes).all()) or not bool(torch.isfinite(values).all()) or
            not bool(torch.isfinite(query).all()) or bool((nodes[1:] <= nodes[:-1]).any())):
        raise ValueError("Interpolation requires finite, strictly increasing knots and matching values")
    if len(nodes) == 1:
        return values[0].expand(len(query), *values.shape[1:])
    lo = (torch.searchsorted(nodes, query.contiguous(), right=True) - 1).clamp(0, len(nodes)-2)
    weight = ((query-nodes[lo]) / (nodes[lo+1]-nodes[lo])).to(values.dtype)
    shape = (len(query),) + (1,)*(values.ndim-1)
    result = values[lo] + weight.reshape(shape) * (values[lo+1]-values[lo])
    result = torch.where((query <= nodes[0]).reshape(shape), values[0], result)
    return torch.where((query >= nodes[-1]).reshape(shape), values[-1], result)


def mean_nonprice(records, cfg, helper):
    # Never average resource prices by resource index, including temporarily.
    first = records[0]["theta"]
    result = {name: torch.stack([torch.as_tensor(r["theta"][name]).float() for r in records]).double().mean(0).float()
              for name in helper.THETA_FIELDS if name != "resource_prices"}
    result["resource_prices"] = torch.ones_like(torch.as_tensor(first["resource_prices"]), dtype=torch.float32)
    a = result["resource_interactions"]
    result["resource_interactions"] = (a+a.transpose(-1, -2))/2
    helper.validate_theta(result, cfg)
    return result


def fit_tables(records, grouping, clock, configs, helper):
    groups = defaultdict(list)
    for record in records:
        if record["dataset"] != "synthetic_trace":
            raise ValueError("Schedules can only fit the independent synthetic trace set")
        groups[key(record, grouping)].append(record)
    tables = {}
    for group, members in groups.items():
        cfg = configs[members[0]["M"]]
        x = np.asarray([r[clock] for r in members], dtype=np.float64)
        if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("Invalid fitting clock")
        edges = np.quantile(x, np.linspace(0, 1, 11))
        bins = np.clip(np.searchsorted(edges, x, side="right")-1, 0, 9)
        knots, values, counts = [], [], []
        for index in range(10):
            mask = bins == index
            if not mask.any():
                continue
            subset = [r for r, keep in zip(members, mask) if keep]
            position = float(np.median(x[mask]))
            value = mean_nonprice(subset, cfg, helper)
            if knots and position == knots[-1]:
                # Tied clock values must not produce zero-width interpolation.
                total = counts[-1] + len(subset)
                values[-1] = {n: (values[-1][n]*counts[-1]+value[n]*len(subset))/total for n in DYNAMIC}
                counts[-1] = total
            else:
                knots.append(position); values.append({n: value[n] for n in DYNAMIC}); counts.append(len(subset))
        base = mean_nonprice(members, cfg, helper)
        table = {"grouping": grouping, "clock": clock, "group": group, "M": members[0]["M"],
                 "knots": knots, "bin_counts": counts, "decision_count": len(members),
                 "base": {n: v.tolist() for n, v in base.items()},
                 "values": {n: torch.cat([v[n] for v in values], 0).tolist() for n in DYNAMIC}}
        # Convex interpolation preserves the checked scalar/simplex constraints.
        for i in range(len(knots)):
            theta = {n: v.clone() for n, v in base.items()}
            for n in DYNAMIC:
                theta[n] = torch.as_tensor(table["values"][n], dtype=torch.float32)[i:i+1]
            helper.validate_theta(theta, cfg)
        tables[group] = table
    return tables


def constant_table(theta, clock="tau_cap"):
    """Regression fixture: preserve the entire supplied G, including its prices."""
    return {"clock": clock, "base": {n: torch.as_tensor(v).tolist() for n, v in theta.items()},
            "knots": [0., 1.], "values": {n: torch.cat([theta[n], theta[n]], 0).tolist() for n in DYNAMIC}}


class ScheduleActor(nn.Module):
    def __init__(self, table, theta_class, cfg, helper, *, base=None, dynamic=DYNAMIC, price_actor=None):
        super().__init__()
        self.theta_class, self.cfg, self.helper = theta_class, cfg, helper
        self.clock, self.dynamic = table["clock"], tuple(dynamic)
        self.price_actor = price_actor
        self.register_buffer("nodes", torch.as_tensor(table["knots"], dtype=torch.float64))
        source = table["base"] if base is None else base
        for name in helper.THETA_FIELDS:
            self.register_buffer("base_"+name, torch.as_tensor(source[name], dtype=torch.float32))
        for name in self.dynamic:
            self.register_buffer("values_"+name, torch.as_tensor(table["values"][name], dtype=torch.float32))
        helper.validate_theta(source, cfg)
        self.previous = None
        self.decisions = 0
        self.clock_bounds = {n: [1., 0.] for n in CLOCKS}

    def forward(self, state):
        current = clocks(state)
        if bool((state.selected.sum(1) == 0).all()):
            self.previous = None
        if self.previous is not None:
            for name in CLOCKS:
                if bool((current[name] < self.previous[name]-1e-12).any()):
                    raise ValueError(f"Nonmonotone clock: {name}")
        self.previous = {n: v.detach().clone() for n, v in current.items()}
        for name, value in current.items():
            self.clock_bounds[name][0] = min(self.clock_bounds[name][0], float(value.min()))
            self.clock_bounds[name][1] = max(self.clock_bounds[name][1], float(value.max()))
        batch, n = state.profits.shape
        theta = {name: getattr(self, "base_"+name).expand(batch, *getattr(self, "base_"+name).shape[1:])
                 for name in self.helper.THETA_FIELDS}
        for name in self.dynamic:
            theta[name] = interpolate(self.nodes, getattr(self, "values_"+name), current[self.clock])
        if self.price_actor is not None:
            live, _ = self.price_actor(state)
            theta["resource_prices"] = live.resource_prices
        for row in range(batch):
            self.helper.validate_theta({name: value[row:row+1] for name, value in theta.items()}, self.cfg)
        self.decisions += 1
        return self.theta_class(**theta), state.profits.new_zeros((batch, n), dtype=torch.float32)


def crossfit(records, configs, helper):
    """Split complete trajectories within each category; never split decisions."""
    categories, assignments = defaultdict(set), {}
    for r in records:
        categories[key(r, "category")].add(r["id"])
    for category, ids in categories.items():
        if len(ids) < 2:
            raise ValueError("Two-fold diagnostics need at least two trajectories per category")
        for i, iid in enumerate(sorted(ids)):
            assignments[iid] = i % 2
    outputs = []
    for grouping in ("category", "M"):
        for clock in CLOCKS:
            scored = defaultdict(list)
            for held_out in (0, 1):
                training = [r for r in records if assignments[r["id"]] != held_out]
                validation = [r for r in records if assignments[r["id"]] == held_out]
                tables = fit_tables(training, grouping, clock, configs, helper)
                validation_groups = defaultdict(list)
                for r in validation:
                    validation_groups[key(r, grouping)].append(r)
                for group, members in validation_groups.items():
                    table = tables[group]
                    q = torch.tensor([r[clock] for r in members], dtype=torch.float64)
                    predicted = {n: interpolate(table["knots"], torch.tensor(table["values"][n], dtype=torch.float32), q).numpy().reshape(len(members),-1)
                                 for n in DYNAMIC}
                    for i,r in enumerate(members):
                        observed = {n: np.asarray(r["theta"][n], dtype=np.float64).reshape(-1) for n in DYNAMIC}
                        for name in SCALARS:
                            scored[name].append((r["id"], float(observed[name][0]), float(predicted[name][i,0]), held_out))
                        for j in range(len(observed["companion_weights"])):
                            scored[f"companion_{j}"].append((r["id"], observed["companion_weights"][j], predicted["companion_weights"][i,j], held_out))
                        scored["companion_sum"].append((r["id"], float(observed["companion_weights"].sum()),
                                                       float(predicted["companion_weights"][i].sum()), held_out))
            for field, entries in scored.items():
                y = np.asarray([r[1] for r in entries]); p = np.asarray([r[2] for r in entries])
                by_instance = defaultdict(list)
                for i, (iid, _, _, _) in enumerate(entries):
                    by_instance[iid].append(i)
                within_sse = within_ss = 0.
                for indices in by_instance.values():
                    a, b = y[indices], p[indices]
                    within_sse += float(((a-a.mean() - (b-b.mean()))**2).sum())
                    within_ss += float(((a-a.mean())**2).sum())
                raw_ss = float(((y-y.mean())**2).sum()); raw_sse = float(((y-p)**2).sum())
                outputs.append({"grouping": grouping, "clock": clock, "field": field, "decisions": len(y),
                    "instances": len(by_instance), "raw_sse": raw_sse, "raw_sst": raw_ss,
                    "raw_R2": None if raw_ss == 0 else 1-raw_sse/raw_ss,
                    "within_sse": within_sse, "within_sst": within_ss,
                    "within_trajectory_R2": None if within_ss == 0 else 1-within_sse/within_ss,
                    "fold_rule": "sorted instance IDs alternate within category; complete held-out trajectories"})
    return outputs
