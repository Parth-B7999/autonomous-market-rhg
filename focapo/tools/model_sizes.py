"""Print Gurobi model sizes for the largest models (license check, Phase A4).

The pip-bundled restricted gurobipy license allows at most 2000 variables and
2000 linear constraints per model. Run: python tools/model_sizes.py
"""
import sys, types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'med_scale_case_study'))
import gurobipy as gp                      # noqa: E402
import miqp_admm                           # noqa: E402
from agents_med import build_med_fleet     # noqa: E402
from agents import H_RT, H_DA, DT_RT, DT_DA  # noqa: E402

SIZES = []


class _Recorder(gp.Model):
    def optimize(self, *a, **k):
        self.update()
        SIZES.append((LABEL, self.NumVars, self.NumIntVars, self.NumConstrs,
                      self.NumQConstrs, self.NumGenConstrs, self.NumSOS))
        return super().optimize(*a, **k)


shim = types.SimpleNamespace(**{n: getattr(gp, n) for n in dir(gp) if not n.startswith('__')})
shim.Model = _Recorder
miqp_admm.gp = shim

LIMIT = 2000


def main() -> None:
    global LABEL
    fleet = build_med_fleet()
    for ag in fleet:
        ag.h2_target_kg_day = 0.5 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24
    N = len(fleet)
    rng = np.random.default_rng(0)
    lam24, lam12 = rng.uniform(20, 80, H_DA), rng.uniform(20, 80, H_RT)
    soc = np.array([0.5 * ag.bat_cap_kwh for ag in fleet])

    for i, ag in enumerate(fleet):
        LABEL = f'_solve_agent DAM H=24  {ag.name}'
        miqp_admm._solve_agent(ag, H_DA, DT_DA, soc[i], lam24, 0.0, np.full(H_DA, 50.0),
                               1e-4, np.zeros(H_DA), np.zeros(H_DA), None, 0.0, True, 100.0)
    for i, ag in enumerate(fleet):
        LABEL = f'_solve_agent RTM H=12  {ag.name}'
        miqp_admm._solve_agent(ag, H_RT, DT_RT, soc[i], lam12, 150.0, np.full(H_RT, 50.0),
                               0.05, np.zeros(H_RT), np.zeros(H_RT), 1, ag.gamma, False, 100.0,
                               h2_target_rtm=np.zeros(H_RT), l_min_dev=100.0)
    LABEL = f'commit_u_gne H=12 N={N}'
    miqp_admm.commit_u_gne(fleet, lam12, np.full((N, H_RT), 50.0), H_RT, DT_RT, soc,
                           100.0, float('inf'), h2_target_rtm=np.zeros((N, H_RT)))

    print(f"{'model':42s} {'vars':>6s} {'int':>5s} {'lin':>6s} {'quad':>5s} {'gen':>4s} {'sos':>4s}")
    for s in SIZES:
        flag = '  OVER LIMIT' if s[1] > LIMIT or s[3] > LIMIT else ''
        print(f"{s[0]:42s} {s[1]:6d} {s[2]:5d} {s[3]:6d} {s[4]:5d} {s[5]:4d} {s[6]:4d}{flag}")
    print(f"max vars {max(s[1] for s in SIZES)}, max linear constrs {max(s[3] for s in SIZES)} (limit {LIMIT})")


if __name__ == '__main__':
    main()
