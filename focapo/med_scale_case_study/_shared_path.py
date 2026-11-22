"""Where the shared modules (agents, forecasting, main, ...) live.

Single source of truth for the folder that used to be called
``admm_elz_market``. Med-scale files append this folder to ``sys.path``
*after* their own folder, so the med versions of ``miqp_admm.py`` and
``simulation_sc.py`` always win over the small-scale ones.
"""
from pathlib import Path

SHARED_DIR_NAME = 'small_scale_case_study'
SHARED_DIR = Path(__file__).resolve().parent.parent / SHARED_DIR_NAME
