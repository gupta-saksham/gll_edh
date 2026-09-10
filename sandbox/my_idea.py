# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Submission adapter for the reusable controller and stress tariff.

Change TARIFF_PARAMS to test a tariff; score() searches the same complete policy
bank each time. The default controller uses local voltage-trend feedback and
persistent staggering. See CONTROLLER_FRAMEWORK_PLAN.md and the experiments
notebook for independent train/validation/test comparisons.
"""

from sandbox.controller_family import TUNE_OVER as TUNE_OVER
from sandbox.controller_family import family_policy, init_family_memory
from sandbox.tariff_family import DEFAULT_TARIFF_PARAMS, stress_tariff

CONTROLLER_PARAMS = {"policy_id": 15.0, "voltage_enabled": 1.0}
INIT_CARRY = init_family_memory
TARIFF_PARAMS = {
    **DEFAULT_TARIFF_PARAMS,
    "export_threshold_kw": 45.0,
    "export_strength_chf_per_kwh": 0.30,
}


def my_controller(obs, carry, params, key):
    """One household: return active inverter kW and fixed-shape local memory."""
    return family_policy(obs, carry, params, key)


def my_tariff(grid, carry, params):
    """All connection points: fair LEG plus directional stress redistribution."""
    return stress_tariff(grid, carry, params)


def check(days: int = 1, detail: bool = False, fast: bool = True) -> None:
    """Check fixed behaviour; use score() to include tariff-driven re-tuning."""
    from sandbox.check import run_check

    run_check(days=days, detail=detail, fast=fast)


def score(detail: bool = True) -> None:
    """Run the unchanged official four-cell evaluation."""
    from sandbox.check import run_score

    run_score(detail=detail)
