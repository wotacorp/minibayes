# Copyright 2026 WOTA CORP.
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

"""Custom exceptions for minibayes."""


class MinibayesError(Exception):
    """Base exception for minibayes."""


class SamplingError(MinibayesError):
    """Raised when sampling fails."""


class SamplingTimeoutError(SamplingError):
    """Raised when sampling exceeds the specified timeout."""


class ConvergenceWarning(UserWarning):
    """Raised when diagnostics suggest non-convergence."""


class NumericalError(MinibayesError):
    """Raised on numerical issues (NaN, Inf in log_prob)."""


class ModelSpecError(MinibayesError):
    """Raised when model is mis-specified."""
