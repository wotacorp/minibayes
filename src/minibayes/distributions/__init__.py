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

"""Probability distributions for minibayes."""

from minibayes.distributions.base import Distribution, Support
from minibayes.distributions.bernoulli import Bernoulli
from minibayes.distributions.beta import Beta
from minibayes.distributions.cauchy import Cauchy
from minibayes.distributions.exponential import Exponential
from minibayes.distributions.gamma import Gamma
from minibayes.distributions.half_normal import HalfNormal
from minibayes.distributions.inverse_gamma import InverseGamma
from minibayes.distributions.laplace import Laplace
from minibayes.distributions.lkj_cholesky import LKJCholesky
from minibayes.distributions.lognormal import LogNormal
from minibayes.distributions.multivariate_normal import MultivariateNormal
from minibayes.distributions.normal import Normal
from minibayes.distributions.poisson import Poisson
from minibayes.distributions.student_t import StudentT
from minibayes.distributions.truncated_normal import TruncatedNormal
from minibayes.distributions.uniform import Uniform

__all__ = [
    # Base
    "Distribution",
    "Support",
    # Continuous - REAL support
    "Normal",
    "StudentT",
    "Cauchy",
    "Laplace",
    # Continuous - REAL support (multivariate)
    "MultivariateNormal",
    "LKJCholesky",
    # Continuous - POSITIVE support
    "HalfNormal",
    "Exponential",
    "Gamma",
    "LogNormal",
    "InverseGamma",
    # Continuous - UNIT support
    "Beta",
    # Continuous - BOUNDED support
    "Uniform",
    "TruncatedNormal",
    # Discrete
    "Bernoulli",
    "Poisson",
]
