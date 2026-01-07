┌─────────────────────────────────────────────────────────────────────────┐
│                           FOUNDATION (Week 1)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   utils.py ──────────┬──────────────► transforms.py                     │
│   (numerical)        │                (Identity, Log, Logit)            │
│                      │                       │                          │
│                      ▼                       ▼                          │
│              distributions/base.py ◄─────────┘                          │
│              (Support enum, ABC)                                        │
│                      │                                                  │
│                      ▼                                                  │
│              distributions/continuous.py                                │
│              (Normal, HalfNormal, Beta, Gamma, Uniform, Exponential)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE (Week 2)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   model.py ◄──────── distributions + transforms                         │
│   (Model class: priors, likelihood, log_prob, transforms)               │
│                      │                                                  │
│                      ▼                                                  │
│   samplers/base.py ──► samplers/mh.py ──► samplers/adaptive.py          │
│   (Sampler ABC)        (basic MH)         (Adaptive MH)                 │
│                                                                         │
│   results.py ◄─────── samplers                                          │
│   (InferenceResult dataclass)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTERFACE (Week 3)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   __init__.py ◄────── model + samplers + results                        │
│   (mb.sample entry point, public API)                                   │
│                                                                         │
│   diagnostics.py ◄─── results                                           │
│   (ESS, R-hat, summary)                                                 │
│                                                                         │
│   export.py ◄──────── results                                           │
│   (save/load npz, json)                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
