# DeepEP binding patch

`deepep-v1.2.1-zero-copy-layout.patch` applies to DeepEP commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` and changes only `csrc/deep_ep.cpp`. It adds guarded support for passing the registered, padded combine view directly while preserving the upstream contiguous-placeholder convention.

The underlying DeepEP source is Copyright (c) 2025 DeepSeek and distributed under the [MIT license](DEEPEP-LICENSE). The pinned upstream code and its license remain authoritative for that dependency.

The regression runner is `scripts/validate_zero_copy_contract.py`. It checks rejected views, buffer-index preservation and the existing placeholder convention. The setup script checks whether the patch is already applied before building.
