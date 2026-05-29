# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

"""Import utilities for checking package availability."""

import importlib
import shutil
import sys


def module_available(module_name: str) -> bool:
    """Check if a Python module is available."""
    try:
        # Handle version specs like "package>=1.0"
        pkg = module_name.split(">=")[0].split("==")[0].split("<")[0].strip()
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


def module_available_version(module_name: str) -> bool:
    """Check if a module is available and meets version requirement.

    For simple package names, just checks availability.
    For versioned specs like 'package>=1.0', checks version too.
    """
    if (
        ">=" not in module_name
        and "==" not in module_name
        and "<" not in module_name
    ):
        return module_available(module_name)

    # Parse package name and version
    import packaging.version

    for op in [">=", "==", "<", "<=", ">", "!="]:
        if op in module_name:
            pkg, ver = module_name.split(op, 1)
            pkg = pkg.strip()
            ver = ver.strip()
            try:
                mod = importlib.import_module(pkg)
                installed_ver = getattr(mod, "__version__", "0.0.0")
                installed = packaging.version.parse(installed_ver)
                required = packaging.version.parse(ver)
                if op == ">=":
                    return installed >= required
                elif op == "==":
                    return installed == required
                elif op == "<":
                    return installed < required
                elif op == "<=":
                    return installed <= required
                elif op == ">":
                    return installed > required
                elif op == "!=":
                    return installed != required
            except Exception:
                return False
    return False


_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Core availability checks
_NLTK_AVAILABLE = module_available("nltk")
_ROUGE_SCORE_AVAILABLE = module_available("rouge_score")
_BERTSCORE_AVAILABLE = module_available("bert_score")
_SCIPY_AVAILABLE = module_available("scipy")
_SCIPY_GREATER_EQUAL_1_8 = module_available_version("scipy>=1.8.0")
_SCIPY_AVAILABLE = _SCIPY_AVAILABLE  # Typo compatibility alias
_PYCOCOTOOLS_AVAILABLE = module_available("pycocotools")
_PYCOCOTOOLS_GREATER_EQUAL_2_0_9 = module_available_version(
    "pycocotools>=2.0.9"
)
_TORCHVISION_AVAILABLE = False  # Not available in Paddle ecosystem
_TORCH_FIDELITY_AVAILABLE = False  # PyTorch-specific
_LPIPS_AVAILABLE = False  # PyTorch-specific
_TQDM_AVAILABLE = module_available("tqdm")
# transformers triggers torch import which may crash on some systems (DLL issues)
_TRANSFORMERS_AVAILABLE = False
_TRANSFORMERS_GREATER_EQUAL_4_4 = False
_TRANSFORMERS_GREATER_EQUAL_4_10 = False
_PESQ_AVAILABLE = module_available("pesq")
_GAMMATONE_AVAILABLE = module_available("gammatone")
_TORCHAUDIO_AVAILABLE = False  # PyTorch-specific
_REGEX_AVAILABLE = module_available("regex")
_PYSTOI_AVAILABLE = module_available("pystoi")
_REQUESTS_AVAILABLE = module_available("requests")
_LIBROSA_AVAILABLE = module_available("librosa")
_ONNXRUNTIME_AVAILABLE = module_available("onnxruntime")
_FAST_BSS_EVAL_AVAILABLE = module_available("fast_bss_eval")
_MATPLOTLIB_AVAILABLE = module_available("matplotlib")
_SCIENCEPLOT_AVAILABLE = module_available("scienceplots")
_MULTIPROCESSING_AVAILABLE = module_available("multiprocessing")
_XLA_AVAILABLE = False  # Not applicable to Paddle
_PIQ_GREATER_EQUAL_0_8 = False  # PyTorch-specific
_FASTER_COCO_EVAL_AVAILABLE = module_available("faster_coco_eval")
_MECAB_AVAILABLE = module_available("MeCab")
_MECAB_KO_AVAILABLE = module_available("mecab_ko")
_MECAB_KO_DIC_AVAILABLE = module_available("mecab_ko_dic")
_IPADIC_AVAILABLE = module_available("ipadic")
_SENTENCEPIECE_AVAILABLE = module_available("sentencepiece")
_TORCH_LINEAR_ASSIGNMENT_AVAILABLE = False  # PyTorch-specific
_AEON_AVAILABLE = module_available("aeon")
_PYTDC_AVAILABLE = module_available("pyTDC")
_TORCH_VMAF_AVAILABLE = False  # PyTorch-specific
_EINOPS_AVAILABLE = module_available("einops")
_SKLEARN_AVAILABLE = module_available("sklearn")
_SKIMAGE_AVAILABLE = module_available("skimage")
_LATEX_AVAILABLE: bool = shutil.which("latex") is not None
