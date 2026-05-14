"""Confidence prediction heads and per-sample score summaries.

置信度预测头与单样本得分汇总。

After the diffusion module produces atom coordinates, the confidence head
predicts how reliable each part of the structure is: pLDDT (per-atom),
PAE / PDE (per token-pair), pTM / iPTM (whole-complex), contact / clash
flags.

扩散模块产出坐标之后，置信度头预测每一部分结构的可靠度：pLDDT (每个原子)、
PAE / PDE (token 对)、pTM / iPTM (整复合物)、contact / clash 标志。

Files / 文件:
    confidence_head.py   ConfidenceHead   (Algorithm 31)
    distogram_head.py    DistogramHead    (Algorithm 1 line 14)
    bins.py              bin parameters, logits → prob / expected score
    clash.py             VDW clash + AF3 clash detectors
    scores.py            pTM / iPTM / gpde / chain-pair-PAE / chain-PLDDT
    summary.py           compute_full_data_and_summary
    external_clash.py    Clash nn.Module helper
"""
from confidence.bins import (  # noqa: F401
    calculate_normalization,
    compute_contact_prob,
    get_bin_centers,
    get_bin_params,
    logits_to_prob,
    logits_to_score,
)
from confidence.clash import calculate_clash, calculate_vdw_clash  # noqa: F401
from confidence.confidence_head import ConfidenceHead  # noqa: F401
from confidence.distogram_head import DistogramHead  # noqa: F401
from confidence.external_clash import Clash  # noqa: F401
from confidence.scores import (  # noqa: F401
    calculate_chain_based_gpde,
    calculate_chain_based_plddt,
    calculate_chain_based_ptm,
    calculate_chain_pair_pae,
    calculate_iptm,
    calculate_ptm,
)
from confidence.summary import (  # noqa: F401
    break_down_to_per_sample_dict,
    compute_full_data_and_summary,
    merge_per_sample_confidence_scores,
    traverse_and_aggregate,
)
