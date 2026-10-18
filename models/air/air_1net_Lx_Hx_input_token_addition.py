"""Symmetric Lx_Hx AIR with an element-wise learned level embedding.

This is the one-copy counterpart of
``air_1net_L2x_H2x_input_token_addition``.  Each update receives exactly one
copy of the encoded input, matching the symmetric ``Lx_Hx`` control, while a
learned L/H identity embedding is added to the state being updated.
"""

from typing import Dict, Tuple

import torch
from torch import nn

from models.air.air_1net_L2x_H2x_input_token_addition import (
    AIR as _BaseAIR,
    AIRConfig,
    AIRInnerCarry,
    AIR_Inner as _BaseAIRInner,
)


class AIR_Inner(_BaseAIRInner):
    def forward(
        self,
        carry: AIRInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[AIRInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = {
            "cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        }
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
        )

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for H_step in range(self.config.H_cycles):
                for L_step in range(self.config.L_cycles):
                    is_final_L = (
                        H_step == self.config.H_cycles - 1
                        and L_step == self.config.L_cycles - 1
                    )
                    if not is_final_L:
                        z_L = self.f_level(
                            z_L + self.input_token_L,
                            z_H + input_embeddings,
                            **seq_info,
                        )

                if H_step != self.config.H_cycles - 1:
                    z_H = self.f_level(
                        z_H + input_embeddings + self.input_token_H,
                        z_L,
                        **seq_info,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        z_L = self.f_level(
            z_L + self.input_token_L,
            z_H + input_embeddings,
            **seq_info,
        )
        z_H = self.f_level(
            z_H + input_embeddings + self.input_token_H,
            z_L,
            **seq_info,
        )

        new_carry = AIRInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class AIR(_BaseAIR):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = AIRConfig(**config_dict)
        self.inner = AIR_Inner(self.config)
