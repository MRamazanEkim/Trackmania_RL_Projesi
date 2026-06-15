"""
discrete_action.py
==================
DQN için ayrık aksiyon sarmalayıcı.

DQN yalnız ayrık (Discrete) aksiyon uzayıyla çalışır; Trackmania ortamımız ise
sürekli `[gaz, fren, direksiyon]` bekler. Bu wrapper 5 ayrık aksiyonu sürekli
kontrole çevirir — böylece DQN, PPO ve SAC ile AYNI ortamda (eşit koşul)
karşılaştırılabilir.

5 ayrık aksiyon
---------------
  0 = gaz + düz       [1, 0,  0]
  1 = gaz + sol       [1, 0, -1]
  2 = gaz + sağ       [1, 0, +1]
  3 = fren            [0, 1,  0]
  4 = boş (coast)     [0, 0,  0]

tmrl mapping: aksiyon[0]=ileri/gaz, [1]=geri/fren, [2]=direksiyon (>0.5 sağ, <-0.5 sol).
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces


# Ayrık index → sürekli [gaz, fren, direksiyon]
DISCRETE_ACTIONS = np.array(
    [
        [1.0, 0.0,  0.0],   # 0: gaz + düz
        [1.0, 0.0, -1.0],   # 1: gaz + sol
        [1.0, 0.0,  1.0],   # 2: gaz + sağ
        [0.0, 1.0,  0.0],   # 3: fren
        [0.0, 0.0,  0.0],   # 4: boş
    ],
    dtype=np.float32,
)


class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Sürekli `[gaz, fren, direksiyon]` ortamını 5'li Discrete aksiyona indirger.
    DQN bu wrapper ile sarılmış ortamı doğrudan kullanabilir.

    Observation uzayı değişmez — yalnız action_space Discrete(5) olur.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, action) -> np.ndarray:
        # SB3 DQN tek int gönderir; diziyse ilk elemanı al
        idx = int(np.asarray(action).ravel()[0])
        idx = max(0, min(len(DISCRETE_ACTIONS) - 1, idx))
        return DISCRETE_ACTIONS[idx].copy()
