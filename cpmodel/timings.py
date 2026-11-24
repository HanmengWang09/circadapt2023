"""固定来源TimingInit和SarcEf2Sf/Depolarization的原顺序规则。"""
import numpy as np


def original_timings(period, reference_period, tau_av, time_fac, patch_to_wall, patch_delay):
    mapping = np.asarray(patch_to_wall, dtype=int)
    dtime = np.asarray(patch_delay, dtype=float)
    if mapping.shape != (20,) or dtime.shape != (20,):
        raise ValueError('原20Patch时序需要20个壁索引和分区时延')
    if not np.isfinite([period, reference_period, tau_av, time_fac]).all() or min(period, reference_period, time_fac) <= 0 or not np.isfinite(dtime).all():
        raise ValueError('原时序输入须有限且周期/时间尺度为正')
    duration = np.where(mapping < 2, .15*(period/.85)*time_fac,
                        (.10*reference_period+.40*period)*time_fac)
    # Ra1,La1,Lv1,Sv1,Rv1索引；保留原Pathway和壁内双向路径的写入顺序。
    paths = [(1, 1, period), (1, 0, .02*(reference_period/.85)*time_fac),
             (1, 19, tau_av), (1, 2, tau_av), (19, 14, 0.), (14, 2, 0.)]
    for wall in range(5):
        indices = np.flatnonzero(mapping == wall)
        paths.extend((int(a), int(b), 0.) for a, b in zip(indices[:-1], indices[1:]))
        paths.extend((int(b), int(a), 0.) for a, b in zip(indices[:-1], indices[1:]))
    activation = np.zeros(20)
    for prox, dist, delay in paths:
        activation[dist] = activation[prox]+delay+dtime[dist]-dtime[prox]
        activation[1] = 0.
    return duration, activation
