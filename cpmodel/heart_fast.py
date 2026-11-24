"""Munneke20Patch 的确定性 Numba 加速；heart.py 为逐式参考实现。

不使用 fastmath，不改变四次材料迭代或两次 TriSeg 修正。
参数在每次调用显式传入；没有跨求值状态或科学参数缓存。
"""
import numpy as np
from numba import njit
from .heart import MunnekeHeart, MMHG_PA, PROX, DIST


PATCH_FIELDS = ('LsRef', 'Lsi0Act', 'TimeAct', 'TR', 'TD', 'SfAct',
                'CRest', 'dLsPas', 'Ls0Pas', 'k1', 'SfPas',
                'LenSeriesElement', 'vMax', 'AmRef', 'VWall')


@njit(cache=True)
def _sarc(p, c, lsi, ef, t, delay, period):
    # rows: Sf, DSfDEf, CDot, LsiDot, Ls, SfEcm, SfTit, Ef, tA
    out = np.empty((9, 20))
    external_clock = p.shape[0] > 16 and p[16,0] != 0.
    phase = t if external_clock or 0. <= t <= period else t % period
    for i in range(20):
        tc = phase-delay[i]
        if tc < 0.:
            tc += period
        ls = np.exp(ef[i])*p[0, i]
        length = max(lsi[i]/p[1, i]-1., .0001)
        ta = (.75+.50*length)*p[2, i]
        tr = .55*p[3, i]*p[2, i]
        td = .33*p[4, i]*p[2, i]
        x = min(8., max(0., tc/tr))
        ft1 = x**3*np.exp(-x)*(8-x)**2*.020/tr
        x = (tc-ta)/td
        tanhx = .5+.5*np.sin(np.sign(x)*min(np.pi/2, abs(x)))
        cdot = np.tanh(9.1204*length**2)*ft1-c[i]*tanhx/td
        sf_iso = c[i]*length*1.51*p[5, i]
        if p.shape[0] > 15 and p[15,i] != 1.:
            sf_iso *= p[15,i]
        sf_rest = length*1.51*p[6, i]*p[5, i]
        kk3 = 2*p[0, i]/p[7, i]
        lfp = np.exp(ef[i])*p[0, i]/p[8, i]
        yecm, y = lfp**p[9, i], lfp**kk3
        sf_ecm = (yecm-1)*.0349*p[10, i]
        sf_tit = (y-1)*.01*p[5, i]
        dsf_pas = y*.01*p[5, i]*kk3+yecm*.0349*p[9, i]*p[10, i]
        lnorm_se = (ls-lsi[i])/p[11, i]
        out[0, i] = sf_tit+sf_ecm+(sf_iso+sf_rest)*lnorm_se-sf_rest
        out[1, i] = dsf_pas+sf_iso*ls/p[11, i]
        out[2, i] = cdot
        out[3, i] = (lnorm_se-1)*p[12, i]
        out[4, i], out[5, i], out[6, i] = ls, sf_ecm, sf_tit
        out[7, i], out[8, i] = ef[i], ta
    return out


@njit(cache=True)
def _aggregate(am0, dadt, patch_to_wall, am_dead):
    aw0, compliance = np.zeros(5), np.zeros(5)
    for i in range(20):
        w = patch_to_wall[i]
        aw0[w] += am0[i]
        compliance[w] += dadt[i]
    return am_dead+aw0, compliance


@njit(cache=True)
def _geometry(volume, vs0, ys0, vwall, aw0, dadt, rho, p0, dt):
    # Five wall entries (La,Ra,Lv,Sv,Rv), four cavity entries (Ra,La,Lv,Rv).
    aw, cm, tension = np.empty(5), np.empty(5), np.empty(5)
    pressure, area, impedance = np.empty(4), np.empty(4), np.empty(4)
    for w in range(2):
        chamber = 1-w
        vol = max(0., volume[chamber])
        vm = vol+.5*vwall[w]
        cm[w] = (4*np.pi/(3*vm))**(1/3)
        aw[w] = 4*np.pi/cm[w]**2
        compliance = dadt[w]*np.exp(100*max(0., 1-aw[w]/aw0[w]))
        tension[w] = (aw[w]-aw0[w])/compliance
        pressure[chamber] = 2*cm[w]*tension[w]-.1*p0*max(0., .5/max(vol/vwall[w], .1)-1)**2
        area[chamber] = vm/(2*vm**(1/3))
        impedance[chamber] = .5*np.sqrt(rho/(abs(area[chamber])**1.5*compliance))
    vwl, vwr = .5*(vwall[2]+vwall[3]), .5*(vwall[3]+vwall[4])
    vl, vr = volume[2]+vwl, volume[3]+vwr
    vs, ys = vs0, ys0
    for _ in range(2):
        a_ref = np.pi*ys**2
        fv, fp, fvv, fvp, fpp = 0., 0., 0., 0., 0.
        for j in range(3):
            v = (vs-vl if j == 0 else vs if j == 1 else vs+vr)/(a_ref*ys)
            a0, dtda = aw0[j+2]/a_ref, a_ref/dadt[j+2]
            x = 2*np.sinh(np.arcsinh(3*v)/3)
            xx = x*x
            rr = 1+xx
            rr2 = rr*rr
            tens = max(0., (rr-a0)*dtda)
            av, ap = 4*x/rr, (1-xx)/rr
            avv, avp, app = 8*ap/rr2, -2*av/rr2, 3*v*av/rr2
            fv += tens*av
            fp += tens*ap
            fvv += dtda*av**2+tens*avv
            fvp += dtda*av*ap+tens*avp
            fpp += dtda*ap**2+tens*app
        det = fvv*fpp-fvp*fvp
        dv = -(fv*fpp-fp*fvp)/det
        dp = -(-fv*fvp+fp*fvv)/det
        vs, ys = vs+ys**3*dv, ys+.5*ys*dp
    al, ass, ar = aw0[2], aw0[3], aw0[4]
    f1 = 1.08
    ylo = np.sqrt(ass/(1+ass/min(al, ar))/np.pi)/f1
    yhi = f1*np.sqrt(ass/(1+ass/(max(al, ar)*f1*f1))/np.pi)
    vlo = -f1**3*(2*np.pi/3)*(al/(2*np.pi))**1.5+vl
    vhi = f1**3*(2*np.pi/3)*(ar/(2*np.pi))**1.5-vr
    ys = max(ylo, min(yhi, ys))
    vs = max(vlo, min(vhi, vs))
    a_ref = np.pi*ys**2
    for j in range(3):
        v = (vs-vl if j == 0 else vs if j == 1 else vs+vr)/(a_ref*ys)
        x = 2*np.sinh(np.arcsinh(3*v)/3)
        rr = x*x+1
        tension[j+2] = max(0., (rr-aw0[j+2]/a_ref)*(a_ref/dadt[j+2]))
        aw[j+2], cm[j+2] = a_ref*rr, (2*x/rr)/ys
    for j in range(2):
        vm = vl+vwl if j == 0 else vr+vwr
        area[j+2] = vm/(2*vm**(1/3))
        impedance[j+2] = .5*np.sqrt(rho/(dadt[2+2*j]*abs(area[j+2])**1.5))
        vnlo = max(.1, (vl/vwl if j == 0 else vr/vwr)-1)
        dp_lo = .2*p0*max(0., .3/vnlo-1)**2
        w = 2+2*j
        pressure[j+2] = (-1. if j == 0 else 1.)*2*cm[w]*tension[w]-dp_lo
    return aw, cm, tension, pressure, area, impedance, vs, ys


@njit(cache=True)
def _evaluate_core(t, state, ports, pleural, dt, p, patch_to_wall, am_dead,
                   vwall, delay, period, rho, p0, bag, valve, port_area):
    volume, c, lsi, q = state[:4], state[6:26], state[26:46], state[46:52]
    stretch = lsi/p[0]
    am = stretch**2*p[13]
    sarc = _sarc(p, c, lsi, np.log(stretch), t, delay, period)
    de = sarc[0]/sarc[1]
    corr = 1+1.16*(2*p[11]/lsi+de)
    am0, dadt = am*np.exp(-2*de), 4*corr*am**2/(sarc[1]*p[14])
    initial_sarc = sarc.copy()
    initial_am0, initial_dadt = am0.copy(), dadt.copy()
    aw0, wall_dadt = _aggregate(am0, dadt, patch_to_wall, am_dead)
    initial_aw0, initial_wall_dadt = aw0.copy(), wall_dadt.copy()
    for _ in range(4):
        aw, cm, tension, trans, area, impedance, vs, ys = _geometry(
            volume, state[4], state[5], vwall, aw0, wall_dadt, rho, p0, dt)
        am = am0+tension[patch_to_wall]*dadt
        ef = .5*np.log(am/p[13])
        sarc = _sarc(p, c, lsi, ef, t, delay, period)
        de = sarc[0]/sarc[1]
        am0, dadt = am*np.exp(-2*de), 4*am**2/(sarc[1]*p[14])
        aw0, wall_dadt = _aggregate(am0, dadt, patch_to_wall, am_dead)
    peri_volume = volume.sum()+vwall.sum()
    peri = (peri_volume/bag[0])**bag[1]*bag[2]
    external = peri+pleural*MMHG_PA
    source = trans+external
    inflow = np.array([q[0]-q[4], q[1]-q[5], q[5]-q[2], q[4]-q[3]])
    node_pressure = np.empty(8)
    node_pressure[:4], node_pressure[4:] = source+impedance*inflow, ports*MMHG_PA
    node_area = np.zeros(8)
    for j in range(6):
        opening = max(valve[0, j], valve[1, j])
        node_area[PROX[j]] += opening
        node_area[DIST[j]] += opening
    node_area[:4] += area
    node_area[4:] += port_area
    qdot, valve_open = np.empty(6), np.empty(6)
    for j in range(6):
        drop = node_pressure[PROX[j]]-node_pressure[DIST[j]]
        aopen, aleak = valve[0, j], valve[1, j]
        amax = max(aopen, aleak)
        aprox, adist = node_area[PROX[j]]-amax, node_area[DIST[j]]-amax
        bwq, bwp = (1. if q[j] < 0. else 0.), (1. if drop < 0. else 0.)
        forward = 1-bwq*bwp
        a = aleak+forward*(aopen-aleak)
        resistance = 1.+rho*abs(q[j])*abs(a**-2-1/((1-bwq)*aprox**2+bwq*adist**2))
        dpb = .5*q[j]*resistance
        inertia = 1.5*rho*(valve[2, j]/a+.5*(1/np.sqrt(aprox)+1/np.sqrt(adist)))
        tau = inertia/resistance
        qdot[j] = (1-1/(1+dt/tau))*(drop-dpb)/(dt*resistance)
        valve_open[j] = forward
    derivative = np.empty(52)
    derivative[:4] = inflow
    derivative[4], derivative[5] = (vs-state[4])/dt, (ys-state[5])/dt
    derivative[6:26], derivative[26:46], derivative[46:] = sarc[2], sarc[3], qdot
    return (derivative, source, node_pressure, sarc, tension, aw, am, peri,
            node_area, impedance, area, cm, aw0, wall_dadt, am0, dadt, trans,
            peri_volume, valve_open, vs, ys, initial_sarc, initial_am0,
            initial_dadt, initial_aw0, initial_wall_dadt)


class FastMunnekeHeart(MunnekeHeart):
    """与 MunnekeHeart 相同输入/诊断；首次调用会触发 Numba 编译。"""

    def __init__(self, parameters_path=None):
        super().__init__(parameters_path)
        self._p = np.stack([np.broadcast_to(self.patch[k], (20,)) for k in PATCH_FIELDS])
        self._p_views = [self._p[i] for i in range(len(PATCH_FIELDS))]
        for key, view in zip(PATCH_FIELDS, self._p_views):
            self.patch[key] = view
        self._valve = np.stack([self.valve[k] for k in ('AOpen', 'ALeak', 'Len')])
        self._valve_views = [self._valve[i] for i in range(3)]
        for key, view in zip(('AOpen', 'ALeak', 'Len'), self._valve_views):
            self.valve[key] = view
        self._bag = np.empty(3)

    def _packed_parameters(self):
        self.refresh_timings()
        # 公开数组原位写入直接共享；数组替换也会同步，避免过期参数。
        for i, key in enumerate(PATCH_FIELDS):
            if self.patch[key] is not self._p_views[i]:
                self._p[i] = self.patch[key]
                self.patch[key] = self._p_views[i]
        for i, key in enumerate(('AOpen', 'ALeak', 'Len')):
            if self.valve[key] is not self._valve_views[i]:
                self._valve[i] = self.valve[key]
                self.valve[key] = self._valve_views[i]
        for i, key in enumerate(('VRef', 'k', 'pAdapt')):
            self._bag[i] = self.parameters['bag'][key]
        return self._p, self._bag, self._valve

    def evaluation_timings(self):
        return self.activation_delay_s,self.period_s

    def evaluate(self, t, state, port_pressure_mmhg, pleural_mmhg, dt):
        state = np.asarray(state, dtype=float)
        self._validate_state(state)
        ports = np.asarray(port_pressure_mmhg, dtype=float)
        if ports.shape != (4,) or not np.isfinite(ports).all():
            raise ValueError('四端口压力须为有限的TV/PV/SA/PA数组')
        if not np.isfinite([t, pleural_mmhg, dt]).all() or dt <= 0:
            raise ValueError('时间、胸膜压和正步长须有限')
        p, bag, valve = self._packed_parameters()
        delay,period = self.evaluation_timings()
        result = _evaluate_core(float(t), state, ports, float(pleural_mmhg), float(dt),
            p, self.patch_to_wall, self.am_dead, self.wall_volume,
            delay, period, self.rho, self.p0, bag,
            valve, self.port_area)
        (derivative, source, node_pressure, sarc, tension, aw, am, peri,
         node_area, impedance, area, cm, aw0, wall_dadt, am0, dadt, trans,
         peri_volume, valve_open, vs, ys, initial_sarc, initial_am0,
         initial_dadt, initial_aw0, initial_wall_dadt) = result
        if not np.isfinite(derivative).all():
            raise FloatingPointError('Munneke2023方程产生非有限导数；未裁剪或调参')
        sarc_names = ('Sf', 'DSfDEf', 'CDot', 'LsiDot', 'Ls', 'SfEcm', 'SfTit', 'Ef', 'tA')
        initial_material = {name: initial_sarc[i] for i, name in enumerate(sarc_names)}
        initial_material.update(Am0=initial_am0, DADT=initial_dadt)
        return {'derivative': derivative, 'pressure_mmhg': source/MMHG_PA,
                'source_pressure_mmhg': source/MMHG_PA, 'node_pressure_mmhg': node_pressure/MMHG_PA,
                'volume_ml': state[:4]*1e6, 'flow_ml_s': state[46:52]*1e6,
                'C': state[6:26].copy(), 'Lsi': state[26:46].copy(),
                'sarcomere_length': sarc[4], 'wall_tension_N_m': tension,
                'wall_area_m2': aw, 'patch_area_m2': am, 'patch_stress_pa': sarc[0],
                'pericardial_pressure_mmhg': peri/MMHG_PA, 'node_area_m2': node_area,
                'source_impedance_pa_s_m3': impedance, 'cavity_area_m2': area,
                'cavity_impedance_pa_s_m3': impedance, 'patch_stiffness_pa': sarc[1],
                'patch_ecm_stress_pa': sarc[5], 'patch_titin_stress_pa': sarc[6],
                'patch_strain': sarc[7], 'wall_curvature_m_inv': cm,
                'wall_Am0_m2': aw0, 'wall_compliance_m3_N': wall_dadt,
                'patch_Am0_m2': am0, 'patch_compliance_m3_N': dadt,
                'cavity_transmural_pressure_pa': trans, 'pericardial_volume_m3': peri_volume,
                'valve_open_fraction': valve_open, 'valve_acceleration_m3_s2': derivative[46:],
                'activation_delay_s': delay.copy(),
                'triseg_volume_m3': vs, 'junction_radius_m': ys,
                'initial_linear_patch': initial_material,
                'initial_linear_wall': {'Am0': initial_aw0, 'DADT': initial_wall_dadt, 'VWall': self.wall_volume}}
