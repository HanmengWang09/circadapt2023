"""Munneke2023 经典 C/Lsi 五壁20 Patch心脏的直接 Python 数值实现。

逐式移植 SarcEf2Sf、PatchWallA2T、Wall2Patch2Sarc、ChamberV2p、
TriSegV2p、SVarDot、pCavity、ValveqDot。内部 SI，肌节长度沿用微米。
无 DLL、MEX、MATLAB 运行时；TV/PV/SA/PA 为理想压力端口。
"""
from pathlib import Path
import json
import numpy as np
from .settings import PA
from .timings import original_timings

MMHG_PA = PA
WALL_NAMES = ('La', 'Ra', 'Lv', 'Sv', 'Rv')
CHAMBER_NAMES = ('Ra', 'La', 'Lv', 'Rv')
FLOW_NAMES = ('VcRa', 'PuVeLa', 'LvAo', 'RvPuAr', 'RaRv', 'LaLv')
NODE_NAMES = CHAMBER_NAMES+('TV', 'PV', 'SA', 'PA')
PATCH_NAMES = ('La1', 'Ra1')+tuple(f'Lv{i}' for i in range(1, 13))+tuple(f'Sv{i}' for i in range(1, 6))+('Rv1',)
PROX = np.array([4, 5, 2, 3, 0, 1])
DIST = np.array([0, 1, 6, 7, 3, 2])
INCIDENCE = np.zeros((8, 6))
INCIDENCE[PROX, np.arange(6)] = -1.
INCIDENCE[DIST, np.arange(6)] = 1.


def cap_height(volume_over_pi_radius_cubed):
    """VdPi2X 的等价实数表达，避免负体积立方根的消减误差。"""
    return 2*np.sinh(np.arcsinh(3*np.asarray(volume_over_pi_radius_cubed))/3)


def sarc_ef2sf(patch, c, lsi, ef, t, activation_delay_s, period):
    """原 SarcEf2Sf：ECM+titin、经典收缩与 C/Lsi 导数。"""
    phase = float(t) if 0. <= t <= period else float(t) % period
    tc = phase-np.asarray(activation_delay_s)
    tc = np.where(tc < 0., tc+period, tc)
    ls = np.exp(ef)*patch['LsRef']
    length = np.maximum(lsi/patch['Lsi0Act']-1., .0001)
    ta = (.75+.50*length)*patch['TimeAct']
    tr = .55*patch['TR']*patch['TimeAct']
    td = .33*patch['TD']*patch['TimeAct']
    x = np.minimum(8., np.maximum(0., tc/tr))
    ft1 = x**3*np.exp(-x)*(8-x)**2*.020/tr
    x = (tc-ta)/td
    tanhx = .5+.5*np.sin(np.sign(x)*np.minimum(np.pi/2, np.abs(x)))
    cdot = np.tanh(9.1204*length**2)*ft1-c*tanhx/td
    sf_iso = c*length*1.51*patch['SfAct']
    sf_rest = length*1.51*patch['CRest']*patch['SfAct']
    kk3 = 2*patch['LsRef']/patch['dLsPas']
    lfp = np.exp(ef)*patch['LsRef']/patch['Ls0Pas']
    yecm, y = lfp**patch['k1'], lfp**kk3
    sf_ecm = (yecm-1)*.0349*patch['SfPas']
    sf_tit = (y-1)*.01*patch['SfAct']
    dsf_pas = y*.01*patch['SfAct']*kk3+yecm*.0349*patch['k1']*patch['SfPas']
    lnorm_se = (ls-lsi)/patch['LenSeriesElement']
    return {'Sf': sf_tit+sf_ecm+(sf_iso+sf_rest)*lnorm_se-sf_rest,
            'DSfDEf': dsf_pas+sf_iso*ls/patch['LenSeriesElement'],
            'CDot': cdot, 'LsiDot': (lnorm_se-1)*patch['vMax'], 'Ls': ls,
            'SfEcm': sf_ecm, 'SfTit': sf_tit, 'Ef': np.asarray(ef), 'tA': ta}


def patch_wall_linear(patch, c, lsi, t, activation_delay_s, period):
    """原 PatchWallA2T 的起始线性化；Corr 仅出现在本步。"""
    stretch = lsi/patch['LsRef']
    am = stretch**2*patch['AmRef']
    sarc = sarc_ef2sf(patch, c, lsi, np.log(stretch), t, activation_delay_s, period)
    def_ = sarc['Sf']/sarc['DSfDEf']
    corr = 1+1.16*(2*patch['LenSeriesElement']/lsi+def_)
    return {**sarc, 'Am0': am*np.exp(-2*def_),
            'DADT': 4*corr*am**2/(sarc['DSfDEf']*patch['VWall'])}


def chamber_geometry(volume, vwall, aw0, dadt, rho, p0):
    """原 ChamberV2p；屈曲只改变腔的局部顺应性。"""
    volume = np.maximum(0., volume)
    vm = volume+.5*vwall
    cm = (4*np.pi/(3*vm))**(1/3)
    aw = 4*np.pi/cm**2
    compliance = dadt*np.exp(100*np.maximum(0., 1-aw/aw0))
    tension = (aw-aw0)/compliance
    pressure = 2*cm*tension-.1*p0*np.maximum(0., .5/np.maximum(volume/vwall, .1)-1)**2
    area = vm/(2*vm**(1/3))
    impedance = .5*np.sqrt(rho/(np.abs(area)**1.5*compliance))
    return {'Aw': aw, 'Cm': cm, 'T': tension, 'pTrans': pressure,
            'A': area, 'Z': impedance, 'DADT_buckled': compliance}


def triseg_geometry(vl_cavity, vr_cavity, vs0, ys0, vwall, aw0, dadt, rho, p0, dt):
    """原 TriSegV2p 两次 DvDp 修正、源码限幅及 Tau=dt 低通。"""
    vwl = .5*(vwall[0]+vwall[1])
    vwr = .5*(vwall[1]+vwall[2])
    vl, vr = vl_cavity+vwl, vr_cavity+vwr
    vs, ys = vs0, ys0
    for _ in range(2):
        a_ref = np.pi*ys**2
        v = np.array([vs-vl, vs, vs+vr])/(a_ref*ys)
        a0, dtda = aw0/a_ref, a_ref/dadt
        x = cap_height(v)
        xx = x*x
        rr = 1+xx
        rr2 = rr*rr
        tension = np.maximum(0., (rr-a0)*dtda)
        av, ap = 4*x/rr, (1-xx)/rr
        avv, avp, app = 8*ap/rr2, -2*av/rr2, 3*v*av/rr2
        fv, fp = np.sum(tension*av), np.sum(tension*ap)
        fvv = np.sum(dtda*av**2+tension*avv)
        fvp = np.sum(dtda*av*ap+tension*avp)
        fpp = np.sum(dtda*ap**2+tension*app)
        det = fvv*fpp-fvp*fvp
        dv = -(fv*fpp-fp*fvp)/det
        dp = -(-fv*fvp+fp*fvv)/det
        # Original DvDp uses YS^3, without pi; fixed two iterations.
        vs, ys = vs+ys**3*dv, ys+.5*ys*dp
    al, ass, ar = aw0
    f1 = 1.08
    ylo = np.sqrt(ass/(1+ass/min(al, ar))/np.pi)/f1
    yhi = f1*np.sqrt(ass/(1+ass/(max(al, ar)*f1*f1))/np.pi)
    vlo = -f1**3*(2*np.pi/3)*(al/(2*np.pi))**1.5+vl
    vhi = f1**3*(2*np.pi/3)*(ar/(2*np.pi))**1.5-vr
    ys = max(ylo, min(yhi, ys))
    vs = max(vlo, min(vhi, vs))
    a_ref = np.pi*ys**2
    v = np.array([vs-vl, vs, vs+vr])/(a_ref*ys)
    x = cap_height(v)
    rr = x*x+1
    tension = np.maximum(0., (rr-aw0/a_ref)*(a_ref/dadt))
    aw, cm = a_ref*rr, (2*x/rr)/ys
    ptrans_wall = 2*cm*tension
    # Original Vm adds VWL/VWR a second time after defining VL/VR above.
    vm = np.array([vl+vwl, vr+vwr])
    area = vm/(2*vm**(1/3))
    impedance = .5*np.sqrt(rho/(dadt[[0, 2]]*np.abs(area)**1.5))
    vnlo = np.maximum(.1, np.array([vl, vr])/np.array([vwl, vwr])-1)
    dp_lo = .2*p0*np.maximum(0., .3/vnlo-1)**2
    pressure = np.array([-ptrans_wall[0], ptrans_wall[2]])-dp_lo
    return {'Aw': aw, 'Cm': cm, 'T': tension, 'A': area, 'Z': impedance,
            'pTrans': pressure, 'wall_pTrans': ptrans_wall-np.mean(ptrans_wall),
            'VS': vs, 'YS': ys, 'VDot': (vs-vs0)/dt, 'YDot': (ys-ys0)/dt}


def valve_derivative(q, node_pressure, node_area, aopen, aleak, length, rho, dt):
    """原 ValveqDot：逆流且逆压时关闭，保留 Bernoulli 与惯性公式。"""
    drop = node_pressure[PROX]-node_pressure[DIST]
    amax = np.maximum(aopen, aleak)
    aprox, adist = node_area[PROX]-amax, node_area[DIST]-amax
    bwq, bwp = (q < 0.).astype(float), (drop < 0.).astype(float)
    forward = 1-bwq*bwp
    area = aleak+forward*(aopen-aleak)
    resistance = 1.+rho*np.abs(q)*np.abs(area**-2-1/((1-bwq)*aprox**2+bwq*adist**2))
    dpb = .5*q*resistance
    inertia = 1.5*rho*(length/area+.5*(1/np.sqrt(aprox)+1/np.sqrt(adist)))
    tau = inertia/resistance
    return (1-1/(1+dt/tau))*(drop-dpb)/(dt*resistance)


class MunnekeHeart:
    """20 Patch 五壁、四腔52态；evaluate 只依赖显式输入。"""
    state_size = 52
    patch_count = 20

    def __init__(self, parameters_path=None):
        path = Path(parameters_path) if parameters_path else Path(__file__).resolve().parents[1]/'data/munneke_parameters.json'
        self.parameters = json.loads(path.read_text(encoding='utf-8-sig'))
        self.patch = {k: np.asarray(v, dtype=float) for k, v in self.parameters['patch'].items()
                      if isinstance(v, (int, float, list))}
        self.patch_to_wall = np.asarray(self.parameters['wall']['patch_to_wall'], dtype=int)
        self.am_dead = np.asarray(self.parameters['wall']['AmDead'], dtype=float)
        self.wall_volume = np.bincount(self.patch_to_wall, weights=self.patch['VWall'], minlength=5)
        self.valve = {k: np.asarray(self.parameters['valve'][k], dtype=float) for k in ('AOpen', 'ALeak', 'Len')}
        self.port_area = np.asarray(self.parameters['port_area_m2'], dtype=float)
        general = self.parameters['general']
        self.period_s = float(general['period_s'])
        self.rho, self.p0 = float(general['rho_b']), float(general['p0'])
        self.activation_delay_s = np.asarray(self.parameters['timings']['activation_delay_s'], dtype=float)
        self.refresh_timings()

    def refresh_timings(self):
        general = self.parameters['general']
        duration, delay = original_timings(self.period_s, general['reference_period_s'],
            general['tau_av'], general['time_fac'], self.patch_to_wall, self.patch['dT'])
        self.patch['TimeAct'][...] = duration
        self.activation_delay_s = delay

    def initial_state(self, chamber_volumes_ml=None):
        initial = self.parameters['initial']
        volume = np.asarray(initial['chamber_volume_m3'], dtype=float)
        if chamber_volumes_ml is not None:
            volume = np.asarray(chamber_volumes_ml, dtype=float)*1e-6
        state = np.r_[volume, initial['triseg_volume_m3'], initial['junction_radius_m'],
                      initial['C'], initial['Lsi'], initial['flow_m3_s']].astype(float)
        self._validate_state(state)
        return state

    @staticmethod
    def _validate_state(state):
        if state.shape != (52,) or not np.isfinite(state).all():
            raise ValueError('Munneke心脏需要52维有限物理状态')
        if np.any(state[:4] <= 0) or state[5] <= 0 or np.any(state[6:26] < 0) or np.any(state[26:46] <= 0):
            raise ValueError('心腔容积、交界半径和Lsi须正，C须非负')

    def _aggregate_wall(self, material):
        return {'Am0': self.am_dead+np.bincount(self.patch_to_wall, weights=material['Am0'], minlength=5),
                'DADT': np.bincount(self.patch_to_wall, weights=material['DADT'], minlength=5),
                'VWall': self.wall_volume}

    def evaluate(self, t, state, port_pressure_mmhg, pleural_mmhg, dt):
        self.refresh_timings()
        state = np.asarray(state, dtype=float)
        self._validate_state(state)
        port_pressure = np.asarray(port_pressure_mmhg, dtype=float)*MMHG_PA
        if port_pressure.shape != (4,) or not np.isfinite(port_pressure).all():
            raise ValueError('四端口压力须为有限的TV/PV/SA/PA数组')
        if not np.isfinite([t, pleural_mmhg, dt]).all() or dt <= 0:
            raise ValueError('时间、胸膜压和正步长须有限')
        volume, c, lsi, q = state[:4], state[6:26], state[26:46], state[46:52]
        material = patch_wall_linear(self.patch, c, lsi, t, self.activation_delay_s, self.period_s)
        initial_material = material
        wall = self._aggregate_wall(material)
        initial_wall = wall
        for _ in range(4):
            atria = chamber_geometry(volume[[1, 0]], self.wall_volume[:2], wall['Am0'][:2],
                                      wall['DADT'][:2], self.rho, self.p0)
            tri = triseg_geometry(volume[2], volume[3], state[4], state[5], self.wall_volume[2:],
                                   wall['Am0'][2:], wall['DADT'][2:], self.rho, self.p0, dt)
            tension = np.r_[atria['T'], tri['T']]
            am = material['Am0']+tension[self.patch_to_wall]*material['DADT']
            ef = .5*np.log(am/self.patch['AmRef'])
            # Both source SarcEf2Sf calls receive the same Ef/C/Lsi.
            sarc = sarc_ef2sf(self.patch, c, lsi, ef, t, self.activation_delay_s, self.period_s)
            def_ = sarc['Sf']/sarc['DSfDEf']
            material = {'Am0': am*np.exp(-2*def_),
                        'DADT': 4*am**2/(sarc['DSfDEf']*self.patch['VWall'])}
            wall = self._aggregate_wall(material)
        bag = self.parameters['bag']
        pericardial = ((volume.sum()+self.wall_volume.sum())/bag['VRef'])**bag['k']*bag['pAdapt']
        external = pericardial+pleural_mmhg*MMHG_PA
        source = np.r_[atria['pTrans'][[1, 0]], tri['pTrans']]+external
        impedance = np.r_[atria['Z'][[1, 0]], tri['Z']]
        cavity_area = np.r_[atria['A'][[1, 0]], tri['A']]
        net_inflow = INCIDENCE[:4]@q
        node_pressure = np.r_[source+impedance*net_inflow, port_pressure]
        opening_area = np.maximum(self.valve['AOpen'], self.valve['ALeak'])
        node_area = np.r_[cavity_area, self.port_area]+np.abs(INCIDENCE)@opening_area
        qdot = valve_derivative(q, node_pressure, node_area, self.valve['AOpen'],
                                self.valve['ALeak'], self.valve['Len'], self.rho, dt)
        valve_open = 1.-((q < 0.) & (node_pressure[PROX]-node_pressure[DIST] < 0.)).astype(float)
        derivative = np.r_[net_inflow, tri['VDot'], tri['YDot'], sarc['CDot'], sarc['LsiDot'], qdot]
        if not np.isfinite(derivative).all():
            raise FloatingPointError('Munneke2023方程产生非有限导数；未裁剪或调参')
        return {'derivative': derivative, 'pressure_mmhg': source/MMHG_PA,
                'source_pressure_mmhg': source/MMHG_PA, 'node_pressure_mmhg': node_pressure/MMHG_PA,
                'volume_ml': volume*1e6, 'flow_ml_s': q*1e6, 'C': c.copy(), 'Lsi': lsi.copy(),
                'sarcomere_length': sarc['Ls'], 'wall_tension_N_m': tension,
                'wall_area_m2': np.r_[atria['Aw'], tri['Aw']], 'patch_area_m2': am,
                'patch_stress_pa': sarc['Sf'], 'pericardial_pressure_mmhg': pericardial/MMHG_PA,
                'node_area_m2': node_area, 'source_impedance_pa_s_m3': impedance,
                'cavity_area_m2': cavity_area, 'cavity_impedance_pa_s_m3': impedance,
                'patch_stiffness_pa': sarc['DSfDEf'], 'patch_ecm_stress_pa': sarc['SfEcm'],
                'patch_titin_stress_pa': sarc['SfTit'], 'patch_strain': sarc['Ef'],
                'wall_curvature_m_inv': np.r_[atria['Cm'], tri['Cm']],
                'wall_Am0_m2': wall['Am0'], 'wall_compliance_m3_N': wall['DADT'],
                'patch_Am0_m2': material['Am0'], 'patch_compliance_m3_N': material['DADT'],
                'cavity_transmural_pressure_pa': np.r_[atria['pTrans'][[1, 0]], tri['pTrans']],
                'pericardial_volume_m3': volume.sum()+self.wall_volume.sum(),
                'valve_open_fraction': valve_open, 'valve_acceleration_m3_s2': qdot,
                'activation_delay_s': self.activation_delay_s.copy(),
                'triseg_volume_m3': tri['VS'], 'junction_radius_m': tri['YS'],
                'initial_linear_patch': initial_material, 'initial_linear_wall': initial_wall}
