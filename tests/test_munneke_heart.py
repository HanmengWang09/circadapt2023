"""Munneke 经典心肌：直接计算、端口守恒和独立参考。"""
import json
from pathlib import Path
import unittest

import numpy as np
from cpmodel import heart as kernel


class MunnekeHeartTests(unittest.TestCase):
    def make_heart(self):
        self.assertTrue(hasattr(kernel, 'MunnekeHeart'), '需要经典C/Lsi的Munneke心脏')
        return kernel.MunnekeHeart()

    def test_initial_52_states_and_twenty_patches(self):
        heart = self.make_heart()
        state = heart.initial_state([50., 60., 120., 130.])
        self.assertEqual(heart.state_size, 52)
        self.assertEqual(heart.patch_count, 20)
        self.assertEqual(state.shape, (52,))
        np.testing.assert_allclose(state[:4], [50e-6, 60e-6, 120e-6, 130e-6])
        np.testing.assert_array_equal(state[6:26], heart.parameters['initial']['C'])
        np.testing.assert_array_equal(state[26:46], heart.parameters['initial']['Lsi'])

    def test_classic_ecm_titin_and_active_series_element(self):
        heart = self.make_heart()
        patch = heart.patch
        c = np.full(20, .2)
        lsi = np.full(20, 1.9)
        ef = np.log(np.full(20, 2.1)/patch['LsRef'])
        result = kernel.sarc_ef2sf(patch, c, lsi, ef, .2, heart.activation_delay_s, heart.period_s)
        normalized = 2.1/patch['Ls0Pas']
        ecm = .0349*patch['SfPas']*(normalized**patch['k1']-1)
        titin = .01*patch['SfAct']*(normalized**(2*patch['LsRef']/patch['dLsPas'])-1)
        active = 1.51*patch['SfAct']*c*np.maximum(lsi/patch['Lsi0Act']-1, .0001)
        np.testing.assert_allclose(result['SfEcm'], ecm, rtol=1e-13)
        np.testing.assert_allclose(result['SfTit'], titin, rtol=1e-13)
        np.testing.assert_allclose(result['Sf'], ecm+titin+active*(2.1-lsi)/patch['LenSeriesElement'], rtol=1e-13)
        np.testing.assert_allclose(result['LsiDot'], ((2.1-lsi)/patch['LenSeriesElement']-1)*patch['vMax'], rtol=1e-13)

    def test_conservation_and_cavity_pressure_are_distinct_from_node(self):
        heart = self.make_heart()
        state = heart.initial_state()
        state[46:] = np.array([30., 40., 50., 60., 70., 80.])*1e-6
        result = heart.evaluate(.1, state, [5., 8., 80., 15.], -3., 50e-6)
        np.testing.assert_allclose(result['derivative'][:4], [-40e-6, -40e-6, 30e-6, 10e-6], atol=1e-19)
        np.testing.assert_allclose(result['pressure_mmhg'], result['source_pressure_mmhg'])
        np.testing.assert_allclose((result['node_pressure_mmhg'][:4]-result['pressure_mmhg'])*kernel.MMHG_PA,
            result['source_impedance_pa_s_m3']*result['derivative'][:4], atol=3e-12)
        self.assertEqual(result['patch_area_m2'].shape, (20,))
        self.assertEqual(result['wall_area_m2'].shape, (5,))
        self.assertTrue(np.isfinite(result['derivative']).all())
        self.assertNotIn('Xb', result)
        self.assertNotIn('Ca', result)

    def test_pleural_pressure_once_and_evaluation_without_history(self):
        heart = self.make_heart()
        state = heart.initial_state()
        first = heart.evaluate(.2, state, [5., 8., 80., 15.], -3., 50e-6)
        second = heart.evaluate(.2, state, [5., 8., 80., 15.], 2., 50e-6)
        np.testing.assert_allclose(second['pressure_mmhg']-first['pressure_mmhg'], [5.]*4, atol=1e-13)
        repeated = heart.evaluate(.2, state, [5., 8., 80., 15.], -3., 50e-6)
        np.testing.assert_array_equal(first['derivative'], repeated['derivative'])

    def test_invalid_physical_states_raise(self):
        heart = self.make_heart()
        for index, value in [(0, -1e-6), (5, 0.), (6, -.1), (26, 0.)]:
            state = heart.initial_state()
            state[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                heart.evaluate(.1, state, [5., 8., 80., 15.], -3., 50e-6)

    def test_independent_original_matlab_reference(self):
        # 捕获：误用Arts材料、Corr遗漏、TriSeg修正次数/pi错误、少做外循环、
        # 瓣膜关闭分支/惯性错误、心包重标，以及把节点压力当腔压。
        heart = self.make_heart()
        fixture = Path(__file__).parent/'fixtures/munneke_reference.json'
        reference = json.loads(fixture.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(reference['cases']), 6)
        for case in reference['cases']:
            result = heart.evaluate(case['t_s'], np.asarray(case['state']),
                case['port_pressure_mmhg'], case['pleural_mmhg'], case['dt_s'])
            for field, expected in case['expected'].items():
                if field.startswith('initial_linear_'):
                    for quantity, value in expected.items():
                        with self.subTest(case=case['name'], field=field+'.'+quantity):
                            np.testing.assert_allclose(result[field][quantity], value,
                                rtol=3e-10, atol=3e-12)
                else:
                    with self.subTest(case=case['name'], field=field):
                        np.testing.assert_allclose(result[field], expected, rtol=3e-10, atol=3e-12)

    def test_activation_keeps_the_original_cycle_endpoint(self):
        # 捕获：把当前周期终点错误映射为下周期起点，令Ra的C降解消失。
        heart = self.make_heart()
        c, lsi = np.full(20, .2), np.full(20, 1.9)
        ef = np.zeros(20)
        end = kernel.sarc_ef2sf(heart.patch, c, lsi, ef, heart.period_s,
                               heart.activation_delay_s, heart.period_s)
        self.assertAlmostEqual(end['CDot'][1], -.2/(.33*.4*.15), places=12)
        start = kernel.sarc_ef2sf(heart.patch, c, lsi, ef, 0.,
                                 heart.activation_delay_s, heart.period_s)
        self.assertEqual(start['CDot'][1], 0.)

    def test_invalid_port_pressure_or_step_raises(self):
        heart = self.make_heart()
        state = heart.initial_state()
        for ports, dt in [([1., 2., 3.], 1e-4), ([1., 2., np.nan, 4.], 1e-4),
                          ([1., 2., 3., 4.], 0.), ([1., 2., 3., 4.], -1e-4)]:
            with self.subTest(ports=ports, dt=dt), self.assertRaises(ValueError):
                heart.evaluate(.1, state, ports, -2., dt)


if __name__ == '__main__':
    unittest.main()
