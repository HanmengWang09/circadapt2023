function run_reference_beat(outputDir)
% 使用发布的 PRef 与原 MATLAB 方程运行一个心搏，导出原始压力、容积和流量。
root = fileparts(mfilename('fullpath'));
sourceDir = fullfile(root, 'original_matlab', 'Source Code', 'CircAdapt');
addpath(sourceDir);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

global P;
reference = load(fullfile(root, 'original_matlab', 'PRef.mat'), 'P');
P = reference.P;
assert(P.Patch.n == 20 && isequal(P.Wall.nPatch, [1 1 12 5 1]));
P.General.dt = P.General.Dt;
LinkMatrix;
Indexation;
P2SVar;
P.General.In = [];
P.General.Out = [];
P.General.BeatNr = 0;
TimingInit;
period = P.General.tCycle;
state = P.SVar(end, :);
state(1) = 0;
P.SVar = OdeCA('SVarDot', P.General.dt, period, state);
SVarDot(P.SVar');

pressure = Get('Cavity', 'p', {'Ra', 'La', 'Rv', 'Lv'}) / 133.322368;
volume = Get('Cavity', 'V', {'Ra', 'La', 'Rv', 'Lv'}) * 1e6;
arteries = Get('Node', 'p', {'Ao', 'PuAr'}) / 133.322368;
flow = Get('Valve', 'q', {'VcRa', 'PuVeLa', 'RaRv', 'LaLv', 'LvAo', 'RvPuAr'}) * 1e6;
values = [P.t, pressure, arteries, volume, flow];
assert(isreal(values) && all(isfinite(values(:))) && all(diff(P.t) > 0));
names = {'time_s', 'p_ra_mmhg', 'p_la_mmhg', 'p_rv_mmhg', 'p_lv_mmhg', 'p_ao_mmhg', ...
    'p_pa_mmhg', 'v_ra_ml', 'v_la_ml', 'v_rv_ml', 'v_lv_ml', 'q_vc_ra_ml_s', ...
    'q_pv_la_ml_s', 'q_tv_ml_s', 'q_mv_ml_s', 'q_ao_ml_s', 'q_pa_ml_s'};
writetable(array2table(values, 'VariableNames', names), fullfile(outputDir, 'reference_beat.csv'));
fprintf('Munneke2023 reference beat: %d samples, cycle %.6f s\n', numel(P.t), period);
end
