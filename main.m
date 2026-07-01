%% Illustrative Example for the thesis: "Graph Neural Network for Wide Area 6DoF Sound Field Reconstruction"
clearvars;
close all;
clc;

%% Parameters Configuration
CFG = struct();

% Paths
CFG.MODEL_PATH      = 'GNN_weights.mat';
CFG.ARVEDI_BASE_DIR = 'Dataset/arvedi_auditorium_dataset_reduced/';
CFG.DRY_SIG_1 = 'Dataset/EARS Dataset/p002_emo_adoration_sentences.wav';
CFG.DRY_SIG_2 = 'Dataset/EARS Dataset/p010_emo_cuteness_sentences.wav';
CFG.AUDIO_OUT_DIR = fullfile(pwd, 'Audio results');

CFG.FS = 16000;
CFG.WIN_LEN = 512;
CFG.HOP = CFG.WIN_LEN/4;
CFG.NFFT = 512;
CFG.DIRAC_SMOOTH_ALPHA = 0.2;
CFG.DIRAC_SH_NORM = "N3D"; 

KNOWN_NODES = {
    'A101';
    'A109';
    'A403';
    'A507';
    'A416'
};
QUERY_NODES = {
    'A301';
    'A206';
    'A311'
};

% You can choose the source
SOURCE_NAME_1 = 'S1'; % Source (S0, S1, S2, S3, S4)
SOURCE_NAME_2 = 'S4'; % Source (S0, S1, S2, S3, S4)

%% Model loading
if ~exist(CFG.MODEL_PATH, 'file')
    error('Modello non trovato. Assicurati di aver concluso il training.');
end

model_data = load(CFG.MODEL_PATH, 'params', 'CFG','history');
params = model_data.params;
CFG_TRAIN = model_data.CFG; 

% All the structural parameters come from the training
train_fields = fieldnames(CFG_TRAIN);
for k = 1:numel(train_fields)
    campo = train_fields{k};
    if ~isfield(CFG, campo)
        CFG.(campo) = CFG_TRAIN.(campo);
    end
end

%% Dry Signals loading

% Dry Signal 1
if ~exist(CFG.DRY_SIG_1, 'file'), error('File mancante: %s', CFG.DRY_SIG_1); end
[drySig1, fsDry1] = audioread(CFG.DRY_SIG_1);
if size(drySig1, 2) > 1, drySig1 = mean(drySig1, 2); end
drySig1 = drySig1(:);
if fsDry1 ~= CFG.FS, drySig1 = resample(drySig1, CFG.FS, fsDry1); drySig1 = drySig1(:); end

% Dry Signal 2
if ~exist(CFG.DRY_SIG_2, 'file'), error('File mancante: %s', CFG.DRY_SIG_2); end
[drySig2, fsDry2] = audioread(CFG.DRY_SIG_2);
if size(drySig2, 2) > 1, drySig2 = mean(drySig2, 2); end
drySig2 = drySig2(:);
if fsDry2 ~= CFG.FS, drySig2 = resample(drySig2, CFG.FS, fsDry2); drySig2 = drySig2(:); end

% No longer than 30 seconds
max_samples = 30 * CFG.FS;
if length(drySig1) > max_samples, drySig1 = drySig1(1:max_samples); end
if length(drySig2) > max_samples, drySig2 = drySig2(1:max_samples); end

lenD1 = length(drySig1);
lenD2 = length(drySig2);
maxD = max(lenD1, lenD2);
drySig1 = [drySig1; zeros(maxD - lenD1, 1)];
drySig2 = [drySig2; zeros(maxD - lenD2, 1)];

drySig1 = drySig1 / (max(abs(drySig1)) + 1e-12);
drySig2 = drySig2 / (max(abs(drySig2)) + 1e-12);

%% Arvedi signals loading
all_nodes = [KNOWN_NODES; QUERY_NODES];
N_tot = size(all_nodes, 1);
M = size(KNOWN_NODES, 1);
Q = size(QUERY_NODES, 1);

% Usiamo due celle separate per le RIR in FOA
foa_rirs_1 = cell(N_tot, 1);
foa_rirs_2 = cell(N_tot, 1);
mic_centers = zeros(N_tot, 3);

pos_path = fullfile(CFG.ARVEDI_BASE_DIR, 'pos_receivers.csv');
opts = detectImportOptions(pos_path, 'Delimiter', ';');
opts.VariableNamingRule = 'preserve'; 
pos_table = readtable(pos_path, opts);

for i = 1:N_tot
    hom_id  = all_nodes{i};
    idx_mic = find(strcmp(pos_table.mic, hom_id));
    if isempty(idx_mic), error('Coordinate non trovate nel CSV: %s', hom_id); end
    
    capsule_data = table2array(pos_table(idx_mic, 5:28));
    cap_pos = reshape(capsule_data, 3, 8)';
    mic_centers(i, :) = table2array(pos_table(idx_mic, 2:4)); 
    
    % RIR Source 1
    rir_path_1 = fullfile(CFG.ARVEDI_BASE_DIR, sprintf('rirs/rir-%s-%s.wav', SOURCE_NAME_1, hom_id));
    raw_rir_1 = audioread(rir_path_1);
    foa_rirs_1{i} = resample(encode_raw_to_foa(raw_rir_1, cap_pos), CFG.FS, 48000);
    
    % RIR Source 2
    rir_path_2 = fullfile(CFG.ARVEDI_BASE_DIR, sprintf('rirs/rir-%s-%s.wav', SOURCE_NAME_2, hom_id));
    raw_rir_2 = audioread(rir_path_2);
    foa_rirs_2{i} = resample(encode_raw_to_foa(raw_rir_2, cap_pos), CFG.FS, 48000);
end

mic_pos_known = mic_centers(1:M, :);
mic_pos_query = mic_centers(M+1:end, :);

%% Convolution and FOA Preparation
lenR1 = size(foa_rirs_1{1}, 1);
lenR2 = size(foa_rirs_2{1}, 1);
out_len = max(maxD + lenR1 - 1, maxD + lenR2 - 1);

rec_known = zeros(out_len, 4, M, 'single');
rec_query = zeros(out_len, 4, Q, 'single');

% Convolution on known nodes
for i = 1:M
    for c = 1:4
        % Zero padding for the reverberant tail
        conv1 = single(fftfilt(foa_rirs_1{i}(:,c), [drySig1; zeros(lenR1-1, 1)]));
        conv2 = single(fftfilt(foa_rirs_2{i}(:,c), [drySig2; zeros(lenR2-1, 1)]));
        
        % Final Mix
        rec_known(1:length(conv1), c, i) = rec_known(1:length(conv1), c, i) + conv1;
        rec_known(1:length(conv2), c, i) = rec_known(1:length(conv2), c, i) + conv2;
    end
end

% Convoluton and mix for gt
for i = 1:Q
    for c = 1:4
        conv1 = single(fftfilt(foa_rirs_1{M+i}(:,c), [drySig1; zeros(lenR1-1, 1)]));
        conv2 = single(fftfilt(foa_rirs_2{M+i}(:,c), [drySig2; zeros(lenR2-1, 1)]));
        
        rec_query(1:length(conv1), c, i) = rec_query(1:length(conv1), c, i) + conv1;
        rec_query(1:length(conv2), c, i) = rec_query(1:length(conv2), c, i) + conv2;
    end
end

max_known = max(abs(rec_known(:)));
max_query = max(abs(rec_query(:)));
norm_factor = max(max_known, max_query) + 1e-12;
rec_known = rec_known / norm_factor;
rec_query = rec_query / norm_factor;

[~, fvec, tvec] = stft_mics_fast(rec_known, CFG.FS, CFG.WIN_LEN, CFG.HOP, CFG.NFFT);
[X_qry, ~, ~]    = stft_mics_fast(rec_query, CFG.FS, CFG.WIN_LEN, CFG.HOP, CFG.NFFT);

%% OLA Predictions
x_gt_time   = istft_multi(X_qry, CFG.WIN_LEN, CFG.HOP, CFG.NFFT);
L_total = size(rec_query, 1); 
x_pred_time = zeros(L_total, 4, Q, 'single');

for q = 1:Q
    q_pos = mic_pos_query(q, :);
    
    fprintf('\n--- Query elaboration %d/%d ---\n', q, Q);
    
    % GNN
    x_pred_q = predict_gnn_query_ola(rec_known, mic_pos_known, q_pos, params, CFG, 4.0, false, 'GNN');
    
    x_pred_time(:, :, q) = x_pred_q;
end


%% Audio Signals generation
if ~exist(CFG.AUDIO_OUT_DIR, 'dir'), mkdir(CFG.AUDIO_OUT_DIR); end

for q = 1:Q
    q_name = QUERY_NODES{q, 1};
    
    % Binaural transformation
    binaural_gt      = shc_to_binaural(x_gt_time(:, :, q));
    binaural_pred    = shc_to_binaural(x_pred_time(:, :, q));
    
    % Normalization
    binaural_gt      = binaural_gt / (max(abs(binaural_gt(:))) + 1e-12);
    binaural_pred    = binaural_pred / (max(abs(binaural_pred(:))) + 1e-12);
    
    % FOA
    foa_gt           = x_gt_time(:, :, q);
    foa_pred         = x_pred_time(:, :, q);
    foa_gt           = foa_gt / (max(abs(foa_gt(:))) + 1e-12);
    foa_pred         = foa_pred / (max(abs(foa_pred(:))) + 1e-12);
    
    % Binaural
    audiowrite(fullfile(CFG.AUDIO_OUT_DIR, sprintf('%s_GT_binaural.wav', q_name)), binaural_gt, CFG.FS);
    audiowrite(fullfile(CFG.AUDIO_OUT_DIR, sprintf('%s_GNN_binaural_pred.wav', q_name)), binaural_pred, CFG.FS);
    
    % Ambisonics
    audiowrite(fullfile(CFG.AUDIO_OUT_DIR, sprintf('%s_GT_FOA.wav', q_name)), foa_gt, CFG.FS);
    audiowrite(fullfile(CFG.AUDIO_OUT_DIR, sprintf('%s_GNN_FOA_pred.wav', q_name)), foa_pred, CFG.FS);
    
end

fprintf('\nOpen the "Audio Results" folder and the reconstruction will be there!\n');

%% Functions
function foa_sig = encode_raw_to_foa(raw_sig, cap_pos)
    cap_pos = cap_pos - mean(cap_pos, 1);
    [az, el, ~] = cart2sph(cap_pos(:,1), cap_pos(:,2), cap_pos(:,3));
    colat = pi/2 - el;
    Y = compute_real_sh_matrix(1, az, colat); 
    enc_matrix = pinv(Y); 
    foa_sig = (enc_matrix * raw_sig')';
end

function [X, fvec, tvec] = stft_mics_fast(x, fs, winLen, hop, nfft)
    if ndims(x) == 2, x = reshape(x, size(x,1), size(x,2), 1); end
    [Ns,C,M] = size(x); win = hann(winLen, 'periodic'); nFrames = 1 + floor((Ns - winLen) / hop); F = nfft/2 + 1;
    X = complex(zeros(F, nFrames, C, M, 'single'));
    for m = 1:M, for c = 1:C, for n = 1:nFrames, spec = fft(single(x((1:winLen)+(n-1)*hop,c,m)) .* single(win), nfft); X(:,n,c,m) = spec(1:F); end, end, end
    fvec = (0:F-1)' * fs / nfft; tvec = ((0:nFrames-1) * hop + winLen/2) / fs;
end

function x = istft_multi(X, winLen, hop, nfft)
    [~, T, C, M] = size(X);
    win = hann(winLen, 'periodic');
    Ns = (T-1)*hop + winLen;
    x = zeros(Ns, C, M, 'single');
    wsum = zeros(Ns, 1, 'single');
    for m = 1:M
        for t = 1:T
            for c = 1:C
                half = squeeze(X(:,t,c,m)); frame = real(ifft([half; conj(half(end-1:-1:2))], nfft));
                x((1:winLen)+(t-1)*hop, c, m) = x((1:winLen)+(t-1)*hop, c, m) + single(frame(1:winLen) .* win);
            end
            if m == 1, wsum((1:winLen)+(t-1)*hop) = wsum((1:winLen)+(t-1)*hop) + single(win.^2); end
        end
    end
    wsum(wsum < max(wsum)*0.01) = max(wsum); x = x ./ wsum;
end