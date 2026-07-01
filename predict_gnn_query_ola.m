function [rec_query_hat] = predict_spatial_query_ola(rec_known, mic_pos_known, q_pos, params, CFG, chunk_dur_sec, debug, method)
    % PREDICT_SPATIAL_QUERY_OLA: Ricostruisce il campo sonoro FOA nella posizione 'q_pos'
    % tramite Overlap-Add (OLA). Supporta sia pesi dinamici (GNN) che statici (Delaunay).
    % 
    % INPUT:
    % - rec_known: [Samples, 4, M] Segnali FOA registrati dai microfoni noti
    % - mic_pos_known: [M, 3] Coordinate spaziali dei microfoni noti
    % - q_pos: [1, 3] Coordinate del microfono da ricostruire
    % - params: Pesi addestrati della rete GNN
    % - CFG: Configurazione utente
    % - chunk_dur_sec: (Opzionale) Durata blocco in sec (Default: 4.0)
    % - debug: (Opzionale) Se true, mostra l'evoluzione live dei pesi
    % - method: (Opzionale) Stringa: 'GNN' (default) oppure 'Delaunay'
    
    if nargin < 6 || isempty(chunk_dur_sec), chunk_dur_sec = 4.0; end
    if nargin < 7 || isempty(debug), debug = false; end
    if nargin < 8 || isempty(method), method = 'GNN'; end

    fs = CFG.FS;
    M = size(rec_known, 3);
    C = size(rec_known, 2); % 4 canali FOA
    L_total = size(rec_known, 1);

    %   PRE-CALCOLO DELAUNAY (Se richiesto)  
    if strcmpi(method, 'Delaunay')
        fprintf('\nMetodo: DELAUNAY. Calcolo pesi baricentrici statici...\n');
        [w_del, idx_loc] = dirac_local_weights_delaunay(mic_pos_known, q_pos, CFG.DIRAC_DELAUNAY_FALLBACK);
        % Creiamo un vettore denso 1xM con i pesi Delaunay sui microfoni corretti
        w_dense = zeros(1, M, 'single');
        w_dense(idx_loc) = w_del;
    else
        fprintf('\nMetodo: GNN. Inferenza dinamica spazio-spettrale attivata...\n');
    end

    %   0. GLOBAL NORMALIZATION (PRE-SCALING)  
    global_max = max(abs(rec_known(:)));
    rec_known_norm = rec_known / (global_max + 1e-12);

    %   SETUP OVERLAP-ADD (50% Overlap)  
    chunk_samples = round(chunk_dur_sec * fs);
    if mod(chunk_samples, 2) ~= 0, chunk_samples = chunk_samples + 1; end % Forza pari
    overlap_samples = ceil(chunk_samples / 100 * 50);
    hop_samples = chunk_samples - overlap_samples;

    % Pre-padding
    pad_len = overlap_samples;
    rec_known_pad = cat(1, zeros(pad_len, C, M, 'single'), single(rec_known_norm), zeros(pad_len, C, M, 'single'));
    L_pad = size(rec_known_pad, 1);

    % Inizializzazione buffer
    out_buffer = zeros(L_pad, C, 'single');
    win_sum = zeros(L_pad, 1, 'single');
    win = single(hann(chunk_samples, 'periodic'));

    mic_pos_known_batched = reshape(single(mic_pos_known), 1, M, 3);
    eps_val = single(1e-8);
    start_idx = 1;
    chunk_idx = 1;

    fprintf('Inizio ricostruzione OLA (Chunk: %.1fs, Overlap: %.1fs)...\n', chunk_dur_sec, overlap_samples/fs);

    % Setup finestra di Debug
    if debug
        fig_debug = figure('Name', sprintf('OLA Live Debug: %s Weights', upper(method)), 'Color', 'w', 'Position', [100, 100, max(1200, M*300), 400]);
    end

    %   LOOP SUI CHUNK  
    while start_idx + chunk_samples - 1 <= L_pad
        end_idx = start_idx + chunk_samples - 1;
        chunk_in = rec_known_pad(start_idx:end_idx, :, :);

        %   1. STFT del chunk  
        [X_obs, fvec, tvec] = stft_mics_fast(chunk_in, fs, CFG.WIN_LEN, CFG.HOP, CFG.NFFT);

        %   2. Estrazione parametri DirAC  
        [~, E_obs, ~, ~, u_obs, ~, psi_obs] = extract_dirac_states_foa(X_obs, CFG.DIRAC_SMOOTH_ALPHA, CFG.DIRAC_SH_NORM);

        %   3. Preparazione features  
        F = size(E_obs, 1); 
        T = size(E_obs, 2); 
        FT = F * T;
        
        E_obs_flat   = reshape(single(E_obs), FT, M);
        psi_obs_flat = reshape(single(psi_obs), FT, M);
        u_obs_flat   = reshape(single(u_obs), FT, M, 3);

        %   4. PREDIZIONE DEI PESI (GNN vs DELAUNAY)  
        W_full = zeros(FT, M, 'single');
        
        if strcmpi(method, 'GNN')
            freqNorm = normalize01(single(fvec(:)));
            freqNormFT = repmat(freqNorm, 1, T);
            freqNormFT = freqNormFT(:);

            evalBatch = 8192; 
            for s = 1:evalBatch:FT
                e = min(s + evalBatch - 1, FT);
                Bcur = e - s + 1;
                
                [w_b, ~] = predict_weights_from_batch(...
                    params, E_obs_flat(s:e, :), psi_obs_flat(s:e, :), u_obs_flat(s:e, :, :), ...
                    repmat(single(q_pos), Bcur, 1), freqNormFT(s:e), ...
                    repmat(mic_pos_known_batched, Bcur, 1, 1), CFG, ones(Bcur, M, 'single'));
                
                W_full(s:e, :) = gather(w_b);
            end
            % Normalizzazione baricentrica per la GNN
            W_full_sum = sum(W_full, 2);
            W_full = W_full ./ max(W_full_sum, eps_val);
            
        elseif strcmpi(method, 'Delaunay')
            % Delaunay usa pesi statici per tutta la durata e per tutte le frequenze!
            W_full = repmat(w_dense, FT, 1);
        end

        % =================== DASHBOARD LIVE DEBUG =====================
        if debug && isvalid(fig_debug)
            figure(fig_debug); clf;
            t_layout = tiledlayout(1, M, 'Padding', 'compact', 'TileSpacing', 'compact');
            title(t_layout, sprintf('%s - Elaborazione Chunk OLA N.%d (%.2f sec -> %.2f sec)', ...
                upper(method), chunk_idx, (start_idx-1)/fs, (end_idx-1)/fs), 'FontSize', 14, 'FontWeight', 'bold');
            
            for m = 1:M
                W_m_map = reshape(W_full(:, m), F, T);
                ax_m = nexttile;
                imagesc(tvec, fvec, W_m_map);
                axis(ax_m, 'xy'); colormap(ax_m, parula); colorbar(ax_m);
                caxis(ax_m, [0, 1]); 
                
                title(ax_m, sprintf('Pesi Mic Noto %d', m));
                xlabel(ax_m, 'Tempo Locale [s]');
                if m == 1, ylabel(ax_m, 'Freq [Hz]'); end
            end
            drawnow limitrate; 
        end
        % ==============================================================

        %   5. Interpolazione Parametri  
        [E_hat, psi_hat, u_hat] = reconstruct_dirac_from_weights(W_full, E_obs_flat, psi_obs_flat, u_obs_flat);

        % Anchor per la fase
        [~, anchor_idx] = min(vecnorm(mic_pos_known - q_pos, 2, 2));
        anchor_X = X_obs(:, :, :, anchor_idx);

        %   6. Rendering Audio FOA  
        X_pred = dirac_render_from_local_params(...
            reshape(E_hat, F, T), reshape(u_hat, F, T, 3), reshape(psi_hat, F, T), ...
            anchor_X, CFG.DIRAC_DIFFUSE_MODE, CFG.DIRAC_SH_NORM);

        % Trasformata Inversa
        chunk_out = istft_multi(reshape(X_pred, F, T, C, 1), CFG.WIN_LEN, CFG.HOP, CFG.NFFT);
        len_out = size(chunk_out, 1);
        if len_out > chunk_samples
            chunk_out = chunk_out(1:chunk_samples, :);
        elseif len_out < chunk_samples
            chunk_out = cat(1, chunk_out, zeros(chunk_samples - len_out, C, 'single'));
        end

        %   7. Somma sul Buffer Globale (Crossfade)  
        out_buffer(start_idx:end_idx, :) = out_buffer(start_idx:end_idx, :) + chunk_out .* win;
        win_sum(start_idx:end_idx) = win_sum(start_idx:end_idx) + win;

        start_idx = start_idx + hop_samples;
        chunk_idx = chunk_idx + 1;
    end
    
    %   CHECK OLA: VERIFICA SOMMA FINESTRE  
    % Controlliamo la parte "veramente a regime" del segnale, 
    % escludendo l'intero primo e ultimo blocco di transitorio OLA.
    if L_pad > 2 * chunk_samples
        steady_state_win = win_sum(chunk_samples + 1 : end - chunk_samples);
        min_w = min(steady_state_win);
        max_w = max(steady_state_win);
        
        fprintf('   -> [OLA Check] Somma finestre | Min: %.3f | Max: %.3f\n', min_w, max_w);
        
        if abs(min_w - 1.0) > 1e-2 || abs(max_w - 1.0) > 1e-2
            fprintf('   -> [AVVISO] La somma NON è 1. L''overlap non rispetta COLA perfetto.\n');
        else
            fprintf('   -> [PERFETTO] Condizione COLA rigorosamente rispettata (Somma = 1)!\n');
        end
    end

    %   FINALIZZAZIONE OVERLAP-ADD  
    win_sum(win_sum < 1e-6) = 1; 
    out_pad = out_buffer ./ win_sum;
    rec_query_hat = out_pad(pad_len + 1 : pad_len + L_total, :);
    
    %   DENORMALIZZAZIONE (POST-SCALING)  
    rec_query_hat = rec_query_hat * global_max;
    
    fprintf('Ricostruzione OLA completata! (%d grafi elaborati)\n', chunk_idx - 1);
end

function X_out = dirac_render_from_local_params(E_target, u, psi, X_proto, diffuseMode, shNorm)
    eps_val = single(1e-8);
    [F, T, Cuse] = size(X_proto); N = F * T;
    Edes = max(single(E_target(:)), 0);
    psi_ft = min(max(single(psi(:)), 0), 1);
    u_flat = reshape(u, N, 3);
    X_ref = reshape(X_proto, N, Cuse); 
    Ecoh = (1 - psi_ft) .* Edes;
    Ediff = psi_ft .* Edes;
    
    d_full = sh_n3d_vec(1, u_flat.').';
    E_d = batch_dirac_energy(d_full, shNorm);
    phase_w = X_ref(:, 1) ./ (abs(X_ref(:, 1)) + eps_val);
    x_dir = d_full .* sqrt(Ecoh ./ E_d);
    
    noise_vec = complex(randn(N, Cuse, 'single'), randn(N, Cuse, 'single'));
    x_diff = noise_vec .* sqrt(Ediff); 
   
    if strcmpi(string(diffuseMode), "anchor_residual")
    
        % 1. Calcolo del residuo "sporco" originale
        norm_d = real(sum(d_full .* conj(d_full), 2)) + eps_val;
        proj_scalar = sum(conj(d_full) .* X_ref, 2) ./ norm_d;
        P_d_x = d_full .* proj_scalar;
        x_res_raw = X_ref - P_d_x;
    
        % =================================================================
        %   SCORPORO SPAZIALITÀ DAL DIFFUSO (DECORRELAZIONE STFT)  
        % =================================================================
        % Prendiamo SOLO l'inviluppo spettro-temporale (il "timbro") dal canale W.
        % Il canale W (indice 1) è una sfera perfetta, priva di bias direzionali.
        mag_res = abs(x_res_raw(:, 1));
    
        % Generiamo fasi casuali scorrelate per tutti i canali
        rand_phases = exp(1i * 2*pi * rand(N, Cuse, 'single'));
    
        % Creiamo il nuovo campo diffuso: Timbro reale + Spazialità isotropa
        x_res = repmat(mag_res, 1, Cuse) .* rand_phases;
        % =================================================================
    
        E_res = batch_dirac_energy(x_res(:, 1:min(4, Cuse)), shNorm);
    
        mask = E_res < 1e-10;
        if any(mask)
            e1_fallback = zeros(sum(mask), Cuse, 'single');
            e1_fallback(:, 1) = 1;
            proj_e1 = conj(d_full(mask, 1)) ./ norm_d(mask);
            P_d_e1 = d_full(mask, :) .* proj_e1;
            x_res_fallback = e1_fallback - P_d_e1;
            x_res(mask, :) = x_res_fallback;
            E_res(mask) = batch_dirac_energy(x_res_fallback(:, 1:min(4, Cuse)), shNorm);
        end
        x_diff = x_res .* sqrt(Ediff ./ E_res);
    else
        noise_vec = complex(randn(N, Cuse, 'single'), randn(N, Cuse, 'single'));
        E_noise = batch_dirac_energy(noise_vec(:, 1:min(4, Cuse)), shNorm);
        x_diff = noise_vec .* sqrt(Ediff ./ E_noise);
    end
    
    x_syn = x_dir + x_diff;
    % E_syn = batch_dirac_energy(x_syn, shNorm);
    % x_syn = x_syn .* sqrt(Edes ./ E_syn);
    
    phase_syn0 = x_syn(:, 1) ./ (abs(x_syn(:, 1)) + eps_val);
    x_syn = x_syn .* (phase_w .* conj(phase_syn0));
    X_out = reshape(x_syn, F, T, Cuse);
end

function [X, fvec, tvec] = stft_mics_fast(x, fs, winLen, hop, nfft)
    if ndims(x) == 2, x = reshape(x, size(x,1), size(x,2), 1); end
    [Ns,C,M] = size(x);
    win = hann(winLen, 'periodic');
    nFrames = 1 + floor((Ns - winLen) / hop);
    F = nfft/2 + 1;
    X = complex(zeros(F, nFrames, C, M, 'single'));
    for m = 1:M, for c = 1:C, for n = 1:nFrames, spec = fft(single(x((1:winLen)+(n-1)*hop,c,m)) .* single(win), nfft); X(:,n,c,m) = spec(1:F); end, end, end
    fvec = (0:F-1)' * fs / nfft; tvec = ((0:nFrames-1) * hop + winLen/2) / fs;
end

function [a00, E_map, I_map, eta_map, u_map, rhoI_map, psi_map] = extract_dirac_states_foa(X, alpha, shNorm)
    [F, N, C, M] = size(X); a00 = squeeze(X(:,:,1,:));
    P_W = real(X(:,:,1,:) .* conj(X(:,:,1,:))); P_Y = real(X(:,:,2,:) .* conj(X(:,:,2,:)));
    P_Z = real(X(:,:,3,:) .* conj(X(:,:,3,:))); P_X = real(X(:,:,4,:) .* conj(X(:,:,4,:)));
    I_X = real(X(:,:,1,:) .* conj(X(:,:,4,:))); I_Y = real(X(:,:,1,:) .* conj(X(:,:,2,:))); I_Z = real(X(:,:,1,:) .* conj(X(:,:,3,:)));
    b = 1 - alpha; a = [1, -alpha];
    E_4D = max((filter(b, a, P_W, [], 2) + filter(b, a, P_Y, [], 2) + filter(b, a, P_Z, [], 2) + filter(b, a, P_X, [], 2)) / (strcmpi(string(shNorm), "SN3D")*1.5 + 1.5), 1e-8);
    I_4D = cat(3, filter(b, a, I_X, [], 2), filter(b, a, I_Y, [], 2), filter(b, a, I_Z, [], 2)) * (strcmpi(string(shNorm), "SN3D")*0.1547 + 1.1547);
    eta_4D = I_4D ./ E_4D; rho_4D = min(max(sqrt(sum(eta_4D.^2, 3)), 0), 1);
    invalid_mask = rho_4D <= 1e-8; u_4D = eta_4D ./ max(rho_4D, 1e-8);
    if any(invalid_mask, 'all')
        u1 = u_4D(:,:,1,:); u1(invalid_mask) = 1; u_4D(:,:,1,:) = u1; 
        u2 = u_4D(:,:,2,:); u2(invalid_mask) = 0; u_4D(:,:,2,:) = u2; 
        u3 = u_4D(:,:,3,:); u3(invalid_mask) = 0; u_4D(:,:,3,:) = u3;
    end
    E_map = single(reshape(E_4D, [F, N, M])); I_map = single(I_4D); eta_map = single(eta_4D);
    u_map = single(u_4D); rhoI_map = single(reshape(rho_4D, [F, N, M])); psi_map = single(reshape(1 - rho_4D, [F, N, M]));
end

function x = normalize01(x)
    xmin = min(x(:)); xmax = max(x(:)); x = (x - xmin) ./ max(xmax - xmin, 1e-8);
end


function [w, debug] = predict_weights_from_batch(params, E_known, psi_known, u_known, q_pos, freq_norm, mic_pos, CFG, nodePresent)
    B = size(E_known,1); M = size(E_known,2);
    if nargin < 9 || isempty(nodePresent), nodePresent = ones(B, M, 'like', E_known); end
    logE = log10(E_known + 1e-8) .* cast(E_known > 0, 'like', E_known);
    if CFG.GNN_CENTER_FEATURES
        activeCount = max(sum(nodePresent, 2), 1);
        meanLogE = sum(logE .* nodePresent, 2) ./ activeCount;
        meanPsi  = sum(psi_known .* nodePresent, 2) ./ activeCount;
        dlogE = (logE - meanLogE) .* nodePresent; dpsi  = (psi_known - meanPsi) .* nodePresent;
    else
        dlogE = logE .* nodePresent; dpsi = psi_known .* nodePresent;
    end
    knownFeat = cat(3, dlogE, dpsi, repmat(freq_norm, 1, M), nodePresent, zeros(B, M, 'like', E_known));
    qFeat = zeros(B, 1, CFG.GNN_NODE_FEATURE_DIM, 'like', E_known);
    qFeat(:,:,3) = reshape(freq_norm, B, 1); qFeat(:,:,5) = 1;
    [w, debug] = forward_weight_gnn(params, cat(2, knownFeat, qFeat), u_known, mic_pos, q_pos, CFG, nodePresent);
end

function [E_hat, psi_hat, u_hat] = reconstruct_dirac_from_weights(w, E_known, psi_known, u_known)
    epsv = single(1e-8); E_hat = sum(w .* E_known, 2);
    psi_hat = sum(w .* E_known .* psi_known, 2) ./ max(E_hat, epsv); 
    dirGain = w .* E_known .* (1 - psi_known);
    v = [sum(dirGain .* u_known(:,:,1), 2) sum(dirGain .* u_known(:,:,2), 2) sum(dirGain .* u_known(:,:,3), 2)];
    u_hat = v ./ sqrt(sum(v.^2, 2) + epsv);
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

function [w, idx_loc] = dirac_local_weights_delaunay(mic_pos, q_pos, fallbackK)
    activeDims = std(mic_pos,0,1) > 1e-7;
    if nnz(activeDims) < 2, activeDims(1:min(2,size(mic_pos,2))) = true; end
    P = mic_pos(:,activeDims); q = q_pos(:, activeDims); D = size(P,2);
    if size(P,1) < D + 1, [~, ord] = sort(vecnorm(P - q, 2, 2), 'ascend'); idx_loc = ord(1:min(fallbackK, numel(ord))); w = ones(numel(idx_loc),1) / numel(idx_loc); return; end
    DT = delaunayTriangulation(P); ti = pointLocation(DT, q);
    if ~isnan(ti), idx_loc = DT.ConnectivityList(ti,:); w = cartesianToBarycentric(DT, ti, q); w = max(w(:), 0); w = w / (sum(w) + 1e-12);
    else, [~, ord] = sort(vecnorm(P - q, 2, 2), 'ascend'); idx_loc = ord(1:min(fallbackK, numel(ord))); d = vecnorm(P(idx_loc,:) - q, 2, 2); w = exp(-(d.^2)/(max(median(d),1e-3)^2 + 1e-12)); w = w / (sum(w) + 1e-12); end
end

function y = sh_n3d_vec(order, u)
    x = u(1,:); yv = u(2,:); z = u(3,:); y = ones(1, numel(x), 'like', x);
    if order >= 1, y = [y; sqrt(3)*yv; sqrt(3)*z; sqrt(3)*x]; end
end

function E = batch_dirac_energy(X_foa, shNorm)
    E = max(real(abs(X_foa(:, 1)).^2 + sum(abs(X_foa(:, 2:4)).^2, 2) / (strcmpi(string(shNorm), "SN3D")*1.5 + 1.5)), 1e-8);
end

function [w, debug] = forward_weight_gnn(params, nodeFeat, u_known, mic_pos, q_pos, CFG, nodePresent)
    B = size(nodeFeat,1); Ntot = size(nodeFeat,2); L = CFG.GNN_NUM_LAYERS; M = Ntot - 1;
    posAll = cat(2, mic_pos, reshape(cast(q_pos, 'like', mic_pos), B, 1, 3));
    delta = reshape(posAll, B, 1, Ntot, 3) - reshape(posAll, B, Ntot, 1, 3);
    dist = sqrt(sum(delta.^2, 4) + 1e-8); dir_ij = delta ./ dist; 
    U_all = cat(2, u_known, zeros(B, 1, 3, 'like', u_known)); 
    Ui = repmat(reshape(U_all, B, Ntot, 1, 3), 1, 1, Ntot, 1); Uj = repmat(reshape(U_all, B, 1, Ntot, 3), 1, Ntot, 1, 1);
    edgeFeat = cat(4, dist, sum(Ui .* dir_ij, 4), sum(Uj .* dir_ij, 4), sum(Ui .* Uj, 4));
    knownValid = cast(nodePresent > 0, 'like', nodeFeat); nodeValid = cat(2, knownValid, ones(B, 1, 'like', knownValid));
    h = (nodeFeat .* reshape(nodeValid, B, Ntot, 1));
    pairMask = reshape(nodeValid, B, Ntot, 1, 1) .* reshape(nodeValid, B, 1, Ntot, 1) .* reshape(ones(Ntot, Ntot, 'single') - eye(Ntot, 'single'), 1, Ntot, Ntot, 1);
    for l = 1:L
        hi = repmat(reshape(h, B, Ntot, 1, size(h,3)), 1, 1, Ntot, 1); hj = repmat(reshape(h, B, 1, Ntot, size(h,3)), 1, Ntot, 1, 1);
        msg = leaky_relu(mlp_apply(reshape(cat(4, hi, hj, edgeFeat), [], size(hi,4)*2 + size(edgeFeat,4)), params.msg{l}, 'linear'), CFG.GNN_MSG_NEG_SLOPE);
        msg = reshape(msg, B, Ntot, Ntot, []) .* pairMask; agg = squeeze(sum(msg, 3)); if size(agg,3)==1, agg = permute(agg,[3,1,2]); end
        h = leaky_relu(mlp_apply(reshape(cat(3, h, agg), [], size(h,3) + size(agg,3)), params.upd{l}, 'linear'), CFG.GNN_MSG_NEG_SLOPE);
        h = reshape(h, B, Ntot, []) .* reshape(nodeValid, B, Ntot, 1);
    end
    edgeKnownToQ = squeeze(edgeFeat(:,1:M,end,:)); if size(edgeKnownToQ,3)==1, edgeKnownToQ = permute(edgeKnownToQ,[3,1,2]); end
    logits = mlp_apply(reshape(cat(3, h(:,1:M,:), repmat(reshape(h(:,end,:), B, 1, size(h,3)), 1, M, 1), edgeKnownToQ), [], size(h,3)*2 + size(edgeKnownToQ,3)), params.out, 'linear');
    w = (log1p(exp(-abs(reshape(logits, B, M)))) + max(reshape(logits, B, M), 0)) .* knownValid; debug.meanAggNorm = 0;
end

function y = mlp_apply(x, mlp, finalAct)
    y = max(x * mlp.W1.' + mlp.b1.', 0) + 0.10 * min(x * mlp.W1.' + mlp.b1.', 0);
    y = y * mlp.W2.' + mlp.b2.';
    if strcmpi(finalAct, 'softplus'), y = log1p(exp(-abs(y))) + max(y, 0); end
end

function y = leaky_relu(x, alpha), y = max(x, 0) + alpha * min(x, 0); end

function a = log10(b), a = log(b) / log(10); end
function a = log1p(b), a = log(b + 1); end