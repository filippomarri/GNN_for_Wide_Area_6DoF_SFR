function Y = compute_real_sh_matrix(N, azi, colat)
% COMPUTE_REAL_SH_MATRIX Calcola le basi delle armoniche sferiche (SH) reali.
%
% INPUT:
%   N     - Ordine massimo delle armoniche sferiche
%   azi   - Vettore (colonna) degli angoli di azimut in radianti
%   colat - Vettore (colonna) degli angoli di colatitudine in radianti (0 = zenit)
%
% OUTPUT:
%   Y     - Matrice di dimensione [n_points x (N+1)^2] contenente i valori
%           delle armoniche sferiche per ogni punto.

    n_points = length(azi);
    n_coeffs = (N + 1)^2;
    Y = zeros(n_points, n_coeffs);
    
    idx = 1;
    for n = 0:N
        % Calcola le funzioni associate di Legendre per l'ordine n
        P = legendre(n, cos(colat')); 
        
        for m = -n:n
            abs_m = abs(m);
            P_nm = P(abs_m + 1, :)'; 
            
            % Termine di fase di Condon-Shortley
            P_nm = P_nm * (-1)^abs_m;
            
            % Fattore di normalizzazione (Real Spherical Harmonics)
            norm_factor = sqrt(((2*n + 1) / (4*pi)) * factorial(n - abs_m) / factorial(n + abs_m));
            
            if m > 0
                val = norm_factor * P_nm .* cos(m * azi) * sqrt(2);
            elseif m < 0
                val = norm_factor * P_nm .* sin(abs_m * azi) * sqrt(2);
            else
                val = norm_factor * P_nm;
            end
            
            Y(:, idx) = val;
            idx = idx + 1;
        end
    end
end