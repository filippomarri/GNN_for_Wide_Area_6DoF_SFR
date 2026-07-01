clc;
clearvars;
close all;

% Definisci i nomi dei file
file_sources = '/Users/filippo/Documents/PoliMi/Altro/Dataset Tesi/arvedi_auditorium_dataset/pos_sources.csv';
file_receivers = '/Users/filippo/Documents/PoliMi/Altro/Dataset Tesi/arvedi_auditorium_dataset/pos_receivers.csv';

% Leggi i file CSV (specificando il delimitatore ';')
sources = readtable(file_sources, 'Delimiter', ';');
receivers = readtable(file_receivers, 'Delimiter', ';');

% Estrai le coordinate e le etichette delle sorgenti
src_x = sources.x;
src_y = sources.y;
src_z = sources.z;
src_labels = sources.source; 

% Estrai le coordinate centrali e le etichette dei ricevitori
rec_x = receivers.C_x;
rec_y = receivers.C_y;
rec_z = receivers.C_z;
rec_labels = receivers.mic; % Colonna 'mic' per i nomi dei ricevitori

% --- NUOVA SEZIONE: Definizione posizioni selezionate ---
% Inserisci qui le etichette dei ricevitori da evidenziare
selected_pos = {'A413', 'A302', 'A505', 'A102', 'A109', 'A207'}; 

% Trova quali righe corrispondono alle posizioni selezionate
idx_selected = ismember(string(rec_labels), selected_pos);
idx_others = ~idx_selected; % Tutti gli altri ricevitori

% Crea la figura per la mappa 3D
figure('Name', 'Mappa Sorgenti e Ricevitori', 'NumberTitle', 'off');
hold on;
grid on;

% 1. Disegna i ricevitori standard (NON selezionati) in blu
scatter3(rec_x(idx_others), rec_y(idx_others), rec_z(idx_others), ...
    50, 'b', 'filled', 'DisplayName', 'Ricevitori');

% 2. Disegna i ricevitori SELEZIONATI in giallo (con bordo nero per visibilità)
scatter3(rec_x(idx_selected), rec_y(idx_selected), rec_z(idx_selected), ...
    80, 'y', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Ricevitori Selezionati');

% Aggiungi il nome dei ricevitori selezionati vicino ai punti gialli
% find(idx_selected) restituisce gli indici esatti (le righe) da etichettare
idx_list = find(idx_selected)';
for i = idx_list
    text(rec_x(i), rec_y(i), rec_z(i) + 0.08, char(rec_labels(i)), ...
        'FontSize', 11, 'Color', 'k', 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 3. Disegna le sorgenti (triangoli rossi)
scatter3(src_x, src_y, src_z, 100, 'r', '^', 'filled', 'DisplayName', 'Sorgenti');

% Aggiungi il nome della sorgente (es. S0, S1, ecc.)
for i = 1:height(sources)
    text(src_x(i), src_y(i), src_z(i) + 0.08, char(src_labels(i)), ...
        'FontSize', 10, 'Color', 'r', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end

% Impostazioni estetiche del grafico
xlabel('Asse X [m]');
ylabel('Asse Y [m]');
zlabel('Asse Z [m]');
title('Mappa 3D con Posizioni Selezionate Evidenziate');
legend('Location', 'best');
view(3); % Imposta la vista 3D
axis equal; % Proporzioni reali
hold off;