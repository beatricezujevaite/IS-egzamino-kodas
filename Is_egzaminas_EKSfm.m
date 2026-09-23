clear; clc; close all;

% =========================================================================
% 1. TINKLO PARAMETRAI IR KOEFICIENTŲ APRAŠYMAS
% =========================================================================
eta = 0.05;    % Padidintas mokymosi greitis greitesniam konvergavimui
N = 500;       % Laiko žingsniai vienoje epochoje
epochs = 1000; % Epochų skaičius

% Svorių inicializavimas maža atsitiktine reišme apie 0 (tarp -0.5 ir 0.5)
w11_1 = (rand()-0.5); w12_1 = (rand()-0.5); w11_r = (rand()-0.5); b1_1 = (rand()-0.5);
w21_1 = (rand()-0.5); w22_1 = (rand()-0.5); b2_1  = (rand()-0.5);
w31_1 = (rand()-0.5); w32_1 = (rand()-0.5); w32_r = (rand()-0.5); b3_1 = (rand()-0.5);

w11_2 = (rand()-0.5); w12_2 = (rand()-0.5); w13_2 = (rand()-0.5); b1_2 = (rand()-0.5);
w21_2 = (rand()-0.5); w22_2 = (rand()-0.5); w23_2 = (rand()-0.5); b2_2 = (rand()-0.5);

% =========================================================================
% 2. DUOMENŲ PARUOŠIMAS (Sušvelnintas dažnis tolygiam mokymuisi)
% =========================================================================
t = 1:N;
step = 0.1; % Sušvelnintas laiko žingsnis

X1 = sin(t * step);
X2 = cos(t * step);
D1 = sin(t * step + 0.5);
D2 = cos(t * step + 0.5);

Y1 = zeros(1, N); Y2 = zeros(1, N); E = zeros(1, N);

% =========================================================================
% 3. MOKYMOSI CIKLAS SU EPOCHOMIS
% =========================================================================
for ep = 1:epochs
    y1_prev = 0;
    y2_prev = 0;
    
    for n = 1:N
        x1 = X1(n); x2 = X2(n);
        d1 = D1(n); d2 = D2(n);
        
        % Forward pass
        v1_1 = w11_1*x1 + w12_1*x2 + w11_r*y1_prev + b1_1;
        v2_1 = w21_1*x1 + w22_1*x2 + b2_1;
        v3_1 = w31_1*x1 + w32_1*x2 + w32_r*y2_prev + b3_1;
        
        h1 = 1 / (1 + exp(-v1_1));
        h2 = 1 / (1 + exp(-v2_1));
        h3 = 1 / (1 + exp(-v3_1));
        
        y1 = w11_2*h1 + w12_2*h2 + w13_2*h3 + b1_2;
        y2 = w21_2*h1 + w22_2*h2 + w23_2*h3 + b2_2;
        
        % Backpropagation
        e1 = d1 - y1;
        e2 = d2 - y2;
        
        delta1_2 = e1;
        delta2_2 = e2;
        
        delta1_1 = (delta1_2 * w11_2 + delta2_2 * w21_2) * h1 * (1 - h1);
        delta2_1 = (delta1_2 * w12_2 + delta2_2 * w22_2) * h2 * (1 - h2);
        delta3_1 = (delta1_2 * w13_2 + delta2_2 * w23_2) * h3 * (1 - h3);
        
        % Svorių atnaujinimai
        w11_2 = w11_2 + eta * delta1_2 * h1;
        w12_2 = w12_2 + eta * delta1_2 * h2;
        w13_2 = w13_2 + eta * delta1_2 * h3;
        b1_2  = b1_2  + eta * delta1_2;
        
        w21_2 = w21_2 + eta * delta2_2 * h1;
        w22_2 = w22_2 + eta * delta2_2 * h2;
        w23_2 = w23_2 + eta * delta2_2 * h3;
        b2_2  = b2_2  + eta * delta2_2;
        
        w11_1 = w11_1 + eta * delta1_1 * x1;
        w12_1 = w12_1 + eta * delta1_1 * x2;
        w11_r = w11_r + eta * delta1_1 * y1_prev;
        b1_1  = b1_1  + eta * delta1_1;
        
        w21_1 = w21_1 + eta * delta2_1 * x1;
        w22_1 = w22_1 + eta * delta2_1 * x2;
        b2_1  = b2_1  + eta * delta2_1;
        
        w31_1 = w31_1 + eta * delta3_1 * x1;
        w32_1 = w32_1 + eta * delta3_1 * x2;
        w32_r = w32_r + eta * delta3_1 * y2_prev;
        b3_1  = b3_1  + eta * delta3_1;
        
        y1_prev = y1;
        y2_prev = y2;
        
        if ep == epochs
            Y1(n) = y1; Y2(n) = y2;
            E(n) = 0.5 * (e1^2 + e2^2);
        end
    end
end

% =========================================================================
% 4. TESTAVIMAS (SU NAUJAIS DUOMENIMIS)
% =========================================================================
N_test = 50;
t_test = (N+1):(N+N_test);

X1_test = sin(t_test * step);
X2_test = cos(t_test * step);
D1_test = sin(t_test * step + 0.5);
D2_test = cos(t_test * step + 0.5);

Y1_test = zeros(1, N_test); Y2_test = zeros(1, N_test); E_test = zeros(1, N_test);

y1_prev_test = Y1(end);
y2_prev_test = Y2(end);

for n = 1:N_test
    x1 = X1_test(n); x2 = X2_test(n);
    d1 = D1_test(n); d2 = D2_test(n);
    
    v1_1 = w11_1*x1 + w12_1*x2 + w11_r*y1_prev_test + b1_1;
    v2_1 = w21_1*x1 + w22_1*x2 + b2_1;
    v3_1 = w31_1*x1 + w32_1*x2 + w32_r*y2_prev_test + b3_1;
    
    h1 = 1 / (1 + exp(-v1_1));
    h2 = 1 / (1 + exp(-v2_1));
    h3 = 1 / (1 + exp(-v3_1));
    
    y1 = w11_2*h1 + w12_2*h2 + w13_2*h3 + b1_2;
    y2 = w21_2*h1 + w22_2*h2 + w23_2*h3 + b2_2;
    
    Y1_test(n) = y1; Y2_test(n) = y2;
    E_test(n)  = 0.5 * ((d1 - y1)^2 + (d2 - y2)^2);
    
    y1_prev_test = y1;
    y2_prev_test = y2;
end

% =========================================================================
% 5. REZULTATŲ VAIZDAVIMAS
% =========================================================================
figure;
subplot(2,1,1);
plot(t_test, D1_test, 'r--', t_test, Y1_test, 'b-', 'LineWidth', 1.5); hold on;
plot(t_test, D2_test, 'm--', t_test, Y2_test, 'g-', 'LineWidth', 1.5);
grid on; legend('Tikslinis d_1', 'Išbandytas y_1', 'Tikslinis d_2', 'Išbandytas y_2');
xlabel('Laiko žingsnis (n)'); ylabel('Amplitudė');
title('Tinklo TESTAVIMO rezultatai');

subplot(2,1,2);
plot(t_test, E_test, 'k-', 'LineWidth', 1.5);
grid on; xlabel('Laiko žingsnis (n)'); ylabel('Klaida E(n)');
title('Testavimo klaida per laiko žingsnius');