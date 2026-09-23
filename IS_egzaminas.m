clear; clc; close all;

% =========================================================================
% 1. TINKLO PARAMETRAI IR KOEFICIENTŲ (SVORIŲ) APRAŠYMAS
% =========================================================================
eta = 0.01; % Mokymosi žingsnis (learning rate)
N = 100;    % Laiko žingsnių skaičius

% --- PASLĖPTASIS SLUOKSNIS (1 pakopa) ---
% 1-asis neuronas (gauna x1, x2 ir y1(n-1))
w11_1 = rand(1); % iš x1 į 1-ąjį neuroną
w12_1 = rand(1); % iš x2 į 1-ąjį neuroną
w11_r = rand(1); % iš y1(n-1) į 1-ąjį neuroną (grįžtamasis ryšys)
b1_1  = rand(1); % 1-ojo neurono bias

% 2-asis neuronas (gauna tik x1 ir x2)
w21_1 = rand(1); % iš x1 į 2-ąjį neuroną
w22_1 = rand(1); % iš x2 į 2-ąjį neuroną
b2_1  = rand(1); % 2-ojo neurono bias

% 3-iasis neuronas (gauna x1, x2 ir y2(n-1))
w31_1 = rand(1); % iš x1 į 3-iąjį neuroną
w32_1 = rand(1); % iš x2 į 3-iąjį neuroną
w32_r = rand(1); % iš y2(n-1) į 3-iąjį neuroną (grįžtamasis ryšys)
b3_1  = rand(1); % 3-iojo neurono bias

% --- IŠĖJIMO SLUOKSNIS (2 pakopa) ---
% 1-asis išėjimo neuronas y1(n)
w11_2 = rand(1); % iš h1 į y1
w12_2 = rand(1); % iš h2 į y1
w13_2 = rand(1); % iš h3 į y1
b1_2  = rand(1); % y1 išėjimo bias

% 2-asis išėjimo neuronas y2(n)
w21_2 = rand(1); % iš h1 į y2
w22_2 = rand(1); % iš h2 į y2
w23_2 = rand(1); % iš h3 į y2
b2_2  = rand(1); % y2 išėjimo bias

% =========================================================================
% 2. DUOMENŲ IR KINTAMŲJŲ PARUOŠIMAS
% =========================================================================
t = 1:N;

% Įėjimo signalai x1(n) ir x2(n)
X1 = sin(t);
X2 = cos(t);

% Tiksliniai (norimi) išėjimo signalai d1(n) ir d2(n)
D1 = sin(t + 0.5);
D2 = cos(t + 0.5);

% Masyvai rezultatams ir klaidoms saugoti
Y1 = zeros(1, N);
Y2 = zeros(1, N);
E  = zeros(1, N);

% Pradinės z^-1 vėlinimo reikšmės: y1(0) = 0, y2(0) = 0
y1_prev = 0;
y2_prev = 0;

% =========================================================================
% 3. CIKLAS PER LAIKO ŽINGSNIUS (FORWARD + BACKPROPAGATION)
% =========================================================================
for n = 1:N
    % Einamieji įėjimai ir norimi išėjimai
    x1 = X1(n);
    x2 = X2(n);
    d1 = D1(n);
    d2 = D2(n);
    
    % ---------------------------------------------------------------------
    % A. TINKLO ATSAKO SKAIČIAVIMAS (FORWARD PASS)
    % ---------------------------------------------------------------------
    % Paslėptojo sluoksnio įėjimo sumos (v)
    v1_1 = w11_1*x1 + w12_1*x2 + w11_r*y1_prev + b1_1;
    v2_1 = w21_1*x1 + w22_1*x2 + b2_1;
    v3_1 = w31_1*x1 + w32_1*x2 + w32_r*y2_prev + b3_1;
    
    % Paslėptojo sluoksnio aktyvavimas (sigmoidinė funkcija phi1)
    h1 = 1 / (1 + exp(-v1_1));
    h2 = 1 / (1 + exp(-v2_1));
    h3 = 1 / (1 + exp(-v3_1));
    
    % Išėjimo sluoksnis (linijinė funkcija phi2, kur phi2(v) = v)
    y1 = w11_2*h1 + w12_2*h2 + w13_2*h3 + b1_2;
    y2 = w21_2*h1 + w22_2*h2 + w23_2*h3 + b2_2;
    
    % Išsaugome atsakus
    Y1(n) = y1;
    Y2(n) = y2;
    
    % Klaidos skaičiavimas
    e1 = d1 - y1;
    e2 = d2 - y2;
    E(n) = 0.5 * (e1^2 + e2^2);
    
    % ---------------------------------------------------------------------
    % B. KOEFICIENTŲ ATNAUJINIMAS (BACKPROPAGATION)
    % ---------------------------------------------------------------------
    % 1. Išėjimo sluoksnio gradientai (delta2)
    delta1_2 = e1; % linijinei funkcijai išvestinė lygi 1
    delta2_2 = e2;
    
    % 2. Paslėptojo sluoksnio gradientai (delta1)
    % Sigmoidos išvestinė: h * (1 - h)
    delta1_1 = (delta1_2 * w11_2 + delta2_2 * w21_2) * h1 * (1 - h1);
    delta2_1 = (delta1_2 * w12_2 + delta2_2 * w22_2) * h2 * (1 - h2);
    delta3_1 = (delta1_2 * w13_2 + delta2_2 * w23_2) * h3 * (1 - h3);
    
    % 3. Svorių ir bias atnaujinimas: Išėjimo sluoksnis
    w11_2 = w11_2 + eta * delta1_2 * h1;
    w12_2 = w12_2 + eta * delta1_2 * h2;
    w13_2 = w13_2 + eta * delta1_2 * h3;
    b1_2  = b1_2  + eta * delta1_2;
    
    w21_2 = w21_2 + eta * delta2_2 * h1;
    w22_2 = w22_2 + eta * delta2_2 * h2;
    w23_2 = w23_2 + eta * delta2_2 * h3;
    b2_2  = b2_2  + eta * delta2_2;
    
    % 4. Svorių ir bias atnaujinimas: Paslėptasis sluoksnis
    w11_1 = w11_1 + eta * delta1_1 * x1;
    w12_1 = w12_1 + eta * delta1_1 * x2;
    w11_r = w11_r + eta * delta1_1 * y1_prev; % Grįžtamojo ryšio svoris y1(n-1)
    b1_1  = b1_1  + eta * delta1_1;
    
    w21_1 = w21_1 + eta * delta2_1 * x1;
    w22_1 = w22_1 + eta * delta2_1 * x2;
    b2_1  = b2_1  + eta * delta2_1;
    
    w31_1 = w31_1 + eta * delta3_1 * x1;
    w32_1 = w32_1 + eta * delta3_1 * x2;
    w32_r = w32_r + eta * delta3_1 * y2_prev; % Grįžtamojo ryšio svoris y2(n-1)
    b3_1  = b3_1  + eta * delta3_1;
    
    % ---------------------------------------------------------------------
    % C. VĖLINIMO ATNAUJINIMAS z^-1
    % ---------------------------------------------------------------------
    y1_prev = y1;
    y2_prev = y2;
end

% =========================================================================
% 4. REZULTATŲ VAIZDAVIMAS
% =========================================================================
figure;

subplot(2,1,1);
plot(t, D1, 'r--', t, Y1, 'b-', 'LineWidth', 1.5);
hold on;
plot(t, D2, 'm--', t, Y2, 'g-', 'LineWidth', 1.5);
grid on;
legend('Norimas d_1(n)', 'Gautas y_1(n)', 'Norimas d_2(n)', 'Gautas y_2(n)');
xlabel('Laiko žingsnis (n)');
ylabel('Amplitudė');
title('Rekurentinio tinklo išėjimų y_1(n) ir y_2(n) mokymasis');

subplot(2,1,2);
plot(t, E, 'k-', 'LineWidth', 1.5);
grid on;
xlabel('Laiko žingsnis (n)');
ylabel('Klaida E(n)');
title('Mokymosi klaidos E(n) kitimas');