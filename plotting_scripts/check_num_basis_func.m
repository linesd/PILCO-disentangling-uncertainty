clearvars;
close all

%% Load the cartpole data

% load actual state action transitions (needs to be the same episode as trajectories)
X_data = load('data_3/X_data_15.mat');
X = X_data.X_tr;
Y_data = load('data_3/Y_data_15.mat');
Y = Y_data.Y_tr;

[N, Dx] = size(X);
[~, Dy] = size(Y);

% get a training and test set
t1 = floor(0.8*N);
ind = randperm(N);
ind1 = ind(1:t1);
ind2 = ind(t1+1:end);
X_tr = X(ind1,:);
Y_tr = Y(ind1,:);
X_te = X(ind2,:);
Y_te = Y(ind2,:);

%% Try different number of basis

basis_functions = 20:100:1020;
num_basis = length(basis_functions);
iteropt = -1000;

NMSE1 = nan(num_basis,Dy);
NMLP1 = nan(num_basis,Dy);
NMSE2 = nan(num_basis,Dy);
NMLP2 = nan(num_basis,Dy);

for m = 1:num_basis
    for d = 1:Dy
        
        NMSE1_accum=0;
        NMLP1_accum=0;
        NMSE2_accum=0;
        NMLP2_accum=0;
        
        for i = 1:10 %average over 10
            loghyper = rand(Dx+2,1); % initialise hyperparams
            
            % train model with optimised spectral features
            [NMSE1_, mu1, S12, NMLP1_, loghyper1, convergence1] = ...
                ssgpr_ui(X_tr, Y_tr(:,d), X_te, Y_te(:,d), m, iteropt, loghyper);

            % train model with fixed spectral features
            [NMSE2_, mu2, S22, NMLP2_, loghyper2, convergence2] = ...
                ssgprfixed_ui(X_tr, Y_tr(:,d), X_te, Y_te(:,d), m, iteropt, loghyper);

            % accumulate for average
            NMSE1_accum=NMSE1_accum + NMSE1_;
            NMLP1_accum=NMLP1_accum + NMLP1_;
            NMSE2_accum=NMSE2_accum + NMSE2_;
            NMLP2_accum=NMLP2_accum + NMLP2_;
            
        end
        
        NMSE1(m,d) = NMSE1_accum/10;
        NMLP1(m,d) = NMLP1_accum/10;
        NMSE2(m,d) = NMSE2_accum/10;
        NMLP2(m,d) = NMLP2_accum/10;
    end
end

%% plot data

figure(1)
colors = ['r', 'g', 'b', 'k'];
shapes = ['o', 's', 'v', '+'];

for d=1:Dy
%     yyaxis left
plot(basis_functions, NMSE1(:,d), 'color', colors(d), 'marker', shapes(d), 'linestyle', '--');
hold on
plot(basis_functions, NMSE2(:,d), 'color', colors(d), 'marker', shapes(d));
end
xlabel("Number of basis functions")
ylabel("Normalised mean squared error")
legend('Dim1', 'Dim1 - fixed', 'Dim2', 'Dim2 - fixed', 'Dim3', 'Dim3 - fixed', 'Dim4', 'Dim4 - fixed')

%     yyaxis right
%     plot(x_plot, NMLP1, '-s');
%     hold on
%     plot(x_plot, NMLP2, '-<');
%     ylabel("Mean negative log-probability")
% end
grid on
set(gca,'GridLineStyle','--')


