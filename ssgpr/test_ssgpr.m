clearvars
close all
N = 20;
a=-10;
b=10;
std = 0.5;
train_x = a + (b-a).*rand(N,1);%reshape(linspace(-10,10,N), [N 1]); % a + (b-a).*rand(N,1)."
fx = 3*sin(0.5*train_x) +3*cos(0.6*sqrt(2)*train_x);
train_y = std.*randn(N,1) + fx; 

test_x = reshape(linspace(-10,10,100),[100 1]);
test_y = 3*sin(0.5*test_x) +3*cos(0.6*sqrt(2)*test_x);
m = 30;

% loghyper = [1.56013833751907;1.77447635997167;-0.723561501187600;3.80999241824925e-10;-2.36548018751266;-3.42551483560806e-10;4.04395037551720;1.02149797081401e-09;5.82934079657486e-11;-3.53572564768707e-10;-4.34378315730031e-11;-4.62597393959075e-10;-1.34387161607831e-10];

[NMSE, mu, S2, NMLP, loghyper, convergence] = ssgprfixed_ui(train_x, train_y, test_x, test_y, m);

X_te = linspace(a,b,100);
[mu_p, cov_p, phistar] = ssgprfixed(loghyper, train_x, train_y, X_te', true);

weights = mvnrnd(mu_p', cov_p, 5);

post = phistar * weights';

plot(test_x, mu)
hold on
plot(train_x, train_y,'*')
plot(test_x, mu+2*sqrt(S2))
plot(test_x, mu-2*sqrt(S2))
plot(X_te, post, '--')
grid on