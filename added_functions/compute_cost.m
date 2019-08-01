function [costs] = compute_cost(target, states, stddev)

[N, ~] = size(states);

costs = nan(N,1);

for n = 1:N
    
    sqdist = sum( (states(n,:)' - target).^2);
    costs(n,1) = 1 - exp(-(1 / (2*stddev^2)) * sqdist);
end

end

