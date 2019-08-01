function states_actions = get_policy(policy, states)

[N, dim] = size(states);
U = zeros(N,1);

for nn = 1:N
    U(nn, 1) = policy.fcn(policy, states(nn,:)', zeros(dim));
end

states_actions = [states U];
end