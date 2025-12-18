function Phi = generate_regressors(x, m)
    x1 = x(:,1);
    x2 = x(:,2);
    Phi = [];

    for i = 0:m
        for j = 0:m
            if(i+j) <= m
                Phi = [Phi, (x1.^i).*(x2.^j)];
            end
        end
    end
end