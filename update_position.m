function X = update_position(X_leader, X_current, a)
    r1 = rand(); r2 = rand();
    A = 2*a*r1 - a;
    C = 2*r2;
    D = abs(C*X_leader - X_current);
    X = X_leader - A*D;
end
