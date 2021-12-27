function p = cauchy(d,g,B,x)

    %calculates the gradient once and forall
    g = g(x);
    B = B(x);
    
    dt = dot(g,B*g);
    if dt <= 0
        t = d/norm(g);
    else
        t = min(d/norm(g),dot(g,g)/dt);
    end
    
    p = -t*g;
end