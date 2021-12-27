function p = dogleg(d,g,B,x)
% DOGLEG PATH: The method is guaranteed to work with SPD B(x) matrix ONLY.
% if B(x) is not SPD, p might NOT be a descent direction
%
% INPUT:
% d: DELTA, boundary of the feasible region
% g: Gradient function
% B: Hessian function
% x: point in which the function must be calculated
%
% OUTPUT:
% p: dogleg approximation of p = argmin m(p) s.t. ||p|| <= DELTA
    
    %calculates the gradient once and forall
    g = g(x);
    B = B(x);
    
    %steepest descent path
    alpha = -dot(g,g)/dot(g,B*g);
    pu = g.*alpha;
    
    c = norm(pu); %if the steepest descent is already out of reach
    if c >= d     %use the Cauchy Point
        p = d/c*pu;
    else
        %Ordinary Newton step, since B is spd it is the global minimiser of the quadratic model function
        pb = -B\g;
        
        %if the Newton step is feasible, use the Newton step
        if norm(pb) <= d 
            p = pb;
        else
            
            %solve the equation ||p_u + t(p_b-p_u)||^2 = d^2 to find the
            %dogleg minimiser 
            a = dot(pb-pu,pb-pu);
            b = dot(pu,pb-pu);
            c = dot(pu,pu) - d^2;
            
            t = (-b + sqrt(b^2-a*c))/a;
            %return the minimiser of the dogleg path
            p = (1-t)*pu + t*pb;
        end
    end
end