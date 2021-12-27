function [xs,k] = TrustNewton(x0,f,g,B,tol,maxiter,method)

    %PARAMETERS
    low_limit = 0.25;
    high_limit = 1-low_limit;
    red_factor = 0.2;
    enl_factor = 2;
    eta = 0.2;
    delta_hat = 10;
    tol = tol*norm(g(x0));
    
    %INITIALISATION
    delta = 1;
    k = 1;
    it = 0;
    x = x0;
    xs = zeros(2,maxiter);
    xs(:,1) = x0;
    
    switch method
        
        case "C"
            getTrustMin = @(d,g,B,x) cauchy(d,g,B,x);
        case "D"
            getTrustMin = @(d,g,B,x) dogleg(d,g,B,x);
        case "E"
            getTrustMin = @(d,g,B,x) SolveSubproblem(d,g,B,x);       
    end
    
    StoppingCriteria = @(x,k) norm(g(x)) > tol && k < maxiter;
    
    %repeat until a close enough solution is reached
    while StoppingCriteria(x,it)
              
        %Calculate (an approximation) of the constrained minimum problem of
        % argmin(p) f(x) + g(x)^Tp + 1/2 p^TB(x)p s.t. p in Trust Region (i.e. ||p|| < delta)
        p = getTrustMin(delta,g,B,x);
        
        
        act = f(x)-f(x+p);
        pred = -dot(g(x),p) - 0.5*dot(p,B(x)*p);
        %calculate ratio of reduction actual reduction / model reduction
        rho = act/pred;
        
        if rho < low_limit
            %m(x) is not a good enough representation of f(x) in the Trust Region:
            %shrink it so that it is
            delta = red_factor*delta;
            
        elseif rho > high_limit 
            
            %m(x) is a very good representation of f(x) in the Trust Region: 
            %probably it is so also outside, so enlarge it 
            delta = min(delta_hat,enl_factor*delta);
        end
        
        %update the value of x only if the agreement between f and m found
        %is good enough, i.e. rho > eta
        if rho > eta
            x = x+p;
            xs(:,k+1) = x;
            k = k+1;
        end
        it = it + 1;
    end
    
    xs = xs(:,1:k);
end


