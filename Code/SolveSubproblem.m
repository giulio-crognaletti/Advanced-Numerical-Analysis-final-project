function p = SolveSubproblem(d,g,B,x)
% EXACT SOLUTION APPROXIMATION of minimisation subproblem
%
% INPUT:
% d: DELTA, boundary of the feasible region
% g: Gradient function
% B: Hessian function
% x: point in which the function must be calculated
%
% OUTPUT:
% p: dogleg approximation of p = argmin m(p) s.t. ||p|| <= DELTA
    
    %calculates gradient and hessian once and for all
    g = g(x);
    B = B(x);
    
    % Tolerance of the product g^Tq, where Bq = lambda_min q
    eps = 1e-12;
    
    %possible descent direction
    p = -B\g;
    np = norm(p);
    
    [eiv,lam] = eigs(B,1,"smallestreal");
    %fprintf("min eig:%f (%f,%f)\n",lam,x(1),x(2))
    
    % either p is not feasible or lambda = 0 is not feasible (in fact, lambda > -lambda_min)
    if np >= d || lam < 0
        
        % check for the hard case 
        if  abs(dot(eiv,g)) < eps
            p = HandleHardCase(d,g,B,lam,eiv);
        else
            p = IterativeScheme(d,g,B,lam);
        end
    end
end

function p = IterativeScheme(d,g,B,eig)
% Newton iterative scheme, to be applied to the problem of finding lambda s.t. ||p|| = DELTA && (B-lambda)p = -g 
%
% Known problems: very loose enforcement of lambda_k > -eig, in general not assured. It will cause a matlab error (chol) 
    
    eps = 1e-8;                 %toleance on when to stop the newton scheme
    gc= 1e-10;                  %Guess Coeffcient: important: it is the only safeguard against lambda_k <= -eig
    lam = choose_lam(eig,gc);
    n = size(B,1);
    it = 0;                     %Number of iterations per call (check: it is typically 2/3)

    while true
        
        %fprintf("lam_%d:%f,eig:%f\n",it,lam,eig)
        R = chol(B+eye(n)*lam);

        y = -R.'\g;
        p = R\y;
        np = norm(p);

        % stopping criteria
        if abs(1/np-1/d) < eps 
            break;
        end

        q = R.'\p;
        nq = norm(q);

        lam = lam + (np/nq)^2*(np/d-1);
        it = it + 1;
    end
end

function p = HandleHardCase(d,g,B,eig,eiv)
    
    p = -(B-eye(n)*eig)\g;
    np = norm(p);
    tau = sqrt(np^2-d^2);
    p = p + tau*eiv;
    
end

function lam = choose_lam(eig,gc)
    if eig > 0
        lam = -(1-gc)*eig;
    else
        lam = -(1+gc)*eig;
    end
end
