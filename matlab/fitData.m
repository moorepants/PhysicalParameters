function [estimates,model] = fitData(t,data)
start_point = [rand(1, 3) 0.5 3.8] ;
model = @dampOscFun;
estimates = fminsearch(model, start_point);
% expfun accepts curve parameters as inputs, and outputs sse,
% the sum of squares error for A * exp(-lambda * xdata) - ydata, 
% and the FittedCurve. FMINSEARCH only needs sse, but we want to 
% plot the FittedCurve at the end.
    function [sse, FittedCurve] = dampOscFun(params)
        A = params(1);
        B = params(2);
        C = params(3);
        dampRatio = params(4);
        wo = params(5);
        FittedCurve = A + exp(-dampRatio*wo*t) .* (B*cos(wo*sqrt(1-dampRatio^2)*t) + C*sin(wo*sqrt(1-dampRatio^2)*t));
        ErrorVector = FittedCurve - data;
        sse = sum(ErrorVector .^ 2);
    end
end