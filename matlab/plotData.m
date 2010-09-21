plot(t,data,'*')
hold on
[sse, FittedCurve] = model(estimates);
plot(t, FittedCurve, 'r')
xlabel('xdata')
ylabel('f(estimates,xdata)')
title(['Fitting to function ', func2str(model)]);
legend('data', ['fit using ', func2str(model)])
hold off