function zero = leg(d,x5,x13,z5,z13,lth,lc)
zero=lth^2-lc^2-(z13-z5)^2-(x5-x13)^2+2*(z13-z5)*(lc^2-d.^2).^0.5-2*(x5-x13).*d;