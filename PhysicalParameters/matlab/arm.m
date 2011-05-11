function zero = arm(d,x19,x21,z19,z21,lla,lua)
zero=(z21-z19)^2+(x19-x21)^2+lla^2-2*(z21-z19)*(lla^2-d.^2).^0.5-2*(x19-x21)*d-lua^2;