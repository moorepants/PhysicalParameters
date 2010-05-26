%---------------------------------------------------------------------%
%    FILE: BRR_05.AL
%    DATE: DECEMBER 6,2007
%  AUTHOR: JASON MOORE
% PROBLEM: GENERATES THE EQUATIONS OF MOTION FOR A RIGID RIDER
%          NO-HANDS BICYCLE MADE UP OF FOUR RIGID BODIES.
%   NOTES: ADDS THE BENCHMARK PARAMETERS IN FROM THE GET GO. FIRST
%          MODEL TO MATCH THE BENCHMARK!
%---------------------------------------------------------------------%
%         DEFAULT SETTINGS
%---------------------------------------------------------------------%

AUTOZ ON
AUTORHS OFF
OVERWRITE ALL

%---------------------------------------------------------------------%
%         NEWTONIAN, BODIES, FRAMES, PARTICLES, POINTS
%---------------------------------------------------------------------%

% DECLARE THE INERTIAL REFERENCE FRAME

NEWTONIAN N

% DECLARE THREE INTERMEDIATE FRAMES
% A: YAW FRAME
% B: ROLL FRAME
% E: HEAD TUBE ANGLE FRAME

FRAMES A,B,E

% DECLARE SIX BODIES
% C: REAR WHEEL
% D: FRAME
% F: FORK
% G: FRONT WHEEL

BODIES C,D,F,G

% DECLARE FIVE POINTS
% NC: REAR CONTACT POINT ON GROUND
% CN: REAR CONTACT POINT ON WHEEL
% NG: FRONT CONTACT POINT ON GROUND
% GN: FRONT CONTACT POINT ON WHEEL

POINTS NC,CN,NG,GN

%---------------------------------------------------------------------%
%         CONSTANTS AND VARIABLES
%---------------------------------------------------------------------%

% W:      WHEELBASE                         [M]
% C:      TRAIL                             [M]
% LAMBDA: STEER AXIS TILT                   [RAD]
% G:      GRAVITY                           [N/KG]
% RR:     REAR WHEEL RADIUS                 [M]
% M_R:    REAR WHEEL MASS                   [KG]
% IRXX:   REAR WHEEL MASS MOMENT OF INERTIA [KG*M^2]
% IRYY:   REAR WHEEL MASS MOMENT OF INERTIA [KG*M^2]
% XB:     REAR BODY CENTER OF MASS LOCATION [M]
% ZB:     REAR BODY CENTER OF MASS LOCATION [M]
% M_B:    REAR BODY MASS                    [KG]
% IBXX:   REAR BODY MASS MOMENT OF INERTIA
% IBYY:
% IBZZ:
% IBXZ:
% XH:
% ZH:
% M_H:
% IHXX:   REAR BODY MASS MOMENT OF INERTIA
% IHYY:
% IHZZ:
% IHXZ:
% RF:
% M_F:
% IFXX:
% IFYY:

CONSTANTS W,C,LAMBDA,G,V
CONSTANTS RR,M_R,IRXX,IRYY
CONSTANTS XB,ZB,M_B,IBXX,IBYY,IBZZ,IBXZ
CONSTANTS XH,ZH,M_H,IHXX,IHYY,IHZZ,IHXZ
CONSTANTS RF,M_F,IFXX,IFYY

% CONVERT THE BENCHMARK CONSTANTS TO MY CONSTANTS
% RF: RADIUS OF FRONT WHEEL
% RR: RADIUS OF REAR WHEEL
% D1: THE PERPENDICULAR DISTANCE FROM THE HEAD TUBE AXIS TO THE CENTER
%     OF THE REAR WHEEL
% D3: THE PERPENDICULAR DISTANCE FROM THE HEAD TUBE AXIS TO THE CENTER
%     OF THE FRONT WHEEL
% L1: THE DISTANCE IN THE D1> DIRECTION FROM THE CENTER OF THE REAR
%     WHEEL TO THE FRAME CENTER OF MASS
% L2: THE DISTANCE IN THE D3> DIRECTION FROM THE CENTER OF THE REAR
%     WHEEL TO THE FRAME CENTER OF MASS
% L3: THE DISTANCE IN THE F1> DIRECTION FROM THE STEER POINT TO THE
%     CENTER OF MASS OF THE FORK
% L4: THE DISTANCE IN THE F3> DIRECTION FROM THE STEER POINT TO THE
%     CENTER OF MASS OF THE FORK

D1                              =  COS(LAMBDA)*(C+W-RR*TAN(LAMBDA))
D3                              = -COS(LAMBDA)*(C-RF*TAN(LAMBDA))
IC11                            =  IRxx
IC22                            =  IRyy
IC33                            =  IRxx
ID11                            =  IBxx
ID12                            =  0.0
ID22                            =  IByy
ID23                            =  0.0
ID31                            =  IBxz
ID33                            =  IBzz
IH = [IHXX,0,IHXZ;0,IHYY,0;IHXZ,0,IHZZ]
CF = [COS(LAMBDA),0,-SIN(LAMBDA);0,1,0;SIN(LAMBDA),0,COS(LAMBDA)]
IHROT                          =  CF*IH*TRANSPOSE(CF)
IF11                            =  IHROT[1,1]
IF12                            =  IHROT[1,2]
IF22                            =  IHROT[2,2]
IF23                            =  IHROT[2,3]
IF31                            =  IHROT[3,1]
IF33                            =  IHROT[3,3]
IG11                            =  IFxx
IG22                            =  IFyy
IG33                            =  IFxx
L1                              =  XB
L2                              =  RR + ZB
L3                              =  COS(LAMBDA)*XH-SIN(LAMBDA)*ZH-C*COS(LAMBDA)-W*COS(LAMBDA)
L4                              =  RR*COS(LAMBDA) + XH*SIN(LAMBDA) + ZH*COS(LAMBDA)
MC                              =  M_R
MD                              =  M_B
MF                              =  M_H
MG                              =  M_F

% DECLARE THE GENERALIZED COORDINATES
% Q1:  PERPENDICULAR DISTANCE FROM THE N2> AXIS TO THE REAR CONTACT
%      POINT IN THE GROUND PLANE
% Q2:  PERPENDICULAR DISTANCE FROM THE N1> AXIS TO THE REAR CONTACT
%      POINT IN THE GROUND PLANE
% Q3:  FRAME YAW ANGLE
% Q4:  FRAME ROLL ANGLE
% Q5:  REAR WHEEL ROTATION ANGLE
% Q6:  FRAME PITCH ANGLE
% Q7:  STEERING ROTATION ANGLE
% Q8:  FRONT WHEEL ROTATION ANGLE

VARIABLES Q{8}'

%---------------------------------------------------------------------%
%         GENERALIZED SPEEDS
%---------------------------------------------------------------------%

MOTIONVARIABLES' U{8}'

%---------------------------------------------------------------------%
%         MASS AND INERTIA PROPERTIES
%---------------------------------------------------------------------%

MASS C=MC,D=MD,F=MF,G=MG
INERTIA C,IC11,IC22,IC33
INERTIA D,ID11,ID22,ID33,ID12,ID23,ID31
INERTIA F,IF11,IF22,IF33,IF12,IF23,IF31
INERTIA G,IG11,IG22,IG33

%---------------------------------------------------------------------%
%         ANGULAR RELATIONSHIPS                                       %
%---------------------------------------------------------------------%

% FRAME YAW

SIMPROT(N,A,3,Q3)

% FRAME ROLL

SIMPROT(A,B,1,Q4)

% REAR WHEEL ROTATION

SIMPROT(B,C,2,Q5)

% FRAME PITCH

SIMPROT(B,D,2,Q6)

% HEAD TUBE ANGLE

SIMPROT(D,E,2,LAMBDA)

% STEERING ANGLE

SIMPROT(E,F,3,Q7)

% FRONT WHEEL ROTATION

SIMPROT(F,G,2,Q8)
PAUSE
%---------------------------------------------------------------------%
%         POSITION VECTORS
%---------------------------------------------------------------------%

% LOCATE THE CENTER OF MASS FOR EACH BODY

P_NO_CO>=Q1*N1>+Q2*N2>-RR*B3>

P_CO_DO>=L1*D1>+L2*D3>

P_CO_FO>=D1*E1>+L3*F1>+L4*F3>

P_CO_GO>=D1*E1>+(RR+D1*SIN(LAMBDA)-RF+D3*SIN(LAMBDA))/COS(LAMBDA)*F3>+D3*F1>

% LOCATE THE GROUND CONTACT POINTS

P_CO_CN>=RR*B3>

P_CN_NC>=0>

P_GO_GN>=RF*UNITVEC(N3>-DOT(F2>,N3>)*F2>)

P_GN_NG>=0>

%---------------------------------------------------------------------%
%         DEFINE THE GENERALIZED SPEEDS
%---------------------------------------------------------------------%

Q1'=U1
Q2'=U2
Q3'=U3
Q4'=U4
Q5'=U5
Q6'=U6
Q7'=U7
Q8'=U8

%---------------------------------------------------------------------%
%         ANGULAR VELOCITIES
%---------------------------------------------------------------------%

ANGVEL(N,A)
ANGVEL(N,B)
ANGVEL(N,C)
ANGVEL(N,D)
ANGVEL(N,E)
ANGVEL(N,F)
ANGVEL(N,G)

%---------------------------------------------------------------------%
%         VELOCITIES
%---------------------------------------------------------------------%

V_CO_N>=DT(P_NO_CO>,N)
V_DO_N>=DT(P_NO_DO>,N)
V_FO_N>=DT(P_NO_FO>,N)
V_GO_N>=DT(P_NO_GO>,N)

V2PTS(N,C,CO,CN)
V2PTS(N,G,GO,GN)

%---------------------------------------------------------------------%
%         DEFINE THE CONFIGURATION CONSTRAINT
%---------------------------------------------------------------------%

% SET THE N3> COMPONENT OF P_NC_NG> EQUAL TO ZERO
PZERO=DOT(P_NC_NG>,N3>)

%---------------------------------------------------------------------%
%         MOTION CONSTRAINTS
%---------------------------------------------------------------------%

% DUE TO THE ASSUMPTIONS OF NO SIDE SLIP AND NO SLIP ROLLING THE
% VELOCITIES OF THE FRONT AND REAR WHEEL CONTACT POINTS, CN AND GN,
% CANNOT HAVE COMPONENTS OF VELOCITY IN THE GROUND PLANE

DEPENDENT[1]=DOT(V_CN_N>,N1>)
DEPENDENT[2]=DOT(V_CN_N>,N2>)
DEPENDENT[3]=DOT(V_GN_N>,N1>)
DEPENDENT[4]=DOT(V_GN_N>,N2>)
DEPENDENT[5]=DT(PZERO)

% THE REAR WHEEL ANGULAR SPEED, U5, THE ROLL RATE, U4, AND THE
% STEERING RATE, U7, ARE TAKEN TO BE THE INDEPENDENT
% GENERALIZED SPEEDS

CONSTRAIN(DEPENDENT[U1,U2,U3,U6,U8])

%---------------------------------------------------------------------%
%         ANGULAR ACCELERATIONS
%---------------------------------------------------------------------%

ALF_C_N>=DT(W_C_N>,N)

ALF_D_N>=DT(W_D_N>,N)

ALF_F_N>=DT(W_F_N>,N)

ALF_G_N>=DT(W_G_N>,N)

%---------------------------------------------------------------------%
%         ACCELERATIONS
%---------------------------------------------------------------------%

A_CO_N>=DT(V_CO_N>,N)
A_DO_N>=DT(V_DO_N>,N)
A_FO_N>=DT(V_FO_N>,N)
A_GO_N>=DT(V_GO_N>,N)

%---------------------------------------------------------------------%
%         FORCES
%---------------------------------------------------------------------%

GRAVITY(G*N3>,C,D,F,G)

%---------------------------------------------------------------------%
%         EQUATIONS OF MOTION
%---------------------------------------------------------------------%

ZERO=FR()+FRSTAR()
KANE()

%---------------------------------------------------------------------%
%         ENERGY
%---------------------------------------------------------------------%

KINETIC=KE()

PE_C=MC*G*DOT(N3>,P_NO_CO>)
PE_D=MD*G*DOT(N3>,P_NO_DO>)
PE_F=MF*G*DOT(N3>,P_NO_FO>)
PE_G=MG*G*DOT(N3>,P_NO_GO>)
POTENTIAL=ABS(PE_C+PE_D+PE_F+PE_G)

%---------------------------------------------------------------------%
%         SIMULATION
%---------------------------------------------------------------------%

FW_CP_1=DOT(P_NO_GN>,N1>) % FRONT WHEEL CONTACT POINT LOCATION
FW_CP_2=DOT(P_NO_GN>,N2>) % FRONT WHEEL CONTACT POINT LOCATION

UNITSYSTEM  KG,M,S

INPUT W=1.02 M,C=0.08 M,LAMBDA=3.141592653589793/10 RAD,G=9.81 N/KG

INPUT RR=0.3 M,M_R=2 KG,IRXX=0.0603 KG*M^2,IRYY=0.12 KG*M^2

INPUT XB=0.3 M,ZB=-0.9 M,M_B=85 KG

INPUT IBXX=9.2 KG*M^2,IBYY=11 KG*M^2,IBZZ=2.8 KG*M^2,IBXZ=2.4 KG*M^2

INPUT XH=0.9 M,ZH=-0.7 M,M_H=4 KG

INPUT IHXX=0.05892 KG*M^2,IHYY=0.06 KG*M^2,IHZZ=0.00708 KG*M^2

INPUT IHXZ=-0.00756 KG*M^2

INPUT RF=0.35 M,M_F=3 KG,IFXX=0.1405 KG*M^2,IFYY=0.28 KG*M^2

INPUT Q1=0,Q2=0,Q3=0,Q4=0,Q5=0,Q6=0,Q7=0,Q8=0

INPUT U4=0.5,U5=-4.6/0.3,U7=0

INPUT TFINAL=5,INTEGSTP=0.01

OUTPUT U1, U2, U3, dot(V_DO_N>, N2>), dot(A_DO_N>, N2>)

CODE DYNAMICS() BRR_05.M

%---------------------------------------------------------------------%
%         SAVE OUTPUT
%---------------------------------------------------------------------%

SAVE BRR_05.ALL
