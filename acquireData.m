% File: acquireData.m
% Date: May 27, 2009
% Author: Jason Moore
% Description: Takes data from a single rate sensor and saves it.
clear all
close all
clc
bicycle = input('What is the bicycle? ', 's');
part = input('What part are you measuring? ', 's');
pendulum = input('What pendulum are you using? ', 's');
angle = input('Which angle are you measuring at? ', 's');
trial = input('What is the trial number? ', 's');
notes = input('Any additional info? ','s');
disp('Press any key to start recording')
pause
%s=daqhwinfo % find out what is plugged in
%s.InstalledAdaptors % show the adaptors
%out=daqhwinfo('nidaq')
%out.ObjectConstructorName(:)
ai=analoginput('nidaq','Dev1'); % set the analog input
%daqhwinfo(ai)
duration=30; % the sample time in seconds
set(ai,'SampleRate',1000) % set the sample rate
ActualRate = get(ai,'SampleRate');
set(ai,'SamplesPerTrigger',duration*ActualRate) % 
set(ai,'TriggerType','Manual')
chan = addchannel(ai,0);
start(ai)
trigger(ai)
wait(ai,duration + 1)
data = getdata(ai);
plot(data,'.-')
delete(ai)
clear ai chan
filename = [bicycle part pendulum angle trial];
save(filename) 