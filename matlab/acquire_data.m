function acquire_data()
% File: acquire_data.m
% Creation Date: May 27, 2009
% Author: Jason Moore
% Description: Data collection script for the moment of inertia measurments of
% a bicycle. Takes data from a single rate sensor and saves it.

clear all;close all;clc;

% make sure the analog input was properly deleted
if exist('ai')
    delete(ai)
end

sd = struct;

validText = '\nValid options are:\n';

% ask which bicycle is being measured
validBicycles = {'Rigid', 'Flexible', 'Pista', 'Fisher', 'Browser', ...
                 'Browserins', 'Yellow', 'Yellowrev', 'Stratos', ...
                 'Crescendo', 'Rod', 'Ucdrod', 'Gyro'};
bicycleQuestion = ['Enter the short name of the bicycle.' validText];
sd.bicycle = check_input(validBicycles, bicycleQuestion);
if strcmp(sd.bicycle, 'q')
    return
end

% ask which part is being measured
validParts = {'Rwheel', 'Fwheel', 'Fork', 'Frame', 'Flywheel', 'Rod', ...
              'Handlebar'};
partQuestion = ['What part are you measuring?' validText];
sd.part = check_input(validParts, partQuestion);
if strcmp(sd.part, 'q')
    return
end

% ask which type of pendulum
validPendulums = {'Torsional', 'Compound'};
pendulumQuestion = ['What pendulum are you using?' validText];
sd.pendulum = check_input(validPendulums, pendulumQuestion);
if strcmp(sd.pendulum, 'q')
    return
end

if strcmp(sd.pendulum, 'Torsional')
    % ask which calibration rod was used
    if ~strcmp(sd.bicycle, 'Rod') && ~strcmp(sd.bicycle, 'Ucdrod')
        validRods = {'Rod', 'Ucdrod'};
        rodQuestion = ['Which calibration rod was used with this bicycle?' validText];
        sd.rod = check_input(validRods, rodQuestion);
    end
    if strcmp(sd.rod, 'q')
        return
    end
end

% ask which order of the angle it is
validAngleOrders = {'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Six'};
angleOrderQuestion = ['Which angle order is this?' validText];
sd.angleOrder = check_input(validAngleOrders, angleOrderQuestion);
if strcmp(sd.angleOrder, 'q')
    return
end

% the trial should be an integer
sd.trial = input('What is the trial number?\n', 's');

% get the angle and distance measurement for the fork and frame torsional
% measurements
if strcmp(sd.pendulum, 'Torsional') && ...
   (strcmp(sd.part, 'Fork') || strcmp(sd.part, 'Frame'))
    sd.angle = input(sprintf('What is the orientation angle of the %s?\n', ...
                             sd.part));
    sd.distance = input('What is the wheel to cg distance?\n');
end

% the sample time in seconds
sd.duration = input('Enter the total sample duration in secs?\n'); 

sd.notes = input('Any additional info?\n','s');

% build the filename
sd.filename = [sd.bicycle sd.part sd.pendulum sd.angleOrder sd.trial '.mat'];
display(sprintf('This is the filename: %s', sd.filename))

% check to make sure you aren't overwriting a file
directory = ['..' filesep 'data' filesep 'pendDat'];
dirInfo = what(directory);
matFiles = dirInfo.mat;
% if the file exists ask the user if they want to overwrite it
if ismember(sd.filename, matFiles)
    overWrite = input(sprintf(['%s already exists, are you sure you want' ...
                              ' to overwrite it? (y or n)\n'], ...
                              sd.filename), 's');
   if strcmp(overWrite, 'y')
       overWrite = input('Are you really sure? (y or n)\n', 's');
   end
else
    overWrite = 'y';
end

% if overwrite is true or the file doesn't exist, then take the data and save
% the file
if strcmp(overWrite, 'y')
    disp('Press any key to start recording')
    pause
    disp('Recording...')

    ai = analoginput('nidaq','Dev1'); % set the analog input
    set(ai, 'InputType', 'SingleEnded') % Differential is default
    set(ai, 'SampleRate', 500) % set the sample rate
    sd.sampleRate = get(ai, 'SampleRate');
    set(ai, 'SamplesPerTrigger', sd.duration * sd.sampleRate) %
    set(ai, 'TriggerType', 'Manual')

    % steer rate gyro is in channel 5
    chan = addchannel(ai, 5);
    % set all the input ranges
    set(chan, 'InputRange', [-10 10])

    start(ai)
    trigger(ai)
    sd.timeStamp = datestr(clock);
    wait(ai, sd.duration + 2)
    display('Finished recording.')

    sd.data = getdata(ai);
    plot(sd.data,'.-')

    delete(ai)
    clear ai chan

    save([directory filesep sd.filename], '-struct', 'sd')

elseif strcmp(overWrite, 'n')
    display('No Data Taken, Start Over')
end

function userInput = check_input(validList, question)
% Returns the user keyboard input as long as it is in the valid list of
% answers.
%
% Parameters
% ----------
% validList : cell array of strings
%   A list of valid answers for the question.
% question : string
%   A question to ask the user.
%
% Returns
% -------
% userInput : string
%   The valid keyboard input of the user.

% append the valid options to the question
for i = 1:length(validList)
    question = [question num2str(i) ' : ' validList{i} '\n'];
end
question = [question(1:end-1) '\n'];

while 1
    % ask the question
    userInput = input(sprintf(question), 's');
            
    % if they entered 'q' then break from the while loop
    if strcmp(userInput, 'q')
        display('You get doodoo!')
        userInput = 'q';
        break
    % if the input is in validList then break from the while loop
    elseif ismember(userInput, validList)
        break
    % if the input is a number and is in the validList then break
    elseif ismember(str2num(userInput), 1:length(validList))
        userInput = validList{str2num(userInput)};
        break
    else
        display('Invalid response, try again or q for quit')
    end
end