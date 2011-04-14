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

% ask which calibration rod was used
if ~strcmp(sd.bicycle, 'Rod') && ~strcmp(sd.bicycle, 'Ucdrod')
    validRods = {'Rod', 'Ucdrod'};
    rodQuestion = ['Which calibration rod was used with this bicycle?' validText];
    sd.rod = check_input(validRods, rodQuestion);
end

% ask which part is being measured
validParts = {'Rwheel', 'Fwheel', 'Fork', 'Frame', 'Flywheel', 'Rod'};
partQuestion = ['What part are you measuring?' validText];
sd.part = check_input(validParts, partQuestion);

% ask which type of pendulum
validPendulums = {'Torsional', 'Compound'};
pendulumQuestion = ['What pendulum are you using?' validText];
sd.pendulum = check_input(validPendulums, pendulumQuestion);

% ask which order of the angle it is
validAngleOrders = {'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Six'};
angleOrderQuestion = ['Which angle order is this?' validText];
sd.angleOrder = check_input(validAngleOrders, angleOrderQuestion);

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
    sd.duration = 10; % the sample time in seconds
    set(ai, 'SampleRate', 500) % set the sample rate
    sd.sampleRate = get(ai, 'SampleRate');
    set(ai, 'SamplesPerTrigger', sd.duration * sd.sampleRate) %
    set(ai, 'TriggerType', 'Manual')

    % steer rate gyro is in channel 5
    chan = addchannel(ai, 5);
    % set all the input ranges
    set(chan, 'InputRange', [-5 5])

    start(ai)
    trigger(ai)
    sd.timeStamp = datestr(clock);
    wait(ai, sd.duration + 1)
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

% ask the question
userInput = input(sprintf(question), 's');

% see if they entered a number
if str2num(userInput)
    userInput = validList{str2num(userInput)};
end

display(sprintf('You entered: %s', userInput))

% ask the question until the user types a valid answer or 'q'
if ~ismember(userInput, validList)
    while ~ismember(userInput, validList)
        display('Invalid response, try again or q for quit')
        userInput = input(sprintf(question), 's');
        if strcmp(userInput, 'q')
            display('You get doodoo!')
            % raise an error
            doodoo
        end
    end
end
