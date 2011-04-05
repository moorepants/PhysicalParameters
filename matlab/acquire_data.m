% File: acquire_data.m
% Creation Date: May 27, 2009
% Author: Jason Moore
% Description: Data collection script for the moment of inertia measurments of
% a bicycle. Takes data from a single rate sensor and saves it.

function acquire_data()

    clear all;close all;clc;

    validText = '\nValid options are:\n';

    validBicycles = {'Rigid', 'Flexible', 'Pista', 'Gary', 'Browser', ...
                     'Browserins', 'Yellow', 'Yellowrev', 'Stratos', ...
                     'Crescendo', 'Rod'};
    bicycleQuestion = ['Enter the short name of the bicycle.' validText];
    bicycle = check_input(validBicycles, bicycleQuestion);

    validParts = {'Rwheel', 'Fwheel', 'Fork', 'Frame', 'Rod'};
    partQuestion = ['What part are you measuring?' validText];
    part = check_input(validParts, partQuestion);

    validPendulums = {'Torsional', 'Compound'};
    pendulumQuestion = ['What pendulum are you using?' validText];
    pendulum = check_input(validPendulums, pendulumQuestion);

    validAngleOrders = {'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Six'};
    angleOrderQuestion = ['Which angle order is this?' validText];
    angleOrder = check_input(validAngleOrders, angleOrderQuestion);

    % the trial should be an integer
    trial = input('What is the trial number?\n', 's');

    if strcmp(pendulum, torsional) && ...
       (strcmp(part, 'Fork') || strcmp(part, 'Frame')
        angle = input(sprintf('What is the orientation angle of the %s?', part))
        distance = input('What is the wheel to cg distance?')
    end

    notes = input('Any additional info?\n','s');

    filename = [bicycle part pendulum angleOrder trial '.m'];
    display(sprintf('This is the filename: %s', filename))

    % check to make sure you aren't overwriting a file
    directory = ['..' filesep 'data' filesep 'pendDat'];
    dirInfo = what(directory)
    matFiles = dirInfo.mat
    overWrite = 'n'
    if ismember(filename, matFiles)
        overWrite = input(sprintf(['Are you sure you want' ...
                                  ' to overwrite %s (y or n)'], ...
                                  filename), 's')
    end

        if strcmp(overWrite, 'y')
        else if strcmp(overWrite, 'n')

    end

    disp('Press any key to start recording')
    pause

    ai = analoginput('nidaq','Dev1'); % set the analog input

    duration = 30; % the sample time in seconds
    set(ai, 'SampleRate', 500) % set the sample rate
    actualRate = get(ai, 'SampleRate');
    set(ai, 'SamplesPerTrigger', duration * actualRate) %
    set(ai, 'TriggerType', 'Manual')

    chan = addchannel(ai, 0);

    start(ai)
    trigger(ai)
    wait(ai, duration + 1)

    data = getdata(ai);
    plot(data,'.-')

    delete(ai)
    clear ai chan

    save(filename)

function userInput = check_input(validList, question)

    for i = 1:length(validList)
        question = [question validList{i} ','];
    end

    userInput = input(sprintf([question(1:end-1) '\n']), 's');

    if ~ismember(userInput, validList)
        while ~ismember(userInput, validList)
            display('Invalid response, try again')
            userInput = input(sprintf([question(1:end-1) '\n']), 's');
        end
    end
