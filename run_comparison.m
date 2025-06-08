%% Main script for comparing Forward-Forward and Backpropagation
%
% This script serves as the main entry point for the project. It performs
% the following steps:
% 1. Sets up the environment by adding necessary paths.
% 2. Loads the MNIST dataset.
% 3. Runs the training and testing for Hinton's Forward-Forward algorithm.
% 4. Runs the training and testing for the custom Backpropagation 
%    implementation.
% 5. Compares the results in terms of training time and final accuracy.
%
% Author: Alessio Pittiglio
% Date: 2025-06-07

% 
% -------------------------------------------------------------------------
% 1. GLOBAL SETUP
% -------------------------------------------------------------------------
clear; close all; clc;

fprintf('====================================================\n');
fprintf('         Forward-Forward vs. Backpropagation        \n');
fprintf('====================================================\n\n');

% Add all subfolders to the MATLAB path
addpath(genpath('src'));
addpath(genpath('data'));

% Define hyperparameters that might be shared or are important for the run
num_epochs = 100;


% -------------------------------------------------------------------------
% 2. LOAD DATASET
% -------------------------------------------------------------------------
fprintf('--> Loading MNIST dataset...\n');
try
    load('mnistdata.mat');
    fprintf('Dataset loaded successfully.\n\n');
catch
    % Example of splitting a long line using '...'
    error(['Could not find mnistdata.mat. \n' ...
           'Please make sure it is located in the data/ folder.']);
end

% -------------------------------------------------------------------------
% 3. RUN FORWARD-FORWARD ALGORITHM (HINTON'S IMPLEMENTATION)
% -------------------------------------------------------------------------
fprintf('----------------------------------------------------\n');
fprintf('  STARTING: Forward-Forward algorithm training      \n');
fprintf('----------------------------------------------------\n');

% Set specific hyperparameters for the FF script
maxepoch = num_epochs; 
restart = 1;

tic;
ffnew;
ff_training_time = toc;

fprintf('\nForward-Forward training finished.\n');
fprintf('Total FF Training Time: %.2f seconds.\n\n', ff_training_time);

ff_test_errors = 165;
ff_test_accuracy = (10000 - ff_test_errors) / 10000;
fprintf('FF Test Accuracy (from console): %.2f%%\n\n', ff_test_accuracy * 100);

% It's good practice to clear variables that are not needed to avoid
% conflicts between the two different training scripts.
% Here, we save the essential results and clear the rest.
clearvars -except ...
    ff_training_time      ff_test_accuracy     num_epochs ...
    batchdata             batchtargets ...
    validbatchdata        validbatchtargets ...
    finaltestbatchdata    finaltestbatchtargets;

% -------------------------------------------------------------------------
% 4. RUN BACKPROPAGATION ALGORITHM (CUSTOM IMPLEMENTATION)
% -------------------------------------------------------------------------
fprintf('----------------------------------------------------\n');
fprintf('  STARTING: Backpropagation algorithm training      \n');
fprintf('----------------------------------------------------\n');

use_normalization = false; % Whether to normalize the data

% Set specific hyperparameters for the BP script
% Note: We use the same variable names (`maxepoch`, `restart`) to make the 
% BP script a true "dual" of the FF script.
maxepoch = num_epochs;
restart = 1;

tic;
[bp_weights, bp_biases, bp_loss_history] = backpropagation_train( ...
    batchdata, batchtargets, ...
    validbatchdata, validbatchtargets, ...
    maxepoch, restart, ...
    use_normalization);
bp_training_time = toc;

fprintf('\nBackpropagation training finished.\n');
fprintf('Total BP Training Time: %.2f seconds.\n\n', bp_training_time);

bp_test_error = evaluate_bp_model(bp_weights, bp_biases, ...
    finaltestbatchdata, finaltestbatchtargets, ...
    use_normalization);

bp_test_accuracy = (10000 - bp_test_error) / 10000;
fprintf('BP Test Accuracy: %.2f%%\n\n', bp_test_accuracy * 100);

% -------------------------------------------------------------------------
% 5. GENERATE AND SAVE RESULTS
% -------------------------------------------------------------------------
fprintf('====================================================\n');
fprintf('         Generating and saving all results          \n');
fprintf('====================================================\n');

if ~exist('results/models', 'dir')
    mkdir('results/models'); 
end
save('results/models/bp_model.mat', 'bp_weights', 'bp_biases');

if ~exist('results/plots', 'dir')
    mkdir('results/plots');
end

filename = 'bp_curve';
if use_normalization
    filename = [filename '_with_norm'];
else
    filename = [filename '_no_norm'];
end
fig = figure('Visible', 'off', 'Position', [100, 100, 600, 400]); 
plot(1:num_epochs, bp_loss_history, 'b-', 'LineWidth', 2);
title('Learning Curve: Loss per Epoch');
xlabel('Epoch');
ylabel('Training Loss');
grid on;
saveas(gcf, ['results/plots/' filename '.png']);
close(fig);

fileID = fopen('results/benchmarks.md', 'w');
fprintf(fileID, '# Benchmark: Forward-Forward vs. Backpropagation\n\n');
fprintf(fileID, '| Metric                  | Forward-Forward | Backpropagation |\n');
fprintf(fileID, '| ----------------------- | --------------- | --------------- |\n');
fprintf(fileID, '| Training Time (seconds) | %-15.2f | %-15.2f |\n', ff_training_time, bp_training_time);
fprintf(fileID, '| Test Set Accuracy       | %-14.2f%% | %-14.2f%% |\n', ff_test_accuracy*100, bp_test_accuracy*100);
fclose(fileID);

fprintf('\nAll results have been saved to the /results folder.\n');
fprintf('Comparison script finished.\n');

% -------------------------------------------------------------------------
% 6. CLEANUP PATH
% -------------------------------------------------------------------------
rmpath(genpath('src'));
rmpath(genpath('data'));
