function [w, pred] = linearRegressionComposer(voices, window, voice_no)
    % Predict a single voice using window based linear regression. 
    %
    % Input:
    % - voices: nx4 matrix containing ground truth data for 4 voices (F.txt)
    % - window: window size used for linear regression
    % - voice_no: voice to predict, integer between 1-4
    %
    % Output:
    % - w: linear regression weight vector
    % - pred: output predictions based on the training data
    
    % Prepare data for time series linear regression
    [m n] = size(voices);
    A = zeros(m - window, window * n);
    for i = window+1:m+1
        A(i-window, :) = reshape(voices(i-window:(i-1), :), [1, window * n]);
    end
    y = zeros(m-(window-1), 1);
    y(1:m-window, :) = voices(window+1:m, voice_no);
    
    % Solve the normal equations, i.e. apply least squares fitting.
    w = A \ y;
    
    % Predict some data, rounding the numbers to get integer note values.
    pred = max(round(A * w), 0);
end
