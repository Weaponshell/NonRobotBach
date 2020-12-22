function [w, b, loss, idx_to_note, note_to_idx] = linearRegressionComposer(voices, window, voice_no)
    % Predict a single voice using window based linear regression. 
    %
    % Input:
    % - voices: nx4 matrix containing ground truth data for 4 voices (F.txt)
    % - window: window size used for linear regression
    % - voice_no: voice to predict, integer between 1-4
    %
    % Output:
    % - w: linear regression weight vector
    % - b: linear regression bias term
    % - pred: output predictions based on the training data
    
    % Prepare data for time series linear regression
    [m n] = size(voices);
    notes = sort(unique(voices(:, voice_no)));
    n_notes = length(notes);
    n_output = m - window;
    
    % Encode the note values into indexes starting at 1.
    idx_to_note = containers.Map(1:n_notes, notes);
    note_to_idx = containers.Map(notes, 1:n_notes);
    
    A = zeros(n_output, window + 1);  % window + 1 because we add a bias term
    for i = window+1:m
        start_idx = i - window;
        end_idx = i - 1;
        A(start_idx, :) = [voices(start_idx:end_idx, voice_no)' 1];
    end
    
    out_notes = zeros(n_output, 1);
    out_notes(1:m-window, :) = voices(window+1:m, voice_no);
    % Convert the (output) data into a set of one-hot encoded vectors, consisting 
    % of all 0s, except for the index of the voice at a particular time
    % step. This ensures we get a probability vector as output.
    y = zeros(n_output, n_notes);
    for t = 1:n_output
        idx = note_to_idx(out_notes(t));
        y(t, idx) = 1;
    end    
    
    % Solve the normal equations, i.e. apply least squares fitting.
    weights = A \ y;
    w_len = size(weights, 1) - 1;
    w = weights(1:w_len, :);
    b = weights(w_len+1, :);
    
    loss = (1 / n_output) * norm(A(:, 1:window) * w + b - y)^2;
end
