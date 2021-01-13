function [W, b, loss, idx_to_note, note_to_idx] = linearRegressionComposer(voices, window, voice_no, ridge_param, poly_fitting)
    % Predict a single voice using window based linear regression. 
    %
    % Input:
    % - voices: nx4 matrix containing ground truth data for 4 voices (F.txt)
    % - window: window size used for linear regression
    % - voice_no: voice to predict, integer between 1-4
    % - ridge_param: ridge regression parameter
    % - poly_fitting: boolean indicating whether to include polynomials of
    %   degree 2
    %
    % Output:
    % - W: linear regression weight matrix, size nxk
    % - b: linear regression bias vector, size nx1
    % - loss: training loss of the linear regression model
    % - idx_to_note: map that maps note indexes to note values
    % - note_to_idx: map that maps note values to note indexes
    
    [m,~] = size(voices);
    notes = sort(unique(voices(:, voice_no)));
    n_notes = length(notes);
    n_output = m - window;
    max_note = notes(n_notes);
    min_note = notes(2); % idx 1 is 0
    
    % Encode the note values into indexes starting at 1.
    idx_to_note = containers.Map(1:n_notes, notes);
    note_to_idx = containers.Map(notes, 1:n_notes);
    
    % Prepare data for time series linear regression.
    if poly_fitting
        X = zeros(n_output, 5 * window + (5 * window)^2 + 1);  % + 1 because we add a bias term
    else 
        X = zeros(n_output, 5 * window + 1);
    end
    for i = window+1:m
        start_idx = i - window;
        end_idx = i - 1;
        window_notes = voices(start_idx:end_idx, voice_no)';
        enc_data = zeros(1, 5 * length(window_notes));
        for j = 1:length(window_notes)
            note = window_notes(j);
            enc_data(5*(j-1)+1:5*j) = note_to_vector(note, min_note, max_note - min_note + 1);
        end
        if poly_fitting
            X(start_idx, :) = [enc_data cartesianProduct(enc_data)' 1];
        else
            X(start_idx, :) = [enc_data 1];
        end
    end
    
    % Prepare output data. 
    out_notes = zeros(n_output, 1);
    out_notes(1:m-window, :) = voices(window+1:m, voice_no);
    
    % Convert the output data into a set of one-hot encoded vectors, consisting 
    % of all 0s, except for the index of the voice at a particular time
    % step. This ensures we get a probability vector as output.
    Y = zeros(n_output, n_notes);
    for t = 1:n_output
        idx = note_to_idx(out_notes(t));
        Y(t, idx) = 1;
    end    
    
    % Solve the normal equations, i.e. apply least squares fitting.
    %weights = X \ y;
    weights = inv(X' * X + ridge_param * eye(size(X, 2))) * X' * Y;
    w_len = size(weights, 1) - 1;
    W = weights(1:w_len, :);
    b = weights(w_len+1, :);
    
    loss = (1 / n_output) * norm(X(:, 1:(size(X, 2)-1)) * W + b - Y)^2;
end
