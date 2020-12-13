function res = preprocess(voice)
    % Preprocess (encode) a single voice.
    %
    % Input:
    % - voice: 1xN sized vector containing the data
    %
    % Output:
    % - res: n_notes x N sized matrix containing one-hot encoded data

    t_steps = size(voice, 1);
    notes = sort(unique(voice));
    n_notes = length(notes);
    
    % Encode the note values into indexes starting at 1.
    idx_to_note = containers.Map(1:n_notes, notes);
    note_to_idx = containers.Map(notes, 1:n_notes);
    
    % Convert the data into a set of one-hot encoded vectors, consisting 
    % of all 0s, except for the index of the voice at a particular time
    % step.
    res = zeros(t_steps, n_notes);
    for t = 1:t_steps
        idx = note_to_idx(voice(t));
        res(t, idx) = 1;
    end
end