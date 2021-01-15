function [samples, ESTTR, ESTEMIT] = hiddenMarkovComposer(voices, states, samples)
    
    [m, ~] = size(voices);
    notes = sort(unique(voices(:)));
    n_notes = length(notes);
    %max_note = notes(n_notes);
    %min_note = notes(2); % idx 1 is 0
    
    % Encode the note values into indexes starting at 1.
    %idx_to_note = containers.Map(1:n_notes, notes);
    note_to_idx = containers.Map(notes, 1:n_notes);
      
    observations = voices(:);
    for i = 1:length(observations)
        observations(i) = note_to_idx(observations(i));
    end
    
    EMITGUESS = zeros(states, n_notes);
    % Count note occurences for initial parameter estimation
    for v = 1:states
        for i = 1:m
            note_idx = note_to_idx(voices(i, v));
            EMITGUESS(v, note_idx) = EMITGUESS(v, note_idx) + 1;
            %EMITGUESS(v, i) = random(makedist('Normal', 0.5, 0.1));
        end
    end
    for s = 1:states
        EMITGUESS(s, :) = EMITGUESS(s, :) / sum(EMITGUESS(s, :));
    end
    
    %EMITGUESS = ones(n_states, n_notes) / n_notes;
    TRGUESS = ones(states, states) / states; % uniform distribution
    %TRGUESS = [[0.5 0.5 0 0 ]; [1/3 1/3 1/3 0]; [0 1/3 1/3 1/3]; [0 0 0.5 0.5]];
    [ESTTR, ESTEMIT] = hmmtrain(observations, TRGUESS, EMITGUESS, 'Verbose', true);
      
    samples = zeros(samples, 1);
    state = 1; % TODO: how to determine the initial state?
    for t = 1:samples
        emit_dist = ESTEMIT(state, :);
        [~, idxs] = maxk(emit_dist, 10);
        mask = zeros(1, n_notes);
        mask(idxs) = 1;
        emit_dist = mask .* emit_dist;
        emit_dist = emit_dist / sum(emit_dist); 
        obs_new = randsample(1:n_notes, 1, true, emit_dist);
        state = randsample(1:states, 1, true, ESTTR(state, :));
        samples(t) = obs_new;
    end
 
    %playVoice(res);
    
end
