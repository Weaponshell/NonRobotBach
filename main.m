function main(voices)
    % hyperparameters
    % sample from the top k biggest output probabilities. 
    % I.e. when top_k = 1, always choose the note with the highest
    % probability.
    top_k = 3;
    window_size = 50;
    voice_no = 1;   
    n = 50;             % number of predictions to make
    
    %seed = zeros(1, window_size); 
    voice_flipped = flip(voices, 3);
    seed = voice_flipped(1:window_size);
    [w, ~, idx_to_note, ~] = linearRegressionComposer(voices, window_size, voice_no);
    out = linearRegressionPredict(w, n, seed, window_size, top_k, idx_to_note);
    playVoice(out);
end