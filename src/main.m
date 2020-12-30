function main(voices)
    % Hyperparameters
    % Sample from the top k biggest output probabilities. 
    % I.e. when top_k = 1, always choose the note with the highest
    % probability.
    top_k = 3;
    window_size = 10;
    voice_no = 3;  % voice to use as training data
    n = 100;  % number of notes to sample
    poly_fitting = false;  % whether to include polynomials
    
    seed = zeros(1, 5 * window_size); 
    %voice_flipped = flip(voices, 3);
    %seed = voice_flipped(1:window_size);
    if poly_fitting
        seed = [seed cartesianProduct(seed)'];
    end
    
    [W, b, loss, idx_to_note, ~] = linearRegressionComposer(voices, window_size, voice_no, 0.1, poly_fitting);
    disp('loss: '), disp(loss)
    out = linearRegressionPredict(W, b, n, seed, window_size, top_k, idx_to_note);
    playVoice(out);
end