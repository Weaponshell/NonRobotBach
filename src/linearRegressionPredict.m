function res = linearRegressionPredict(W, b, n, seed, window, top_k, idx_to_note)
   % Sample notes using the weight vector w (found using linear regression)
   % for n steps.
   %
   % Input:
   % - W: linear regression weight matrix
   % - b: linear regression bias vector 
   % - n: no. of steps to sample
   % - seed: initial 1 x window sized window used for starting the predictions
   % - window: size of the window
   % - top_k: sample from the k largest probabilities
   % 
   % Output:
   % - res: the sampled notes
   
   res = zeros(n, 1);
   n_notes = size(W, 2);
   last_out = seed;
   for i = 1:n
       probs = last_out * W + b;
       [~, idxs] = maxk(probs, top_k);
       mask = zeros(1, n_notes);
       for j = 1:idxs
          mask(j) = 1;
       end
       probs = mask .* probs;  
       probs = exp(probs) / sum(exp(probs));  % apply softmax to ensure the sum of probabilities is 1
       pred_idx = randsample(1:n_notes, 1, true, probs);
       pred_note = idx_to_note(pred_idx);
       res(i) = pred_note;
       
       % Update the window using the predicted note.
       last_out(1:window-1) = last_out(2:window);
       last_out(window) = pred_note;
   end
end