function [posterior] = hiddenMarkovComposer(voices, chosenVoice)
    % Predict a single voice using MATLAB's hidden markov model functions 
    %
    % Input:
    % - voices: data of all the voices
    % - chosenVoice: which voice to train on
    %
    % Output:
    % - posterior: posterior state probabilities of the given voice
    
% get the data prepared
data = (voices(:,chosenVoice))';

% create pseudo-random states
unique_data = unique(data);
emis = histc(data,unique_data);
trans = zeros(1);

% generate states
likelystates = hmmviterbi(data, trans, emis);

% create better trans and emis
[trans, emis] = hmmestimate(data, states);

% get posterior state probabilities
posterior = hmmdecode(data, trans, emis);

end

