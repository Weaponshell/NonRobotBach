function [soundvector] = hiddenMarkovComposer(voices, chosenVoice)
    % Predict a single voice using window based linear regression. 
    %
    % Input:
    % - voices: data of all the voices
    % - chosenVoice: which voice to compose on
    %
    % Output:
    % - soundvector: a soundvector containing the calculated voice
    
    
    
    
    
soundvector = voices;

% still WIP! not functional!
trans = voices;
emis = chosenVoice;
[seq, states] = hmmgenerate(1000, trans, emis);

likelystates = hmmviterbi(seq, trans, emis);

[trans_est, emis_est] = hmmestimate(seq, states);

pstates = hmmdecode(seq, trans, emis);

end

