%% Pre-processing
% load the data
load F.txt -ascii;

% pre-process the data
voice1 = preprocess(F(:,1));
voice2 = preprocess(F(:,2));
voice3 = preprocess(F(:,3));
voice4 = preprocess(F(:,4));

%% Linear Regression
% choose linearRegression variables
top_k = 3;
window_size = 10;
voice_no = 3;  % voice to use as training data
n = 100;  % number of notes to sample
poly_fitting = false;  % whether to include polynomials
seed = zeros(1, 5 * window_size);
if poly_fitting
    seed = [seed cartesianProduct(seed)'];
end

% perform linearRegression
[W, b, loss, idx_to_note, ~] = linearRegressionComposer(F, window_size, 1, 0.1, poly_fitting);
disp('LRvoice1 loss: '), disp(loss);
LRvoice1 = linearRegressionPredict(W, b, n, seed, window_size, top_k, idx_to_note);

[W, b, loss, idx_to_note, ~] = linearRegressionComposer(F, window_size, 2, 0.1, poly_fitting);
disp('LRvoice2 loss: '), disp(loss);
LRvoice2 = linearRegressionPredict(W, b, n, seed, window_size, top_k, idx_to_note);

[W, b, loss, idx_to_note, ~] = linearRegressionComposer(F, window_size, 3, 0.1, poly_fitting);
disp('LRvoice3 loss: '), disp(loss);
LRvoice3 = linearRegressionPredict(W, b, n, seed, window_size, top_k, idx_to_note);

[W, b, loss, idx_to_note, ~] = linearRegressionComposer(F, window_size, 4, 0.1, poly_fitting);
disp('LRvoice4 loss: '), disp(loss);
LRvoice4 = linearRegressionPredict(W, b, n, seed, window_size, top_k, idx_to_note);

%% Hidden Markov Model
% choose Markov variables
states = 4;
samples = 100;

% perform HMM on the entire dataset
[HMvoice, ~, ~] = hiddenMarkovComposer(F, states, samples);

% perform HMM on the four different voices
[HMvoice1, ~, ~] = hiddenMarkovComposer(voice1, states, samples);
[HMvoice2, ~, ~] = hiddenMarkovComposer(voice2, states, samples);
[HMvoice3, ~, ~] = hiddenMarkovComposer(voice3, states, samples);
[HMvoice4, ~, ~] = hiddenMarkovComposer(voice4, states, samples);

%% Export
% play a voice
%playVoice(LRvoice1);

% export the files
exportVoice(LRvoice1,"LRvoice1");
exportVoice(LRvoice2,"LRvoice2");
exportVoice(LRvoice3,"LRvoice3");
exportVoice(LRvoice4,"LRvoice4");
exportVoice(HMvoice,"HMvoice");
exportVoice(HMvoice1,"HMvoice1");
exportVoice(HMvoice2,"HMvoice2");
exportVoice(HMvoice3,"HMvoice3");
exportVoice(HMvoice4,"HMvoice4");