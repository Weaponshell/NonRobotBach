% load the data
load F.txt -ascii;

% pre-process the data
voice1 = preprocess(F(:,1));
voice2 = preprocess(F(:,1));
voice3 = preprocess(F(:,1));
voice4 = preprocess(F(:,1));

% choose linearRegression variables
window = 0;

% perform linearRegression
LRvoice1 = linearRegressionComposer(F, window, 1);
LRvoice2 = linearRegressionComposer(F, window, 2);
LRvoice3 = linearRegressionComposer(F, window, 3);
LRvoice4 = linearRegressionComposer(F, window, 4);

% choose Markov variables


% perform Markov Modelling
HMvoice1 = preprocess(F(:,1));
HMvoice2 = preprocess(F(:,1));
HMvoice3 = preprocess(F(:,1));
HMvoice4 = preprocess(F(:,1));

% transform the voices to playable files

% export the files