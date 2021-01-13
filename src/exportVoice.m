function exportVoice(voice, name)
    %%%%%%%% transform a voice into a soundvector that will be exported
    % plot it to get the Bach feeling 
    % figure(1); plot(voice);

    symbolicLength = length(voice);
    % find minimal key Number
    minKeyNr = 1000;
    for n = 1:symbolicLength
        if voice(n) < minKeyNr && voice(n) ~= 0
            minKeyNr = voice(n);
        end
    end

    % set parameters for playing sound
    baseFreq = 440;% set base frequency (Hz) of lowest note "minKeyNr" in voice
    sampleRate = 10000; % samples per second
    durationPerSymbol = 1/5; % in seconds. A "symbol" here means one entry in the voice vector
    ticksPerSymbol = floor(sampleRate * durationPerSymbol);

    % transform to soundvector
    soundvector1 = zeros(symbolicLength * ticksPerSymbol,1);
    currentSymbol = voice(1); startSymbolIndex = 1;
    for n = 1:symbolicLength
        if voice(n) ~= currentSymbol
            stopSymbolIndex = n-1;
            coveredSoundVectorIndices = ...
                (startSymbolIndex -1)* ticksPerSymbol + 1:...
                stopSymbolIndex * ticksPerSymbol ;
            toneLength = length(coveredSoundVectorIndices);
            frequency = baseFreq * 2^((currentSymbol - minKeyNr)/12 );        
            toneVector = zeros(toneLength,1);
            for t = 1:toneLength
                toneVector(t,1) = sin(2 * pi * frequency * t / sampleRate);
            end
            soundvector1(coveredSoundVectorIndices,1) = toneVector;
            currentSymbol = voice(n);
            startSymbolIndex = n;    
        end    
    end

    filename = name + ".wav";
    audiowrite(filename, soundvector1, sampleRate);
end