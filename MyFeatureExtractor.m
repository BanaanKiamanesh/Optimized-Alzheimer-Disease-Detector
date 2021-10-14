function z = MyFeatureExtractor(signal)

    % This Function Extracts Various Features From the Signal
    % In order to Have a Understanding of the Statistical Distribution...

    meanS      = mean(signal);                      % Signal Mean
    varianceS  = var(signal);                       % Signal Variance
    rmsS       = rms(signal);                       % Signal Root Mean Square
    kurtosisS  = kurtosis(signal);                  % Signal Kurtosis
    skew       = skewness(signal);                  % Signal Skewness
    ent        = wentropy(signal,'shannon');        % Signal Shannon Entropy
    rangeS     = range(signal);                     % Signal Varying Range
    rssqS      = rssq(signal);                      % Signal Root Sum of Squares

    z = [meanS, varianceS, rmsS, rssqS, kurtosisS,...
        skew, ent, rangeS];
end