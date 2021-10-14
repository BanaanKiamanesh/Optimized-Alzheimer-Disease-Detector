function [Err, Acc]=Classifier(Features, Labels, FeatureRows)

    % This Function is Responsible of Classifying and Bringing Back and Err

    % Empty Confusion Mat
    confMat = [0];

    % Feature Filtering Due to Optimization
    myFeatures = Features(:, FeatureRows);

    % Number of Available Data for Each Class
    DataNum = numel(Labels);
    ClassDataNum = DataNum / 3;

    % Test 2 Train Data Ratio
    Ratio = 0.8;

    % Have the Process Done for k Time
    for  k = 1:10
        % Shuffle Data and Test Train Split
        idx = randperm(ClassDataNum);
        idx = idx(1: round(ClassDataNum*Ratio));
        PickIdx = [idx, (idx + ClassDataNum), (idx + ClassDataNum*2)];

        % Train
        TrainLabel = Labels(PickIdx);
        TrainData = myFeatures(PickIdx, :);
        % Test
        TestLabel = Labels; TestLabel(PickIdx) = [];
        TestData = myFeatures; TestData(PickIdx, :) = [];

        % Train Model Using KNN
        Model = fitcknn(TrainData, TrainLabel, 'Distance',...
            'seuclidean', 'NumNeighbors', 9);

        Out = predict(Model, TestData);             % Validate Model

        % % Train Model Using LDA
        % Model = fitcdiscr(TrainData, TrainLabel, 'DiscrimType' ...
        %    , 'pseudolinear');
        % Out  = predict(Model, TestData);             % Validate Model

        % % Train Model Using Decision Tree
        % Model = fitctree(TrainData, TrainLabel);
        % Out  = predict(Model, TestData);             % Validate Model

        confMat  = confMat  + confusionmat(TestLabel, Out);
    end

    % Calculate InterClass Accuracy
    Acc = zeros(1, 3);

    for i = 1: numel(Acc)
        Acc(i) = confMat(i, i)/ sum(confMat(i, :));
    end

    TotalAcc = sum(diag(Acc))/sum(Acc, "all");

    Err = 1/prod(Acc);                  % Error Fuction #1
    % Err = 1/sum(Acc);                   % Error Fuction #2
    % Err = 1/TotalAcc;                   % Error Fuction #3
    % Err = 1-TotalAcc;                   % Error Fuction #4
end