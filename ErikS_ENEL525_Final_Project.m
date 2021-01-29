close all
% import data
input_data = readtable('COVID-19_formatted_dataset.csv');

% Get test data
test = table2array(input_data(:,2));
t = zeros(2,length(test));
for i = 1:length(test)
    if(test(i) == 1)
        t(1,i) = 0;
        t(2,i) = 1;
    else
        t(1,i) = 1;
        t(2,i) = 0;
    end
end
p = table2array(input_data(:,3:17));

% Rescale age input
% x_new = (x - x_min) / (x_max - x_min)
ages = p(:,1);
ages_sorted = sort(ages);
x_min = ages_sorted(1); % min age
x_max = ages_sorted(length(ages_sorted)); % max age
ages_new = ages;
for i = 1:length(ages)
    x = ages(i);
    x_new = (x - x_min) / (x_max - x_min);
    ages_new(i) = x_new;
end
p(:,1) = ages_new;

% Network configuration and parameters
INPUT_LAYER = 15;
OUTPUT_LAYER = 2;
HIDDEN_LAYERS = [6 6];
CONFIG = [INPUT_LAYER HIDDEN_LAYERS OUTPUT_LAYER];
ERROR_THRESHOLD = 0.02;
ALPHA = 0.01;
% Network implementation
% for i = length( HIDDEN_LAYERS )
    W1 = randn( HIDDEN_LAYERS(1),   INPUT_LAYER         );
    W2 = randn( HIDDEN_LAYERS(2),   HIDDEN_LAYERS(1)    );
    W3 = randn( OUTPUT_LAYER,       HIDDEN_LAYERS(2)    );
    b1 = randn( HIDDEN_LAYERS(1),   1                   );
    b2 = randn( HIDDEN_LAYERS(2),   1                   );
    b3 = randn( OUTPUT_LAYER,       1                   );
% end

MAX_LOOP_SIZE = 10000; % Max loop iterations (circuit breaker)
MSE_vector = zeros(1, MAX_LOOP_SIZE);

% Feed into network
error = 1;
count = 0;
errors = zeros(2, 538);
% errors = zeros(2, MAX_LOOP_SIZE);
while (error > ERROR_THRESHOLD)
    count = count + 1;
    for k = 1:538
       % 1. Forward propagate the inputs
        n_1 = W1 * p(k,:)' + b1;
        a_1 = tansig(n_1);
        n_2 = W2 * a_1 + b2;
        a_2 = tansig(n_2);
        n_3 = W3 * a_2 + b3;
        a_3 = tansig(n_3);
        
        errors(:,k) = t(:,k) - a_3;
       
        vals = zeros(HIDDEN_LAYERS(1),1);
        for j = 1:HIDDEN_LAYERS(1)
            vals(j) = (1-a_1(j)*a_1(j));    % 1st derivative of tansig
        end
        F1 = diag(vals);
%         TODO double check
        vals = zeros(HIDDEN_LAYERS(2),1);
        for j = 1:HIDDEN_LAYERS(2)
            vals(j) = (1-a_2(j)*a_2(j));
        end
        F2 = diag(vals);
        vals = zeros(OUTPUT_LAYER,1);
        for j = 1:OUTPUT_LAYER
            vals(j) = (1-a_3(j)*a_3(j));
        end
        F3 = diag(vals);
        
        % 2. Calculate sensitivities
        sensitivity_3 = -2 * F3  * errors(:,k);
        sensitivity_2 = F2 * W3' * sensitivity_3;
        sensitivity_1 = F1 * W2' * sensitivity_2;
        % 3. Update weights and biases
        W3 = W3 - ALPHA * sensitivity_3 * a_2';
        b3 = b3 - ALPHA * sensitivity_3;
        W2 = W2 - ALPHA * sensitivity_2 * a_1';
        b2 = b2 - ALPHA * sensitivity_2;
        W1 = W1 - ALPHA * sensitivity_1 * p(k,:);
        b1 = b1 - ALPHA * sensitivity_1;
    end
    % Calculate mean squared error for iteration
    error = mse(errors);
    MSE_vector(count) = error;
    % Conditional break to prevent looping for too long
    if (count > MAX_LOOP_SIZE)
        disp('Reached max loop iterations, breaking loop...')
        break;
    end
end

% predict next 10% of outputs
results = zeros(2,598);
results_errors = zeros(2,598);
for k = 1:598
    n_1 = W1 * p(k,:)' + b1;
    a_1 = tansig(n_1);
    n_2 = W2 * a_1 + b2;
    a_2 = tansig(n_2);
    n_3 = W3 * a_2 + b3;
    a_3 = tansig(n_3);
    results(:,k) = a_3;
    results_errors(:,k) = t(:,k) - a_3;
end
mse(results_errors)

% Figure 1: MSE_vector
figure
semilogy(MSE_vector)

% Figure 2: prediction
figure
hold on;
plot(results(1,:));
plot(results(2,:)); % plot guessed positive cases
%plot(t(1,539:598));
plot(t(2,:)); % plot true positive cases
hold off;
figure
% hold on;
plot(results(2,539:598));
hold on
plot(t(2,539:598));
