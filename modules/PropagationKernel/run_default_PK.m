% %
% Setup default values and run PK computation. Note that the kernel will
% perform BETTER if you learn the parameters via cross validation instead of
% using these defaults. 
% 
% Marion Neumann (m.neumann@wustl.edu)
% % 

% propagation kernel parameter
num_iterations = 3;     % number of iterations (sth small 2 or 3)

% hashing parameters
w              = 1e-5;  % bin width
distance       = 'tv';  % distance to approximately preserve

% load you dataset HERE
% load('test_example.mat');      

num_nodes   = size(A, 1);   % number graphs
num_classes = max(node_labels);  % number of node label classes


% row-normalize adjacecny matrix A
row_sum = sum(A, 2);
row_sum(row_sum==0)=1;  % avoid dividing by zero => disconnected nodes
A = bsxfun(@times, A, 1 ./ row_sum);


initial_label_distributions = accumarray([(1:num_nodes)', node_labels], 1, [num_nodes, num_classes]);

% create a function handle to a feature transformation. Here we will
% use label diffustion as we have fully labeled graphs.
transformation = @(features) label_diffusion(features, A);


% calculate the graph kernel using the default (linear) base kernel
K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
                       num_iterations, ...
                       'distance', distance, ...
                       'w',        w);

% K is a (m x m x num_iterations+1) array containing PKs for ALL ITERATIONS 
% UP TO num_iterations (m is the number of graphs). Note that K(:,:,1) 
% corresponds to the 0-th iteration, where we only look at node labels (no 
% propgataion yet)
K = K(:,:,end);     % PK for the last iteration 
                   

% % If you want to set the BASE KERNEL to an RBF kernel instead of the default 
% % linear base kernel, use the following.
% length_scale = 3;
% base_kernel = @(counts) ...
%               exp(-(squareform(pdist(counts)).^2 / (2 * length_scale^2)));
% 
%           % calculate the graph kernel again using the new parameters
% K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
%                        num_iterations, ...
%                        'distance',    distance, ...
%                        'w',           w, ...
%                        'base_kernel', base_kernel);
