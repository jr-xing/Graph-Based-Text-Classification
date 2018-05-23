function K = PropagationKernel(path_to_data, dataset, label_transform_model, t_max)
%% Propagation Kernel
% Summary:
%   A function that reads .mat and runs 
%   propagation Kernel based on these data.
% 
    
%% Read graph adjacency matrix A, edge-graph indicator graph_ind,node labels node_labels and attributes node_attr
path2graph = [path_to_data '/' dataset '.mat'];
graph_struct = load(path2graph);
A = graph_struct.A;
num_nodes = size(A,1);

% (n_graph,1) indicator array: node are of which graph
graph_ind = graph_struct.graph_ind;
if size(graph_ind,1) ~= num_nodes
    graph_ind = graph_ind';
end

% (n_node, 1) array, node labels
node_labels = graph_struct.node_labels;
if size(node_labels,1) ~= num_nodes
    node_labels = node_labels';
end

% (n_node, attr_dim), node attributes
node_attr = graph_struct.node_attr;
if size(node_attr,1) ~= num_nodes
    node_attr = node_attr';
end

%% Propagation Kernel
% The PK will perform better if hyperparameters are learned

    % HYPERPARAMETERS
    % propagation kernel parameter    
    num_iterations = t_max;     % number of iterations (something small 2 or 3)

    % hashing parameters
    w              = 1e-5;  % label bin width
%     w_attr         = 1e-5;  % attribute bin width
    w_attr         = 1;     % attribute bin width when attribute is normalized
    distance       = 'tv';  % distance to approximately preserve
    dist_attr      = 'l1';   % 'l1' or 'l2'

    % SET ATTRIBUTE PROPAGATION
    
    
    % SET LABEL PROPAGATION
    % get statistics of node labels
    unq_labels = unique(node_labels);
    [num_unique_labels,~] = size(unq_labels);  % number of node label classes
    num_classes = max(node_labels);   % work for accumarray only

    % row-normalize adjacecny matrix A
    row_sum = sum(A, 2);
    row_sum(row_sum==0)=1;  % avoid dividing by zero => disconnected nodes
    A = bsxfun(@times, A, 1 ./ row_sum);
    
    % Set initial label distribution    
    % deal with node label == 0 & deal with node label == itself degree;
    initial_label_distributions = accumarray([(1:num_nodes)', node_labels+1], 1, [num_nodes, num_classes+1]);
    [r,~] = size(find(node_labels==0));
    u = repmat(1/num_unique_labels,r,num_classes);  % initialize unlabel node's label distribution
    uniform = [zeros(r,1), u];
    initial_label_distributions(node_labels == 0,:) = uniform; 
    initial_label_distributions(:, sum(abs(initial_label_distributions)) == 0) = []; % deal with node label == itself degree;
    
    % LABEL TRANSFORMATION MODELS
    % create a function handle to a feature transformation. Here we will
    % use label diffustion as we have fully labeled graphs.
    switch label_transform_model
        case 'diffusion'
            transformation = @(features) label_diffusion(features, A);
        case 'propagation'
            train_ind = find(node_labels);
            observed_labels = node_labels(train_ind);
            transformation = @(features) ...
                label_propagation(features, A, train_ind, observed_labels);
    end
    
    
    % RUN PROPAGATION KERNEL
    % calculate the graph kernel using the default (linear) base kernel
    isExist = exist('node_attr','var');
    if isExist ~= 0  % label & attribute
        fprintf('computing kernel with label & attribute\n');
        K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
                               num_iterations, ...
                               'distance',  distance, ...    % label
                               'w',         w, ...          
                               'attr',      node_attr, ...   % attribute
                               'w_attr',    w_attr, ...
                               'dist_attr', dist_attr);
    else  % label only
        fprintf('computing kernel with label only\n');
        K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
                               num_iterations, ...
                               'distance',  distance, ...    % label
                               'w',         w);
    end
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
    
end

