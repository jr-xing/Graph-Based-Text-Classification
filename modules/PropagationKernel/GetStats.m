% function GetStats(path_to_data, dataset)
% Get graph statistics of dataset
%
    clear all;
    clc;
    
    path_to_data = '/Users/SunYu/Desktop/598_project/graph';
    % dataset = 'eduIns_horRdr';
%     dataset = '5_classes_small';
%     dataset = '10_classes_small';
%     dataset = '15_classes_small';
%     dataset = '20_classes_small';
%     dataset = 'rate_reviews_Amazon_Instant_Video_small';
%     dataset = 'rate_reviews_Amazon_Musical_Instruments_small';
    dataset = 'r8_reuters_small';
    path2data = [path_to_data '/' dataset '/'];
    
    % graph indicator    
    graph_ind = dlmread([path2data dataset '_graph_indicator.txt']);
    num_graphs = max(graph_ind);
    graph_labels = dlmread([path2data dataset '_graph_labels.txt']);
    graph_dis = histcounts(graph_labels,5);
    
    % graph nodes    
    nodes_dis = histcounts(graph_ind, num_graphs);
    median_num_nodes = median(nodes_dis);
    max_num_nodes = max(nodes_dis);
    num_nodes = size(graph_ind,1);
    
    % edges, adjacency matrix
    edges = dlmread([path2data dataset '_A.txt']);
    num_edges = size(edges,1);

    % edge attributes (e.g. edge weights etc.)
    try
        edge_attr = dlmread([path2data dataset '_edge_attributes.txt']);
        if size(edge_attr,1) ~= num_edges
            fprintf('ERROR: Wrong number of edges in %s!\n', [dataset '_edge_attributes.txt']);
        end
        if size(edge_attr,2)>1
            fprintf('CAUTION: there are more than one edge attributes in %s!\n', [dataset '_edge_attributes.txt']);
            fprintf('CAUTION: only the first one is used in adjacency matrix.\n');
        end
    catch
        edge_attr = ones(num_edges,1);
    end
    % A 
    A = sparse(edges(:,1), edges(:,2), edge_attr(:,1), num_nodes,num_nodes);
    
    % node labels  
    try 
        node_labels = dlmread([path2data dataset '_node_labels.txt']);
    catch
        node_labels = full(sum(A, 2));   % set node label as its degree for unlabeled graph;
    end
    
    if size(node_labels,1) ~= num_nodes
        fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_labels.txt']);
    end
    
    unq_labels = unique(node_labels);
    [num_unique_labels,~] = size(unq_labels);  % number of node label classes
    
    % node attributes
    try
        node_attr = sparse(dlmread([path2data dataset '_node_attributes.txt']));
        if size(node_attr,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_attributes.txt']);
        end
    end