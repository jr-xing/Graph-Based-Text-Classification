%% Introduction
% function K = PropagationKernel(path_to_data, dataset, label_transform_model, t_max)
%     label_transform_model = 'diffusion';
%     path_to_data = '/Users/SunYu/Desktop/598_project/graph';
%     dataset = 'eduIns_horRdr';

%% Set path
path_to_data = 'E:/xdocuments/Courses/2018_Summer/graph/graph_text_clf/data/graph_temp';
dataset = 'demo_mini';
path2data = [path_to_data '/' dataset '/'];
path2A = [path2data dataset '.mat'];
A_struct = load(path2A);
A = A_struct.A;
path2Afull = [path2data dataset '_full.mat'];
A_full = load(path2Afull);
num_nodes = size(A_full.A,1);

graph_ind = A_full.graph_ind;
if size(graph_ind,1) ~= num_nodes
    graph_ind = graph_ind';
end

% (n_node, 1) array, node labels
node_labels = A_full.node_labels;
if size(node_labels,1) ~= num_nodes
    node_labels = node_labels';
end

% (n_node, attr_dim), node attributes
node_attr = A_full.node_attr;
if size(node_attr,1) ~= num_nodes
    node_attr = node_attr';
end


%% Convert Txt files to Mat     
graph_ind = dlmread([path2data dataset '_graph_indicator.txt']);% (n_graph,1) indicator array: node are of which graph
num_nodes = size(graph_ind,1);

% edges, adjacency matrix
edges = dlmread([path2data dataset '_A.txt']);% (n_edge, 2) matrix, each row is ends of an edge
num_edges = size(edges,1);

% edge attributes (e.g. edge weights etc.)
try
    edge_attr = dlmread([path2data dataset '_edge_attributes.txt']);% (n_edge, 1) array, weight of edges
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
% Construct adjacency matrix
% A = sparse(edges(:,1), edges(:,2), edge_attr(:,1), num_nodes, num_nodes);
path2A = [path2data dataset '_adj_mat.mat'];
load(path2A)
% AA = sparse([edges(:,1), edges(:,2), edge_attr(:,1)]);

% node labels  
try 
    node_labels = dlmread([path2data dataset '_node_labels.txt']);% (n_node, 1) array, node labels
catch
    node_labels = full(sum(A, 2));   % set node label as its degree for unlabeled graph;
end

if size(node_labels,1) ~= num_nodes
    fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_labels.txt']);
end

% node attributes
try
    node_attr = dlmread([path2data dataset '_node_attributes.txt']);
    if size(node_attr,1) ~= num_nodes
        fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_attributes.txt']);
    end
end