% %
% Convert .txt files in matlab arrays. 
% %
function convertdata_txt2mat(dataset,path_to_data)

    %path2data = ['./path_to_data/' dataset '/'];
    path2data = [path_to_data '/' dataset '/'];

    % graph labels
    graph_labels = dlmread([path2data dataset '_graph_labels.txt']);
    num_graphs = size(graph_labels,1);

    graph_ind = dlmread([path2data dataset '_graph_indicator.txt']);
    num_nodes = size(graph_ind,1);

    % node labels
    node_labels = dlmread([path2data dataset '_node_labels.txt']);
    if size(node_labels,1) ~= num_nodes
        fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_labels.txt']);
    end

    % node attributes
    try
        node_attr = dlmread([path2data dataset '_node_attributess.txt']);
        if size(node_attr,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_attributes.txt']);
        end
    end

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
    A = sparse(edges(:,1), edges(:,2), edge_attr(:,1), num_nodes,num_nodes);
end




