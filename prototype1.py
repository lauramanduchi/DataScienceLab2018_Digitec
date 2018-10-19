import networkx as nx

filters = ["A", "B", "C", "D"]

# params:       -filters:
#                   list of filters for the given category
#
# returns:      -graph:
#                   complete initialized graph
def createGraph(filters):
  ind = 0
  G = nx.Graph()
  #create the root
  G.add_node(0)
  recursiveGraph(G, 0, filters, 0)
  return G

# G.edges[3, 4]['weight'] = 4.2 to set different weight

def recursiveGraph(G, root, S, t):
    t = t+1
    for s in S:
        if root != 0:
            string = str(root) + " " + s + str(t)
        else:
            string = s + str(t)
        G.add_node(string)
        G.add_edge(root, string)
        Snew = S.copy()
        Snew.remove(s)
        recursiveGraph(G, string, Snew, t)
    return G


# params:       -graph:
#                   list of source strings
#               -initial:
#                   reference string
#
# returns:      -score:
#                   score of the optimal alignment
def dijsktra_old(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path

  def dijsktra(graph, initial):
  	return nx.dijkstra_path(graph,0,4)