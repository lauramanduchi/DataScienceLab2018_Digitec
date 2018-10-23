import networkx as nx
from string import digits

# params:       -filters:
#                   list of filters for the given category
#
# returns:      -graph:
#                   complete initialized graph
def createGraph(filters):
  ind = 0
  G = nx.DiGraph()
  #create the root
  G.add_node(0)
  recursiveGraph(G, 0, filters, 0)
  return G

# G.edges[3, 4]['weight'] = 4.2 to set different weight

def recursiveGraph(G, root, S, t):
    t = t+1
    for s in S:
        if root != 0:
            string = str(root) + " " + s
        else:
            string = s
        G.add_node(string)
        G.add_edge(root, string, weight=0)
        Snew = S.copy()
        Snew.remove(s)
        recursiveGraph(G, string, Snew, t)
    return G

# params:       -graph:
#                  
#               -root:
#                   reference string
#               -h:
#                 history of answers
#               -alpha:
#                 alpha parameter for tuning  
#
# returns:      -graph:
#                   with weights
def setWeights(G, root, h, alpha):
  A = G.neighbors(root)
  while True:
      try:
          # get the next item
          n = next(A)
          remove_digits = str.maketrans('', '', digits)
          questions = s.translate(remove_digits)
          questions.split(" ") #array of questions made

          #possible answers for last question [a1, a2, a3]
          last_q = questions[-1]
          ans = answers(last_q)
          w0 = 0
          for a in ans:
            w0 = w0 + len(product(a,h)) / len(product(h))    #PRODUCT(a) returns all the products given the selected answers

          w = (1/len(ans))*w0 + alpha*(users(h) / users(last_q, h))  #USERS(q,h) returns the number of users that answered h and use q
          G[root][n]['weight'] = w
          # do something with element
      except StopIteration:
          # if StopIteration is raised, break from loop
          break
  return G

def dijsktra(graph, initial):
  return nx.dijkstra_path(graph,0,4)



#TESTING
# filters = ["A", "B", "C"]
# G = createGraph(filters)
# nx.draw(G, with_labels = True)
# nx.get_node_attributes(G,'products')
# G.node["A"]


