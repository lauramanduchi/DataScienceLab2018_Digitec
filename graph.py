class Vertex:
    def __init__(self, key, filter_name):
        self.id = key
        self.filter_name = filter_name
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getFilter_name(self):
        return self.filter_name

    def getWeight(self,nbr):
        return self.connectedTo[nbr]

    def setWeight(self,nbr):
        if nbr in self.connectedTo.keys():
          self.connectedTo[nbr] = weight


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key, filter_name):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key, filter_name)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f, f_filter_name, t, t_filter_name, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f, f_filter_name)
        if t not in self.vertList:
            nv = self.addVertex(t, t_filter_name)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        keys = self.vertList.keys()
        values = self.vertList.getFilter_name()
        return dict(zip(keys, values))

    def __iter__(self):
        return iter(self.vertList.values())