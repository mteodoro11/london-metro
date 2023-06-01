import numpy as np
import random
import math
from statistics import mean
import datetime
import csv

# creates a vertex object
class Vertex:

    def __init__(self, vertex):
        self.value = vertex

    def vertex(self):
        return self.value


# creates an edge object
class Edge:

    def __init__(self, start_vertex, end_vertex):
        self.start = start_vertex
        self.end = end_vertex
        self.vertices = [self.start, self.end]

    def start(self):
        return self.start

    def end(self):
        return self.end

    def vertices(self):
        return self.vertices


# creates a station object
class Station(Vertex):

    # saves id and position
    def __init__(self, vertex, id, position):
        super().__init__(vertex)
        self.id = id
        self.position = position


# creates object binary tree element
class BinaryTreeElement:

    def __init__(self, node, value, left = None, right = None):
        self._node = node
        self.left = left
        self.right = right
        self.value = value

    def __str__(self):
        return self._node


# creates binary tree structure
class BinaryTree:

    def __init__(self):
        self._root = None
        self._length = 0

    # looks up value in tree
    def search_value(self, root, value):
        if not root:
            return False
        if root._node == value:
            return root.value
        res = self.search_value(root.left, value)
        if res:
            return res
        res = self.search_value(root.right, value)
        if res:
            return res

    # adds tree root
    def add_root(self, root_value, value):
        root = BinaryTreeElement(root_value, value)
        if self._root is None:
            self._root = root
            self._length += 1
        else:
            raise ValueError('root already exists')

    # returns root
    def root(self):
        return self._root


# creates a binary research tree structure
class BinaryResearchTree(BinaryTree):

    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return self == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self < other

    def __le__(self, other):
        return self <= other

    def __gt__(self, other):
        return self > other

    def __ge__(self, other):
        return self >= other

    # inserts a node into the tree
    def insert(self, root, node_value, value):

        # root has to be self._root
        node = BinaryTreeElement(node_value, value)
        
        if node._node > root._node:
        
            if root.right:
                self.insert(root.right, node_value, value)
            else:
                root.right = node
                self._length += 1
        
        if node._node < root._node:
        
            if root.left:
                self.insert(root.left, node_value, value)
            else:
                root.left = node
                self._length += 1


# creates a class with a list of stations and a binary search tree to look it up
class StationList:

    # initiates station list from stations in file
    def __init__(self):

        self.station_list = []

        reader = csv.DictReader(open('london.stations.txt'))
        for row in reader:
            object = Station(row["name"], int(row["id"]), (float(row["latitude"]), float(row["longitude"])))
            self.station_list.append(object)

    # creates research tree based on stations
    def create_research_tree(self):
        self.find_station = BinaryResearchTree()
        random.shuffle(self.station_list)
        for station in self.station_list:
            if self.find_station._root is None:
                self.find_station.add_root(station.id, station.value)
            else:
                self.find_station.insert(self.find_station._root, station.id, station.value)


# creates an edge object and adapts it to represent a line between two stations
class EdgeLine(Edge):

    def __init__(self, start_vertex, end_vertex, line, time):
        super().__init__(start_vertex, end_vertex)
        self.line = [line]
        self.time = time


# creates a list with connections for graph partition
class ConnectionsPartition:

    def __init__(self):
        self.connections = []
        reader = csv.DictReader(open('london.connections.txt'))
        for row in reader:
            object = EdgeLine(int(row['station1']), int(row['station2']), row['line'], int(row['time']))
            self.connections.append(object)


# creates a class with used methods for graph partition
class GraphPartitioning:

    def __init__(self, initial_list):

        # creates list with connections
        self.connect = ConnectionsPartition()

        # creates stations list
        self.stations = StationList()

        # creates binary research tree for stations
        self.stations.create_research_tree()

        # starts a zeros matrix to represent the subway
        self.subway = np.zeros((self.stations.station_list.__len__(), self.stations.station_list.__len__()))

        # creates an index dictionary to find station ids within matrix
        self.indexes = {}  

        # associates ids to their matrix indexes
        for idx in range(1,189):  
            self.indexes[idx] = idx - 1
        for idx in range(190, 304):
            self.indexes[idx] = idx - 2

        # finds connections between stations within the matrix and marks them with the value 1
        for conn in self.connect.connections:  # grabs connections
            station1 = self.indexes[conn.start]  # finds id within matrix
            station2 = self.indexes[conn.end]
            self.subway[station1][station2] = 1  # finds connections within matrix and marks them with '1'
            self.subway[station2][station1] = 1
        
        # creates a dictionary to locate in which side of the partition is each vertex
        self.side_dictionary = {}

        # defines initial partition
        for idx in range(0, int(initial_list.__len__() / 2)):
            self.side_dictionary[initial_list[idx]] = 2
        for idx in range(int(initial_list.__len__() / 2), int(initial_list.__len__())):
            self.side_dictionary[initial_list[idx]] = 3
        
        # draws matrix with initial partition
        self.draw_matrix()

        # dictionary to store pairs "marked" by the algorithm
        self.marked = {}

        # marks all pairs as unvisited
        nrow, _ = self.subway.shape
        for idx in range(0, nrow):
            self.marked[idx] = 'No'

    # draws matrix according to current side dictionary
    def draw_matrix(self):
        nrow, ncol = self.subway.shape  # for each point in the matrix
        for row in range(0, nrow):
            for col in range(0, ncol):
                if self.side_dictionary[row] == self.side_dictionary[col]:  # if both points are on the same side
                    if self.subway[row][col] != 0:
                        self.subway[row][col] = self.side_dictionary[row]  # if they are connected, marked with its side, 2 or 3
                else:  # if they are in different partitions
                    if self.subway[row][col] != 0:
                        self.subway[row][col] = 1  # if they are connected, mark the cutting point with '1'

    # calculates D = external cost - interior cost of the vector, saves all D's
    def d_value(self):

        self.d_values = {}  # creates a dictionary with the D for each point
        nrow, ncol = self.subway.shape
        
        # for each row on the matrix
        for row in range(0, nrow):

            # if line isn't marked yet
            if self.marked[row] == 'No':
            
                current_count = 0
                for col in range(0, ncol):
            
                    if self.marked[col] == 'No':  # if column isn't marked

                        # if corresponds to one of the partitions -> discounts 1
                        if self.subway[row][col] == 2 or self.subway[row][col] == 3:
                            current_count -= 1
                        
                        # if corresponds to a cut -> increases 1
                        elif self.subway[row][col] == 1:
                            current_count += 1
                    
                    if self.marked[col] == 'Yes': # if column is marked, therefore will switch sides
                        
                        # if corresponds to one of the partitions -> will be cut -> +1
                        if self.subway[row][col] == 2 or self.subway[row][col] == 3:
                            current_count += 1
                        
                        # if it's cut -> will be part of same partition -> -1
                        elif self.subway[row][col] == 1:
                            current_count -= 1

                self.d_values[row] = current_count  # indexes D to each vector in dictionary

    # calculates g values
    def g_value(self):

        self.g_values = {}  # creates a dictionary for g values

        for idx1 in self.d_values.keys(): # goes through d values dictionary
            for idx2 in self.d_values.keys():
                if idx1 != idx2:

                    # if both points are on different sides and there is a connection among them
                    if self.side_dictionary[idx1] != self.side_dictionary[idx2] and self.subway[idx1][idx2] != 0:
                        # calculate g as the sum of d for both points minus 2
                        g = self.d_values[idx1] + self.d_values[idx2] - 2
                        self.g_values[idx1, idx2] = g  # associates the points g to its dictionary key
                    
                    # if both points are on different sides and there isn't a connection among them
                    elif self.side_dictionary[idx1] != self.side_dictionary[idx2] and self.subway[idx1][idx2] == 0:
                        # calculate g as the sum of d for both points
                        g = self.d_values[idx1] + self.d_values[idx2]  
                        self.g_values[idx1, idx2] = g  # associates the points g to its dictionary key

    # method applies the Kernighan-Lin algorithm
    def algorithm(self):
        
        self.d_value()  # calculates d values
        gain_list = []  # creates an empty gains list
        trade_list_a = []  # creates empty lists that will store points to be traded
        trade_list_b = []

        # while there is unmarked points
        while 'No' in self.marked.values():

            self.g_value()  # calculate g's
            m = max(self.g_values, key=self.g_values.get)  # grabs the key for the max g value
            # if max g value is larger or equal to 0
            if self.g_values[m] >= 0:
                p1, p2 = m  # splits key in the two points, and mark each
                self.marked[p1] = 'Yes'
                self.marked[p2] = 'Yes'
                gain_list.append(self.g_values[m])  # adds g to gains list
                trade_list_a.append(p1)  # adds the found points to the list of tradeables
                trade_list_b.append(p2)
                self.d_value()  # recalculates d values

            # when it finds a negative g -> breaks the circuit
            else:
                break

        # max_gain = 0  # store maximum gain
        # i = 0
        # # iterates gain list
        # while i != gain_list.__len__():
        #     max_gain += gain_list[i]
        #     i += 1

        max_gain = sum(gain_list)
        # shows max gain found
        print('maximum gain found:', max_gain)

        # if max gain is positive
        if max_gain > 0:
            current_idx = 0
            while current_idx != len(gain_list):  # trades all points present in the list of tradeables
                if self.side_dictionary[trade_list_a[current_idx]] == 2:
                    self.side_dictionary[trade_list_a[current_idx]] = 3
                elif self.side_dictionary[trade_list_a[current_idx]] == 3:
                    self.side_dictionary[trade_list_a[current_idx]] = 2
                if self.side_dictionary[trade_list_b[current_idx]] == 2:
                    self.side_dictionary[trade_list_b[current_idx]] = 3
                elif self.side_dictionary[trade_list_b[current_idx]] == 3:
                    self.side_dictionary[trade_list_b[current_idx]] = 2
                current_idx += 1
            self.draw_matrix() # readraws matrix with new trades
            self.marked = {}  # stores pairs marked by the algorithm
            nrow, _ = self.subway.shape
            for row in range(0, nrow):  # returns all pairs to unmarked
                self.marked[row] = 'No'
            self.algorithm()  # re-runs the algorithm

        # if max gain isn't positive, algorithm is over
        else:
            return None

    # calculates cut of the matrix
    def cut_(self):
        self.d_values = {}  # empties d values
        self.cut = 0  # creates an initial cut of 0
        self.cut_list = []  # creates an empty list of cut connections
        nrow, ncol = self.subway.shape
        for row in range(0, nrow):  # for each line on the matrix
            for col in range(0, ncol):  # for each column on the matrix
                if col < row:
                    if self.subway[row][col] == 1:  # if cut is found
                        self.cut += 1  # add to cut size
                        this_cut = []  # searches for what cut it is (ids)
                        for idx1, idx2 in self.indexes.items():  # looks in indexes
                            if idx2 == row:  # if found
                                find = (self.stations.find_station.search_value(self.stations.find_station._root, idx1))
                                this_cut.append(find)
                            elif idx2 == col:
                                find = (self.stations.find_station.search_value(self.stations.find_station._root, idx1))
                                this_cut.append(find)
                        self.cut_list.append(this_cut)  # adds cut to cut list
        for cut in self.cut_list:  # shows all cuts
            print(cut[0], ' - ', cut[1])
        print('-------------------------------------')
        print('CUT FOUND: ', self.cut)  # shows number of cuts
        return self.cut_list, self.cut

    # returns a list with the size of the different sections created by the partition
    def size_of_partitions(self):
        
        nrow, ncol = self.subway.shape

        # reset cuts to 0
        for row in range(0, nrow):
            for col in range(0, ncol):
                if self.subway[row][col] == 1:
                    self.subway[row][col] = 0
                elif self.subway[col][row] == 1:
                    self.subway[col][row] = 0
        
        # mark all connected points with 1
        for row in range(0, nrow):
            for col in range(0, ncol):
                if self.subway[row][col] == 2 or self.subway[row][col] == 3:
                    self.subway[row][col] = 1
                elif self.subway[col][row] == 2 or self.subway[col][row] == 3:
                    self.subway[col][row] = 1
        
        diagonal = np.sum(self.subway, axis=0)
        laplace = np.diag(diagonal) - self.subway
        ev = np.linalg.eig(laplace)
        sort_ev = sorted(ev[0])
        return sort_ev


# creates a list of connections to be used on the short path
class Connections:

    def __init__(self):

        reader = csv.DictReader(open('Insterstations v3.txt'))
        self.connections = []

        # present initial screen and ask for input
        print('At what time would you like to travel?')
        print("a. 7 a.m. - 10 a.m.     b. 10 a.m. - 4 p.m.     c. Other")
        hour = input()

        # load the data corresponding to the key given
        if hour == 'a' or hour == 'A':
            for row in reader:
                found = False
                for conn in self.connections:
                    if (conn.start == int(row['From Station Id']) and conn.end == int(row['To Station Id'])) or (conn.end == int(row['From Station Id']) and conn.start == int(row['To Station Id'])):
                        conn.line.append(row['Line'])
                        conn.time = ((conn.time * (conn.line.__len__() - 1)) + float(row['AM peak (0700-1000) Running Time (Mins)'])) / conn.line.__len__()
                        found = True
                if not found:
                    object = EdgeLine(int(row['From Station Id']), int(row['To Station Id']), row['Line'], float(row['AM peak (0700-1000) Running Time (Mins)']))
                    self.connections.append(object)
        
        elif hour == 'b' or hour == 'B':
            for row in reader:
                found = False
                for conn in self.connections:
                    if (conn.start == int(row['From Station Id']) and conn.end == int(row['To Station Id'])) or (conn.end == int(row['From Station Id']) and conn.start == int(row['To Station Id'])):
                        conn.line.append(row['Line'])
                        conn.time = (conn.time + float(row['Inter peak (1000 - 1600) Running time (mins)'])) / conn.line.__len__()
                        found = True
                if not found:
                    object = EdgeLine(int(row['From Station Id']), int(row['To Station Id']), row['Line'], float(row['Inter peak (1000 - 1600) Running time (mins)']))
                    self.connections.append(object)
        
        elif hour == 'c' or hour == 'C':
            for row in reader:
                found = False
                for conn in self.connections:
                    if (conn.start == int(row['From Station Id']) and conn.end == int(row['To Station Id'])) or (conn.end == int(row['From Station Id']) and conn.start == int(row['To Station Id'])):
                        conn.line.append(row['Line'])
                        conn.time = (conn.time + float(row['Off Peak Running Time (mins)'])) / conn.line.__len__()
                        found = True
                if not found:
                    object = EdgeLine(int(row['From Station Id']), int(row['To Station Id']), row['Line'], float(row['Off Peak Running Time (mins)']))
                    self.connections.append(object)
        
        # if key non-existent, raise error
        else:
            raise ValueError('Key does not exist')


# creates an object that will store the station and the time it takes to switch lines in it
class LineChangeTime:

    def __init__(self, station, time):
        self.current_station = station
        self.average_time = time


# creates a list of objects LineChangeTime
class ChangeLineTimes:

    def __init__(self):
        self.change_lines = {}
        reader = csv.DictReader(open('10PL_Trade Line Times.txt'))
        for row in reader:
            object = LineChangeTime(row["Station"], float(row["Average Time"]))
            self.change_lines[object.current_station] = object.average_time


# creates an object that associates a line with its average platform waiting time
class WaitTimePlatform:

    def __init__(self, line, time):
        self.line = line
        self.time = time


# creates a list of objects WaitTimePlatform
class PlatformWaitTimes:

    def __init__(self):
        self.platform_wait = {}
        reader = csv.DictReader(open('10PL_Wait Platform Time.txt'))
        for row in reader:
            object = WaitTimePlatform(row["Line ID"], float(row["Average Time"]))
            self.platform_wait[object.line] = object.time


# class with methods used to calculate shortest path
class ShortestPath:

    def __init__(self):
        
        self.connect = Connections() # gets connections list
        self.stations = StationList() # gets stations list
        self.wait = PlatformWaitTimes()  # gets platform wait times list
        self.change = ChangeLineTimes()  # gets change line times list

        # removes all edges that have no connections (such as DLR stations for which we have no data)
        for station in self.stations.station_list:  
            found = False
            while not found:
                for conn in self.connect.connections:
                    if station.id == conn.start or station.id == conn.end:
                        found = True
                if not found:
                    self.stations.station_list.remove(station)
                    found = True

        self.stations.create_research_tree()  # creates binary search tree
        self.subway = np.zeros((self.stations.station_list.__len__(), self.stations.station_list.__len__())) # creates zeros matrix to represent subway
        
        self.indexes = {}  # creates indexes dictionary to find station ids within matrix
        idx = 0
        nrow, ncol = self.subway.shape
        while idx != nrow:
            for station in self.stations.station_list:
                self.indexes[station.id] = idx
                idx += 1
        
        # finds connections between stations on the matrix
        for conn in self.connect.connections:  # grabs connections
            s1 = self.indexes[conn.start]  # finds id in matrix
            s2 = self.indexes[conn.end]
            self.subway[s1][s2] = conn.time  # finds connections in matrix and marks with time between stations
            self.subway[s2][s1] = conn.time

        self.visited = {}  # saves pairs marked by the algorithm
        nrow, _ = self.subway.shape
        for idx in range(0, nrow):  # stores all pairs as unmarked
            self.visited[nrow] = 'No'
        
        # creates empty dictionary to store distances
        self.distance = {}
        # creates empty dictionary to look for next node
        self.next_node = {}
        # creates empty dictionary to store paths
        self.paths = {}
        # creates dictionary whose keys will be station ids and values will be corresponding metro lines
        self.index_lines = {}
        for nrow, ncol in self.indexes.items():
            self.index_lines[ncol] = []
            for conn in self.connect.connections:
                if conn.start == nrow or conn.end == nrow:
                    for line in conn.line:
                        if line not in self.index_lines[ncol]:
                            self.index_lines[ncol].append(line)
    
        # calculates average time it takes to switch lines, for stations that aren't in the dataset
        # sum = 0
        # count = 0
        # for i in self.change.change_lines.values():
        #     sum += i
        #     count += 1
        # self.average_trade_time = sum / count

        self.average_trade_time = mean(self.change.change_lines.values())

    # calculates distances from unmarked nodes to initial point to choose next
    def refresh_next_node(self):
        self.next_node = {}
        for key in self.visited.keys():
            if self.visited[key] == 'No':
                self.next_node[key] = self.distance[key]

    # finds shortest path with dijkstra algorithm 
    def shortest_path(self):

        print('This program does not include the Docklands Light Railway line.')
        start_station = input('Choose starting station: ')  # ask for input of starting station
        end_station = input('Choose ending station: ')  # ask for input of final station
        
        start = None
        end = None
        
        # looks up id for both
        for station in self.stations.station_list:
            if station.value == start_station:
                start = self.indexes[station.id]
            elif station.value == end_station:
                end = self.indexes[station.id]  
        
        # if it fails to do so -> raises an error
        if start is None:
            raise ValueError('Could not find the starting station you introduced')
        if end is None:
            raise ValueError('Could not find the starting station you introduced')
        
        # define distance of all points to initial as infinite
        nrow, ncol = self.subway.shape
        for idx in range(0, nrow):
            self.distance[idx] = math.inf
        
        # defines distance from initial point as waiting time in current line (or the average of many, in case station has more than one)
        list = []
        for idx in self.index_lines[start]:
            list.append(self.wait.platform_wait[idx])
        self.distance[start] = round(mean(list), 3)
        
        # creates dictionary to store paths
        self.paths = {}
        for idx in range(0, nrow):
            self.paths[idx] = []

        # defines initial path as only the starting station
        self.paths[start] = [start]

        # creates dictionary to store in which line each path is on
        self.path_line = {}
        for idx in range(0, nrow):
            self.path_line[idx] = None

        # looks up current line
        self.path_line[start] = self.index_lines[start]

        # set starting point as current
        current = start

        # while end point is not visited
        while self.visited[end] == 'No':

            # if current point has not been visited yet
            if self.visited[current] == 'No':

                # look for current point connections
                for col in range(0, ncol):

                    # if connection hasn't been visited and there is a connection
                    if self.visited[col] == 'No' and self.subway[current][col] != 0:
                        
                        current_set = set(self.path_line[current])
                        col_set = set(self.index_lines[col])

                        # if station on the same line is found
                        if len(current_set.intersection(col_set)) > 0:

                            if self.distance[col] > self.subway[current][col] + self.distance[current]:

                                # replaces current distance by distance through new path
                                self.distance[col] = self.subway[current][col] + self.distance[current]

                                # adds current station to current path
                                self.paths[col] = self.paths[current] + [col]

                                # defines current line as the one coming from the previous point
                                self.path_line[col] = self.path_line[current]
                        
                        # if station that requires switching line is found
                        else:

                            # if next point isn't on the same line -> find common line for both
                            common_line = []
                            for line_col in self.index_lines[col]:
                                for line_current in self.index_lines[current]:
                                    if line_col == line_current:
                                        common_line = line_current
                            # grabs line switching time on current station
                            trade_time = self.average_trade_time
                            for id, idx in self.indexes.items():
                                if idx == current:
                                    current_station = self.stations.find_station.search_value(self.stations.find_station._root, id)
                                    print('CURRENT STATION IS THIS:', current_station)
                                    if current_station in self.change.change_lines.keys():
                                        trade_time = self.change.change_lines[current_station]
                                    # if trade time is not on the dataset, assume as average time
                                    else:
                                        trade_time = self.average_trade_time

                            if self.distance[col] > self.subway[current][col] + self.distance[current] + self.wait.platform_wait[common_line] + trade_time:
                                
                                # replaces current distance by distance through new path
                                self.distance[col] = self.subway[current][col] + self.distance[current] + self.wait.platform_wait[common_line] + trade_time
                                
                                # adds current station to current path
                                self.paths[col] = self.paths[current] + [col]

                                # defines current line as the common line between both stations
                                self.path_line[col] = common_line

                # marks point as visited
                self.visited[current] = 'Yes'
                # looks for point with smallest distance
                self.refresh_next_node()
                current = min(self.next_node, key=self.next_node.get)

        # store list with names of stations in the shortest path
        list = []
        for idx in self.paths[end]:
            for station_id, list_idx in self.indexes.items():
                if list_idx == idx:
                    value = self.stations.find_station.search_value(self.stations.find_station._root, station_id)
                    list.append(value)
        
        # shows path
        print()
        print('Shortest path:')
        for station in list:
            if station != list[-1]:
                print(station, end=' -> ')
            else:
                print(station)

        # returns time that shortest path takes
        time = round((self.distance[end] * 60), 0) # transform minutes to seconds
        time = str(datetime.timedelta(seconds=time)) # transform seconds to "hours : minutes : seconds" format
        print('Shortest path takes', time, 'minutes')

        # empty everything used after finding shortest path
        self.visited = {}
        nrow, ncol = self.subway.shape
        for row in range(0, nrow):
            self.visited[row] = 'No'
        self.distance = {}
        self.next_node = {}
        self.paths = {}
        
        # ask user if it wants to find another shortest path
        print('')
        print('Would you like to see another shortest path or leave?')
        print('a - shortest path   //   any other key - leave')
        f = input()
        if f == 'a' or f == 'A':
            return self.shortest_path()
        else:
            return None


# function that develops the shortest path method
def shortest_path():
    metro = ShortestPath()
    metro.shortest_path()


# function that develops the graph partition (thousand iterations)
def graph_partitioning():
    list = []
    for idx in range(0, 302):
        list.append(idx)
    # initiate partition
    random.shuffle(list)
    minimum_cut = math.inf
    minimum_cuts_list = []

    for i in range(0, 1000):
        print('iteration number: ', i)
        print('current minimum cut found: ', minimum_cut)
        for j in minimum_cuts_list:
            print('minimum cut lists found: ', j)
        random.shuffle(list)
        print('shuffling initial list')
        print('creating subway')
        metro = GraphPartitioning(list)
        print('applying the kerninghan lin algorithm..........')
        metro.algorithm()
        print('-------------------------------------')
        metro.cut_()
        print('-------------------------------------')
        if metro.cut < minimum_cut:
            minimum_cut = metro.cut
            minimum_cuts_list = [[metro.cut_list]]
        elif metro.cut == minimum_cut:
            minimum_cuts_list.append(metro.cut_list)
    
    print('after 1000 iterations:')
    print('minimum cut: ', minimum_cut)
    
    for min_cut in minimum_cuts_list:
        print('minimum cut list: ', min_cut)

